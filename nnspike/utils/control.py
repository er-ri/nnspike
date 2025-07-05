"""
ラインフォロワー制御モジュール

このモジュールは、カメラ画像を用いたラインフォロー制御アルゴリズムを実装します。
主に学習データ収集やリアルタイムロボット制御で利用されます。

クラス:
    ControlCalculator:
        画像の中心とラインのオフセットから進行角度（theta）を計算し、
        平滑化や速度調整、パワー差分の計算を行います。
"""

import math
import time
from collections import deque
import numpy as np

# ====【現場でよく調整する推奨パラメータ】====
BASE_POWER = 30             # 通常走行時の基準パワー
STRAIGHT_POWER = 50         # 直線判定時のパワー
CURVE_POWER = 25            # カーブ判定時のパワー（20→25で復帰力UP）

# --- 追加: しきい値・閾値のグローバル定数定義 ---
STRAIGHT_THRESHOLD_DEG = 3   # 直線判定しきい値[deg]
CURVE_THRESHOLD_DEG = 10     # カーブ判定しきい値[deg]

CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # ピクセル単位のカメラ焦点距離（概算値、キャリブレーション推奨）
WHEELBASE = 0.10  # Distance between wheels in meters

MAX_THETA_DEG = 30          # θの最大値[deg]（直線・90度カーブの挙動は維持しつつ、限界付近でパワー差を最大化）
MAX_POWER_DIFF = 40         # PID補正による最大パワー差分（限界付近でのパワー差を強調）

# ---【ライン検出Y座標（カメラ画像基準, camera.pyのOFFSET_Yと揃える）】---
# ライントレース時に進行方向の基準とする画像内Y座標。
# camera.pyのOFFSET_Yと同じ値・意味で統一すること。
OFFSET_Y = 400  # 例: 画像下部付近を基準にする場合

# ---【ダブルループ交差点判定用の相対位置しきい値】---
# run_opencv.py など他ファイルと値を揃えること
POSITION_STRAIGHT = 4500
POSITION_CROSS1 = 11800
POSITION_CROSS2 = 15500
POSITION_CROSS3 = 18500
POSITION_CROSS4 = 20000

class ControlCalculator:
    """
    ラインフォロワー用制御計算クラス。
    """

    def __init__(self, image_width, image_height, debug=True, roi_opencv=None):
        """
        ControlCalculatorの初期化。

        引数:
            image_width: int 画像幅
            image_height: int 画像高さ
            debug: bool デバッグ出力ON/OFF
            roi_opencv: tuple ROI領域 (x, y, w, h)。未指定時はデフォルト値を使用。
        """
        # 制御パラメータの初期化（使用順に並べ替え）
        self.image_width = image_width  # 画像幅
        self.image_height = image_height  # 画像高さ
        self.base_power = BASE_POWER  # 通常時の基準パワー
        self.straight_power = STRAIGHT_POWER  # 直線時の推奨パワー
        self.curve_power = CURVE_POWER  # カーブ時の推奨パワー
        self.straight_threshold = math.radians(STRAIGHT_THRESHOLD_DEG)  # 直線判定しきい値[rad]
        self.curve_threshold = math.radians(CURVE_THRESHOLD_DEG)  # カーブ判定しきい値[rad]
        self.max_theta = math.radians(MAX_THETA_DEG)  # θ最大値[rad]
        self.debug = debug  # デバッグ出力ON/OFF
        self.roi_opencv = roi_opencv
        
        # 段階的ブースト用の状態管理
        self._boost_start_time = None  # ブースト開始時刻
        self._boost_buildup_duration = 1.0  # ブーストが1.2倍に達するまでの時間（秒）
        self._theta_5deg_start_time = None  # theta > 8度状態の開始時刻（5度→8度に変更で安定化）
        self._theta_5deg_duration_threshold = 0.3  # theta > 8度が継続する必要な時間（秒、0.5s→0.3sでさらに緩和）
        self._boost_active = False  # ブースト状態フラグ（ログ出力用）
        self._boost_cooldown_time = None  # ブースト停止後のクールダウン開始時刻
        self._boost_cooldown_duration = 1.0  # クールダウン期間（3秒→1秒に短縮）
        self._theta_cumulative_time = 0.0  # theta > 8度の累積時間
        self._theta_cumulative_threshold = 0.15  # 累積時間のしきい値（0.3秒→0.15秒に大幅緩和）
        self._last_debug_time = 0  # デバッグ出力頻度制御用
        self._last_boost_debug_time = 0  # ブーストデバッグ出力頻度制御用

    @property
    def is_boost_active(self):
        """ブースト状態が有効かどうかを返す"""
        return self._boost_active

    def calc_steer_result(self, left_x, right_x, position=None):
        """
        ライントレース用の制御・可視化に必要な値を計算し、steer_result辞書を返す。

        Args:
            left_x (int or None): ライン左端のx座標（ピクセル）。Noneの場合は未検出。
            right_x (int or None): ライン右端のx座標（ピクセル）。Noneの場合は未検出。
            position (int or None): モーター相対位置（deg単位）。エッジ自動切替用。未使用時はNone。

        Returns:
            dict: steer_result（mx, my, offset_pixels, max_contour, roi_type, follow_edge, position など）
        """

        x1, y1, x2, y2 = self.roi_opencv
        # follow_edge自動判定: positionが指定されていれば区間ごとに切り替え、なければleft
        if position is not None:
            abs_position = abs(position)
            if abs_position <= POSITION_CROSS1:
                follow_edge = "right"
            elif abs_position <= POSITION_CROSS2:
                follow_edge = "left"
            else:
                follow_edge = "right"
        else:
            follow_edge = "left"

        if not hasattr(self, '_last_mx'):
            self._last_mx = None
            self._last_my = None

        # ラインが見つかった場合は常に最新のcandidate_xを目標点とし、絶対に離さない
        if left_x is not None and right_x is not None:
            min_x = x1 + int((x2 - x1) * 0.2)
            max_x = x1 + int((x2 - x1) * 0.8)
            if follow_edge == "left":
                candidate_x = np.clip(left_x, min_x, max_x)
                max_contour = np.array([[[candidate_x - x1, OFFSET_Y - y1]], [[(x2 + x1)//2 - x1, OFFSET_Y - y1]]], dtype=np.int32)
            elif follow_edge == "right":
                candidate_x = np.clip(right_x, min_x, max_x)
                max_contour = np.array([[[((x2 + x1)//2) - x1, OFFSET_Y - y1]], [[candidate_x - x1, OFFSET_Y - y1]]], dtype=np.int32)
            else:
                candidate_x = np.clip((left_x + right_x) // 2, min_x, max_x)
                max_contour = np.array([[[np.clip(left_x, min_x, max_x) - x1, OFFSET_Y - y1]], [[np.clip(right_x, min_x, max_x) - x1, OFFSET_Y - y1]]], dtype=np.int32)
            mx = candidate_x - x1
            my = OFFSET_Y - y1
            self._last_mx = mx
            self._last_my = my
            roi_center_x = (x2 - x1) // 2
            offset_pixels = mx - roi_center_x
        else:
            # ラインが消えても直前の点を絶対に維持（中央や初期値に戻さない）
            if self._last_mx is not None and self._last_my is not None:
                mx = self._last_mx
                my = self._last_my
                roi_center_x = (x2 - x1) // 2
                offset_pixels = mx - roi_center_x
                max_contour = None
            else:
                mx = (x2 - x1) // 2
                my = (y2 - y1) // 2
                offset_pixels = 0
                max_contour = None
        steer_result = {
            "mx": mx,
            "my": my,
            "offset_pixels": offset_pixels,
            "max_contour": max_contour,
            "roi_type": "opencv",
            "follow_edge": follow_edge,
            "position": position
        }
        if self.debug:
            debug_data = {
                "mx": mx,
                "my": my,
                "offset_pixels": offset_pixels,
                "roi_type": "opencv",
                "follow_edge": follow_edge,
                "position": position
            }
        return steer_result

    def calculate_attitude_angle(self, offset_pixels: float) -> float:
        """
        ピクセルオフセットからカメラ幾何を用いて姿勢角（theta）を計算する。
        グローバル定数CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PIXELSを使用。

        引数:
            offset_pixels (float): 画像中心からの横方向オフセット（ピクセル）
            roi (tuple): ROI領域 (x, y, w, h)。y2はy+hとして計算。
        戻り値:
            float: 姿勢角theta[rad]（右ズレ正、左ズレ負）
        注意:
            カメラパラメータはロボットごとに要キャリブレーション。
        使用例:
            roi = (x, y, w, h)
            theta = self.calculate_attitude_angle(offset_pixels, roi)
        """
        x1, y1, x2, y2 = self.roi_opencv

        ground_distance = (
            CAMERA_HEIGHT * CAMERA_FOCAL_LENGTH_PIXELS / (self.image_height - y2)
        )
        lateral_offset_meters = offset_pixels * ground_distance / CAMERA_FOCAL_LENGTH_PIXELS
        theta = math.atan2(lateral_offset_meters, ground_distance)
        if self.debug:
            debug_data = {
                "offset_pixels": offset_pixels,
                "roi_y2": y2,
                "image_height": self.image_height,
                "ground_distance": round(ground_distance, 4),
                "lateral_offset_meters": round(lateral_offset_meters, 4),
                "theta_rad": round(theta, 6),
                "theta_deg": round(math.degrees(theta), 3)
            }
        return theta

    def calculate_adaptive_speed(self, theta, position=None):
        """
        POSITION_STRAIGHTまではストレートパワー、それ以外はベースパワー。
        カーブパワーはbase_powerと同じなので条件式を統合。
        """
        if position is not None:
            abs_position = abs(position)
            if abs_position <= POSITION_STRAIGHT:
                return self.straight_power
            else:
                return self.base_power
        else:
            return self.base_power

    def calculate_power_adjustment(self, pid_corrected_theta, position=None):
        """
        PID補正値（ラジアン）をパワー差分（左右モーター出力の調整値）に変換する。
        - シンプルなリニア変換のみ（ブーストなし）
        - theta_degが8度を超えた状態が1.0秒以上続いた場合、パワー差分を段階的に1.2倍までブースト
        - theta_degが8度以下になると即座にブースト停止
        - 最大パワー差分はMAX_POWER_DIFFでクリップ
        """
        base = (pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF
        
        # theta_degが8度を超えた状態が1.0秒以上続く場合の段階的ブースト処理
        theta_deg = abs(math.degrees(pid_corrected_theta))
        current_time = time.time()
        
        # boost factorの計算（デバッグ表示用）
        current_boost_factor = 1.0
        
        if theta_deg > 8.0:  # 8度を超えた場合にブースト処理開始（5度→8度でより厳格に）
            # 累積時間を増加（フレーム間の時間差を加算）
            if hasattr(self, '_last_frame_time'):
                frame_delta = current_time - self._last_frame_time
                self._theta_cumulative_time += frame_delta
            self._last_frame_time = current_time
            
            # クールダウン中かチェック
            if self._boost_cooldown_time is not None:
                cooldown_elapsed = current_time - self._boost_cooldown_time
                if cooldown_elapsed < self._boost_cooldown_duration:
                    # まだクールダウン中：ブースト無効
                    if self.debug and (current_time - self._last_debug_time) >= 1.0:
                        pos_str = f"{position:>6}" if position is not None else "  None"
                        print(f"[DEBUG] pos={pos_str} | Cooldown: theta={theta_deg:.1f}deg | remaining={self._boost_cooldown_duration - cooldown_elapsed:.1f}s")
                        self._last_debug_time = current_time
                    # クールダウン中はブースト開始しない
                    pass
                else:
                    # クールダウン終了
                    self._boost_cooldown_time = None
                    if self.debug:
                        pos_str = f"{position:>6}" if position is not None else "  None"
                        print(f"[DEBUG] pos={pos_str} | Cooldown ended: theta={theta_deg:.1f}deg")
            
            # クールダウン中でなければ通常のブースト処理
            if self._boost_cooldown_time is None:
                # theta > 8度の状態
                if self._theta_5deg_start_time is None:
                    # theta > 8度状態の開始
                    self._theta_5deg_start_time = current_time
                    if self.debug and (current_time - self._last_debug_time) >= 0.5:
                        pos_str = f"{position:>6}" if position is not None else "  None"
                        print(f"[DEBUG] pos={pos_str} | theta > 8deg started: {theta_deg:.1f}deg (need 0.3s for boost)")
                        self._last_debug_time = current_time
                else:
                    # theta > 8度状態が継続中
                    theta_5deg_elapsed = current_time - self._theta_5deg_start_time
                    if self.debug and (current_time - self._last_debug_time) >= 0.3:
                        pos_str = f"{position:>6}" if position is not None else "  None"
                        print(f"[DEBUG] pos={pos_str} | theta > 8deg continues: {theta_deg:.1f}deg, elapsed={theta_5deg_elapsed:.1f}s")
                        self._last_debug_time = current_time
                
                # theta > 8度が0.5秒以上継続、または累積0.3秒以上でブースト開始
                theta_5deg_elapsed = current_time - self._theta_5deg_start_time
                if theta_5deg_elapsed >= 0.3 or self._theta_cumulative_time >= self._theta_cumulative_threshold:  # 継続0.3秒 OR 累積0.15秒
                    # ブーストが必要な状態
                    if self._boost_start_time is None:
                        # ブースト開始
                        self._boost_start_time = current_time
                        current_boost_factor = 1.0
                        if not self._boost_active:
                            pos_str = f"{position:>6}" if position is not None else "  None"
                            trigger_reason = f"continuous {theta_5deg_elapsed:.1f}s" if theta_5deg_elapsed >= 0.3 else f"cumulative {self._theta_cumulative_time:.1f}s"
                            print(f"[BOOST] pos={pos_str} | Started: theta={theta_deg:.1f}deg ({trigger_reason}) | cooldown was {self._boost_cooldown_duration}s")
                            self._boost_active = True
                            self._last_debug_time = current_time
                    else:
                        # ブースト継続中：時間経過に応じて段階的に増加
                        elapsed_time = current_time - self._boost_start_time
                        boost_progress = min(elapsed_time / self._boost_buildup_duration, 1.0)
                        current_boost_factor = 1.0 + (0.2 * boost_progress)  # 1.0から1.2に段階的に増加（1.3→1.2でより穏やか）
                        if self.debug and (current_time - self._last_boost_debug_time) >= 0.3:  # 0.3秒ごとにブースト状態ログ出力
                            pos_str = f"{position:>6}" if position is not None else "  None"
                            print(f"[BOOST] pos={pos_str} | Active: factor={current_boost_factor:.2f} | theta={theta_deg:.1f}deg | elapsed={elapsed_time:.1f}s")
                            self._last_boost_debug_time = current_time
                    
                    base = base * current_boost_factor
                else:
                    # まだ0.3秒経過していない かつ 累積時間も不足（出力頻度抑制）
                    if self.debug and (current_time - self._last_debug_time) >= 0.3:  # 0.5秒→0.3秒でより頻繁に出力
                        pos_str = f"{position:>6}" if position is not None else "  None"
                        print(f"[DEBUG] pos={pos_str} | Waiting for boost: theta={theta_deg:.1f}deg, elapsed={theta_5deg_elapsed:.1f}s, cumulative={self._theta_cumulative_time:.1f}s (need 0.3s OR 0.15s)")
                        self._last_debug_time = current_time
        else:
            # theta_degが8度以下：累積時間をリセット
            self._theta_cumulative_time = 0.0
            if hasattr(self, '_last_frame_time'):
                self._last_frame_time = current_time
            
            # theta_degが8度以下：即座にブースト停止とtheta > 8度状態のリセット
            if self._boost_active:
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[BOOST] pos={pos_str} | Stopped: theta={theta_deg:.1f}deg (<=8deg) | Starting cooldown")
                self._boost_active = False
                self._boost_cooldown_time = current_time  # クールダウン開始
                self._last_debug_time = current_time
                self._last_boost_debug_time = current_time
            elif self._theta_5deg_start_time is not None:
                # ブーストは開始していないが、theta > 8度状態がリセットされる場合
                theta_5deg_elapsed = current_time - self._theta_5deg_start_time
                if self.debug and theta_5deg_elapsed > 0.1:  # 0.1秒以上継続していた場合のみログ出力
                    pos_str = f"{position:>6}" if position is not None else "  None"
                    print(f"[DEBUG] pos={pos_str} | theta <= 8deg: {theta_deg:.1f}deg | Reset timer (was {theta_5deg_elapsed:.1f}s, cumulative was {self._theta_cumulative_time:.1f}s)")
                    self._last_debug_time = current_time
            self._boost_start_time = None
            self._theta_5deg_start_time = None
        
        # デバッグ：position、theta値、boost状態、factor値を整列表示（通常は2.0秒ごと、ブースト時は頻繁に）
        if self._boost_active:
            # ブースト中はより詳細な情報を0.5秒ごとに表示
            if self.debug and (current_time - self._last_debug_time) >= 0.5:
                pos_str = f"{position:>6}" if position is not None else "  None"
                boost_str = "BOOST"
                print(f"[DEBUG] pos={pos_str} | theta={theta_deg:>5.1f}deg | {boost_str} | factor={current_boost_factor:.2f}")
                self._last_debug_time = current_time
        else:
            # 通常時は2.0秒ごとに表示
            if self.debug and (current_time - self._last_debug_time) >= 2.0:
                pos_str = f"{position:>6}" if position is not None else "  None"
                boost_str = "NORM "
                print(f"[DEBUG] pos={pos_str} | theta={theta_deg:>5.1f}deg | {boost_str} | factor={current_boost_factor:.2f}")
                self._last_debug_time = current_time
        
        if base > 0:
            power_adj = min(int(base), MAX_POWER_DIFF)
        else:
            power_adj = max(int(base), -MAX_POWER_DIFF)
        return power_adj
