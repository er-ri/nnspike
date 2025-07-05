"""
ラインフォロワー制御モジュール

このモジュールは、カメラ画像を用いたラインフォロー制御アルゴリズムを実装します。
主に学習データ        # 段階的ブースト用の状態管理
        self._boost_start_time = None  # ブースト開始時刻
        self._boost_buildup_duration = 1.0  # ブーストが1.2倍に達するまでの時間（秒）
        self._theta_5deg_start_time = None  # theta > 8度状態の開始時刻（5度→8度に変更で安定化）
        self._theta_5deg_duration_threshold = 0.2  # theta > 8度が継続する必要な時間（秒、0.3s→0.2sでさらに緩和）
        self._boost_active = False  # ブースト状態フラグ（ログ出力用）
        self._boost_cooldown_time = None  # ブースト停止後のクールダウン開始時刻
        self._boost_cooldown_duration = 0.2  # クールダウン期間（1秒→0.2秒に大幅短縮）
        self._theta_cumulative_time = 0.0  # theta > 8度の累積時間
        self._theta_cumulative_threshold = 0.1  # 累積時間のしきい値（0.15秒→0.1秒にさらに緩和）
        self._boost_min_duration = 0.8  # ブースト最低継続時間（0.3秒→0.8秒に延長で安定化）ット制御で利用されます。

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
CURVE_POWER = 15            # カーブ判定時のパワー（20→15にさらに下げて制御重視）

# --- 追加: しきい値・閾値のグローバル定数定義 ---
STRAIGHT_THRESHOLD_DEG = 3   # 直線判定しきい値[deg]
BOOST_AND_CURVE_THRESHOLD_DEG = 8  # ブースト発動+カーブパワー切替の統一しきい値[deg]（12→8に下げて早期制御）

CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # ピクセル単位のカメラ焦点距離（概算値、キャリブレーション推奨）
WHEELBASE = 0.10  # Distance between wheels in meters

MAX_THETA_DEG = 30          # θの最大値[deg]（直線・90度カーブの挙動は維持しつつ、限界付近でパワー差を最大化）
MAX_POWER_DIFF = 40         # PID補正による最大パワー差分（40で固定、正しい値）

# ---【ライン検出Y座標（カメラ画像基準, camera.pyのOFFSET_Yと揃える）】---
# ライントレース時に進行方向の基準とする画像内Y座標。
# camera.pyのOFFSET_Yと同じ値・意味で統一すること。
OFFSET_Y = 400  # 例: 画像下部付近を基準にする場合

# ---【ダブルループ交差点判定用の相対位置しきい値】---
# run_opencv.py など他ファイルと値を揃えること
POSITION_STRAIGHT = 4500
POSITION_CROSS1 = 11800
POSITION_CROSS2 = 15200
POSITION_CROSS3 = 17500
POSITION_CROSS4 = 19000

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
        self.boost_and_curve_threshold = math.radians(BOOST_AND_CURVE_THRESHOLD_DEG)  # ブースト+カーブ判定しきい値[rad]
        self.max_theta = math.radians(MAX_THETA_DEG)  # θ最大値[rad]
        self.debug = debug  # デバッグ出力ON/OFF
        self.roi_opencv = roi_opencv
        
        # シンプルなブースト判定用
        self._current_theta_deg = 0.0  # 現在のtheta値（度）
        self._last_boost_log_time = 0  # ブーストログ出力の時間制御用
        self._last_boost_factor = 1.0  # 前回のブーストファクター（急激変化検出用）
        
        # 急激な変化の監視用
        self._last_power = None  # 前回のパワー値
        self._last_theta_deg = 0.0  # 前回のtheta値（度）
        self._last_power_adjustment = 0  # 前回のパワー調整値
        
        # 制御統計情報
        self._boost_count = 0  # ブースト発動回数
        self._curve_power_count = 0  # カーブパワー発動回数
        self._total_calls = 0  # 制御計算回数
        self._last_stats_time = time.time()  # 統計情報表示時刻

    @property
    def is_boost_active(self):
        """実際のブーストファクターが1.0より大きい場合にブースト状態と判定（ヒステリシス考慮）"""
        return self._last_boost_factor > 1.0

    @property
    def current_boost_factor(self):
        """現在適用されているブーストファクターを返す"""
        return self._last_boost_factor
    
    @property
    def current_theta_deg(self):
        """現在のtheta値（度）を返す"""
        return self._current_theta_deg

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
            elif abs_position <= POSITION_CROSS3:
                follow_edge = "right"
            elif abs_position <= POSITION_CROSS4:
                # POSITION_CROSS3以降は左エッジ（難所エリア対応）
                follow_edge = "left"
            else:
                # POSITION_CROSS4以降は右エッジ（最終エリア対応）
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
            line_status = "DETECTED"
        else:
            # ラインが消えても直前の点を絶対に維持（中央や初期値に戻さない）
            if self._last_mx is not None and self._last_my is not None:
                mx = self._last_mx
                my = self._last_my
                roi_center_x = (x2 - x1) // 2
                offset_pixels = mx - roi_center_x
                max_contour = None
                line_status = "LOST_HOLDING"
            else:
                mx = (x2 - x1) // 2
                my = (y2 - y1) // 2
                offset_pixels = 0
                max_contour = None
                line_status = "LOST_CENTER"
        
        # ライン検出状況を定期的にログ出力
        if hasattr(self, '_last_steer_debug'):
            if time.time() - self._last_steer_debug >= 3.0:
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[STEER] pos={pos_str} | {line_status} | mx={mx:>3} offset={offset_pixels:>3}px | edge={follow_edge}")
                self._last_steer_debug = time.time()
        else:
            self._last_steer_debug = time.time()
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
        極端にthetaが大きいとき（20度以上）はカーブパワーに切り替える。
        """
        # theta値をdegreeに変換
        theta_deg = abs(math.degrees(theta)) if theta is not None else 0.0
        
        # 制御統計情報の更新
        self._total_calls += 1
        
        # 急激な変化の検出（BOOST_AND_CURVE_THRESHOLD_DEGを基準とした適度な値）
        theta_change = abs(theta_deg - self._last_theta_deg)
        if theta_change > BOOST_AND_CURVE_THRESHOLD_DEG:  # 統一閾値以上の急激な変化
            pos_str = f"{position:>6}" if position is not None else "  None"
            print(f"[STABILITY_ALERT] pos={pos_str} | SUDDEN_THETA_CHANGE | {self._last_theta_deg:>5.1f}deg → {theta_deg:>5.1f}deg (Δ{theta_change:>5.1f})")
        
        # theta値が大きい場合（BOOST_AND_CURVE_THRESHOLD_DEG度以上）はカーブパワー+ブースト同時発動
        if theta_deg >= BOOST_AND_CURVE_THRESHOLD_DEG:
            self._curve_power_count += 1
            # カーブパワー+ブースト同時発動時の詳細ログ出力（頻度制限）
            if self._total_calls % 100 == 0:  # 100回に1回ログ出力（処理軽量化、大幅削減でバグ解決）
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[CURVE+BOOST] pos={pos_str} | theta={theta_deg:>5.1f}deg >= {BOOST_AND_CURVE_THRESHOLD_DEG}.0 | power={self.curve_power} + BOOST_1.1x")
            selected_power = self.curve_power
        else:
            # ストレート/ベースパワー判定と詳細ログ
            if position is not None:
                abs_position = abs(position)
                if abs_position <= POSITION_STRAIGHT:
                    # ストレートエリア（ログ頻度抑制）
                    if self._total_calls % 100 == 0:  # 100回に1回ログ出力（大幅削減）
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} <= {POSITION_STRAIGHT} | power={self.straight_power} (STRAIGHT)")
                    selected_power = self.straight_power
                elif abs_position >= 19000:  # 19000超えたら通常速度に戻す
                    # 19000以降は通常のベースパワーに戻す
                    if self._total_calls % 100 == 0:  # 100回に1回ログ出力（大幅削減）
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} >= 19000 | power={self.base_power} (RECOVERY)")
                    selected_power = self.base_power
                elif abs_position >= 16000:  # 16000-19000の難所エリアは減速
                    # 難所エリア（カーブが多い箇所）
                    if self._total_calls % 100 == 0:  # 100回に1回ログ出力（大幅削減）
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} 16000-19000 | power={self.curve_power} (DIFFICULT)")
                    selected_power = self.curve_power
                else:
                    # ベースパワーエリア（ログ頻度抑制）
                    if self._total_calls % 100 == 0:  # 100回に1回ログ出力（大幅削減）
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} > {POSITION_STRAIGHT} | power={self.base_power} (BASE)")
                    selected_power = self.base_power
            else:
                # ポジション不明時のデフォルト（ログ頻度抑制）
                if self._total_calls % 100 == 0:  # 100回に1回ログ出力（大幅削減）
                    print(f"[SPEED_SELECT] pos=  None | theta={theta_deg:>5.1f}deg | power={self.base_power} (BASE_DEFAULT)")
                selected_power = self.base_power
        
        # パワー値の急激な変化を検出（BOOST_AND_CURVE_THRESHOLD_DEGベースの適度な値）
        if self._last_power is not None:
            power_change = abs(selected_power - self._last_power)
            if power_change > BOOST_AND_CURVE_THRESHOLD_DEG * 1.25:  # 統一閾値の1.25倍（15）の急激なパワー変化
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[STABILITY_ALERT] pos={pos_str} | SUDDEN_POWER_CHANGE | {self._last_power:>2} → {selected_power:>2} (Δ{power_change:>2})")
        
        # 前回値を更新
        self._last_theta_deg = theta_deg
        self._last_power = selected_power
        
        # 定期的な統計情報表示（15秒ごとに変更）
        current_time = time.time()
        if current_time - self._last_stats_time >= 15.0:
            if self._total_calls > 0:
                boost_rate = (self._boost_count / self._total_calls) * 100
                curve_rate = (self._curve_power_count / self._total_calls) * 100
                print(f"[STATS] calls={self._total_calls} | boost={self._boost_count}({boost_rate:.1f}%) | curve={self._curve_power_count}({curve_rate:.1f}%)")
                # 統計リセット
                self._boost_count = 0
                self._curve_power_count = 0
                self._total_calls = 0
            self._last_stats_time = current_time
        
        return selected_power

    def calculate_power_adjustment(self, pid_corrected_theta, position=None):
        """
        PID補正値（ラジアン）をパワー差分（左右モーター出力の調整値）に変換する。
        theta値が大きい時（BOOST_AND_CURVE_THRESHOLD_DEG度超）は1.1倍のブーストをかける。
        カーブパワーと同時発動。
        """
        # 安全性チェック：入力値の検証
        if pid_corrected_theta is None:
            pid_corrected_theta = 0.0
        if not isinstance(pid_corrected_theta, (int, float)):
            pid_corrected_theta = 0.0
        if self.max_theta is None or self.max_theta == 0:
            self.max_theta = math.radians(30)
        
        # 現在のtheta値を保存（ブースト判定用）
        self._current_theta_deg = abs(math.degrees(pid_corrected_theta))
        
        # ブーストファクターの計算（ヒステリシス付きで安定化）
        # 現在ブーストONの場合は4度まで下がらないとOFFにしない（8-4=4度、ヒステリシス拡大）
        # 現在ブーストOFFの場合は8度を超えたらONにする
        if self._last_boost_factor > 1.0:  # 現在ブーストON
            threshold = BOOST_AND_CURVE_THRESHOLD_DEG - 4  # 4度（ヒステリシスさらに拡大）
        else:  # 現在ブーストOFF
            threshold = BOOST_AND_CURVE_THRESHOLD_DEG  # 8度
        
        current_boost_factor = 1.015 if self._current_theta_deg > threshold else 1.0  # 1.02→1.015にさらに穏やか化（MAX_POWER_DIFF=35対応）
        
        # ブースト統計の更新
        if current_boost_factor > 1.0:
            self._boost_count += 1
            
        # 基本パワー差分の計算
        try:
            base = (pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF
        except (TypeError, ZeroDivisionError):
            base = 0.0
        
        # 安全性チェック
        if not isinstance(base, (int, float)) or math.isnan(base) or math.isinf(base):
            base = 0.0
        
        # ブーストの適用
        boosted_base = base * current_boost_factor
        
        # ブースト状況の詳細ログ出力（頻度制限）
        current_time = time.time()
        boost_changed = abs(current_boost_factor - self._last_boost_factor) > 0.01
        should_log = (current_time - self._last_boost_log_time) >= 5.0 or boost_changed  # 5秒間隔に延長（バグ対策）
        
        if should_log:
            pos_str = f"{position:>6}" if position is not None else "  None"
            boost_status = "BOOST" if current_boost_factor > 1.0 else "NORM "
            factor_change = current_boost_factor - self._last_boost_factor
            change_str = f"({factor_change:+.2f})" if boost_changed else ""
            
            # ブースト発動・解除の詳細情報
            if boost_changed:
                if current_boost_factor > self._last_boost_factor:
                    event_str = "BOOST_ON "
                else:
                    event_str = "BOOST_OFF"
                print(f"[BOOST] pos={pos_str} | {event_str} | theta={self._current_theta_deg:>5.1f}deg | factor={current_boost_factor:.3f}{change_str} | base={base:>5.1f}→{boosted_base:>5.1f}")
            else:
                # 継続状態は50回に1回のみ出力（大幅削減）
                if self._total_calls % 50 == 0:
                    print(f"[BOOST] pos={pos_str} | {boost_status} | theta={self._current_theta_deg:>5.1f}deg | factor={current_boost_factor:.3f} | base={base:>5.1f}→{boosted_base:>5.1f}")
            
            self._last_boost_log_time = current_time
        
        # 前回値を保存
        self._last_boost_factor = current_boost_factor
        
        # パワー差分をクリップして返す
        if boosted_base > 0:
            power_adj = min(int(boosted_base), MAX_POWER_DIFF)
        else:
            power_adj = max(int(boosted_base), -MAX_POWER_DIFF)
        
        # パワー調整値の急激な変化を検出（MAX_POWER_DIFFベースの適度な値）
        if hasattr(self, '_last_power_adjustment'):
            adj_change = abs(power_adj - self._last_power_adjustment)
            if adj_change > 25:  # 25以上の急激な調整値変化（MAX_POWER_DIFF=40の約2/3で警告）
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[STABILITY_ALERT] pos={pos_str} | SUDDEN_ADJ_CHANGE | {self._last_power_adjustment:>+3} → {power_adj:>+3} (Δ{adj_change:>2})")
        
        self._last_power_adjustment = power_adj
            
        return int(power_adj)
