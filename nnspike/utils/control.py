"""
ラインフォロワー制御モジュール

このモジュールは、カメラ画像を用いたラインフォロー制御アルゴリズムを実装します。

機能:
    - 画像の中心とラインのオフセットから進行角度（theta）を計算
    - 平滑化や速度調整、パワー差分の計算
    - ブースト機能（一定の角度しきい値を超えた場合の出力増幅）
    - 位置に応じた速度調整（直線、カーブ、困難地帯）
    - エッジ自動切替（position値に基づく）
    - 統計情報表示（定期的な状態監視）

クラス:
    ControlCalculator:
        ラインフォロワー用制御計算の中心クラス
"""

import math
import time
from collections import deque
import numpy as np

# パラメータ定数
BASE_POWER = 30
STRAIGHT_POWER = 50
CURVE_POWER = 15
STRAIGHT_THRESHOLD_DEG = 3
BOOST_AND_CURVE_THRESHOLD_DEG = 6

CAMERA_HEIGHT = 0.20
CAMERA_FOCAL_LENGTH_PIXELS = 640
WHEELBASE = 0.10
MAX_THETA_DEG = 30
MAX_POWER_DIFF = 40
OFFSET_Y = 400

# 位置しきい値
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
        """ControlCalculatorの初期化"""
        self.image_width = image_width
        self.image_height = image_height
        self.base_power = BASE_POWER
        self.straight_power = STRAIGHT_POWER
        self.curve_power = CURVE_POWER
        self.straight_threshold = math.radians(STRAIGHT_THRESHOLD_DEG)
        self.boost_and_curve_threshold = math.radians(BOOST_AND_CURVE_THRESHOLD_DEG)
        self.max_theta = math.radians(MAX_THETA_DEG)
        self.debug = debug
        self.roi_opencv = roi_opencv
        
        # ブースト制御用
        self._current_theta_deg = 0.0
        self._last_boost_log_time = 0
        self._last_boost_factor = 1.0
        
        # 状態監視用
        self._last_power = None
        self._last_theta_deg = 0.0
        self._last_power_adjustment = 0
        
        # 統計情報
        self._boost_count = 0
        self._curve_power_count = 0
        self._total_calls = 0
        self._last_stats_time = time.time()

    @property
    def is_boost_active(self):
        """ブースト状態判定"""
        return self._last_boost_factor > 1.0

    @property
    def current_boost_factor(self):
        """現在のブーストファクター"""
        return self._last_boost_factor
    
    @property
    def current_theta_deg(self):
        """現在のtheta値（度）"""
        return self._current_theta_deg

    def calc_steer_result(self, left_x, right_x, position=None):
        """ライントレース用のステアリング結果を計算"""
        x1, y1, x2, y2 = self.roi_opencv
        
        # エッジ自動判定
        if position is not None:
            abs_position = abs(position)
            if abs_position <= POSITION_CROSS1:
                follow_edge = "right"
            elif abs_position <= POSITION_CROSS2:
                follow_edge = "left"
            elif abs_position <= POSITION_CROSS3:
                follow_edge = "right"
            elif abs_position <= POSITION_CROSS4:
                follow_edge = "left"
            else:
                follow_edge = "right"
        else:
            follow_edge = "left"

        if not hasattr(self, '_last_mx'):
            self._last_mx = None
            self._last_my = None

        # ライン検出処理
        if left_x is not None and right_x is not None:
            # 左エッジ検出時は検出範囲を拡張して安定性向上
            if follow_edge == "left":
                min_x = x1 + int((x2 - x1) * 0.05)  # 0.2→0.05に拡張
                max_x = x1 + int((x2 - x1) * 0.95)  # 0.8→0.95に拡張
            else:
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
            mx = candidate_x - x1  # ROI内相対座標
            my = OFFSET_Y - y1     # ROI内相対座標
            self._last_mx = mx
            self._last_my = my
            # 画像全体の中心からの差分を計算（画像全体の中心は IMAGE_WIDTH//2 = 320）
            absolute_x = x1 + mx  # 画像全体での絶対座標
            offset_pixels = absolute_x - (self.image_width // 2)
            line_status = "DETECTED"
        else:
            # 前回の点を維持
            if self._last_mx is not None and self._last_my is not None:
                mx = self._last_mx
                my = self._last_my
                # 画像全体の中心からの差分を計算
                absolute_x = x1 + mx  # 画像全体での絶対座標
                offset_pixels = absolute_x - (self.image_width // 2)
                max_contour = None
                line_status = "LOST_HOLDING"
            else:
                mx = (x2 - x1) // 2
                my = (y2 - y1) // 2
                offset_pixels = 0
                max_contour = None
                line_status = "LOST_CENTER"
        
        # 定期ログ出力
        if hasattr(self, '_last_steer_debug'):
            if time.time() - self._last_steer_debug >= 3.0:
                pos_str = f"{position:>6}" if position is not None else "  None"
                absolute_x = x1 + mx if left_x is not None and right_x is not None or (self._last_mx is not None) else x1 + mx
                print(f"[STEER] pos={pos_str} | {line_status} | mx={mx:>3} offset={offset_pixels:>3}px | edge={follow_edge} | abs_x={absolute_x}")
                self._last_steer_debug = time.time()
        else:
            self._last_steer_debug = time.time()
            
        return {
            "mx": mx,
            "my": my,
            "offset_pixels": offset_pixels,
            "max_contour": max_contour,
            "roi_type": "opencv",
            "follow_edge": follow_edge,
            "position": position
        }

    def calculate_attitude_angle(self, offset_pixels: float) -> float:
        """ピクセルオフセットから姿勢角（theta）を計算"""
        # offset_pixelsを数値に変換（numpy型も対応）
        try:
            offset_pixels = float(offset_pixels)
        except (TypeError, ValueError):
            offset_pixels = 0.0
        
        if math.isnan(offset_pixels) or math.isinf(offset_pixels):
            offset_pixels = 0.0
            
        x1, y1, x2, y2 = self.roi_opencv
        denominator = self.image_height - y2
        if denominator <= 0:
            denominator = 1.0
            
        ground_distance = (CAMERA_HEIGHT * CAMERA_FOCAL_LENGTH_PIXELS / denominator)
        
        if CAMERA_FOCAL_LENGTH_PIXELS <= 0:
            lateral_offset_meters = 0.0
        else:
            lateral_offset_meters = offset_pixels * ground_distance / CAMERA_FOCAL_LENGTH_PIXELS
            
        if ground_distance <= 0:
            theta = 0.0
        else:
            theta = math.atan2(lateral_offset_meters, ground_distance)
            
        if math.isnan(theta) or math.isinf(theta):
            theta = 0.0
            
        # デバッグ出力: θ計算過程を詳細表示（頻度制限）
        theta_deg = math.degrees(theta)
        if not hasattr(self, '_last_theta_debug'):
            self._last_theta_debug = 0
        if time.time() - self._last_theta_debug >= 5.0:  # 5秒間隔に変更
            print(f"[THETA_DEBUG] offset_px={offset_pixels:.1f} | theta_deg={theta_deg:.2f}")
            self._last_theta_debug = time.time()
            
        return theta

    def calculate_adaptive_speed(self, theta, position=None):
        """適応的な速度計算"""
        if theta is None or not isinstance(theta, (int, float)):
            theta_deg = 0.0
        elif math.isnan(theta) or math.isinf(theta):
            theta_deg = 0.0
        else:
            theta_deg = abs(math.degrees(theta))
        
        self._total_calls += 1
        
        # 急激な変化の検出
        theta_change = abs(theta_deg - self._last_theta_deg)
        if theta_change > BOOST_AND_CURVE_THRESHOLD_DEG:
            pos_str = f"{position:>6}" if position is not None else "  None"
            print(f"[STABILITY_ALERT] pos={pos_str} | SUDDEN_THETA_CHANGE | {self._last_theta_deg:>5.1f}deg → {theta_deg:>5.1f}deg (Δ{theta_change:>5.1f})")
        
        # カーブパワー+ブースト同時発動（シンプルな判定）
        if theta_deg >= BOOST_AND_CURVE_THRESHOLD_DEG:
            self._curve_power_count += 1
            if self._total_calls % 100 == 0:
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[CURVE+BOOST] pos={pos_str} | theta={theta_deg:>5.1f}deg >= {BOOST_AND_CURVE_THRESHOLD_DEG}.0 | power={self.curve_power}")
            selected_power = self.curve_power
        else:
            # 位置に応じた速度選択
            if position is not None:
                abs_position = abs(position)
                if abs_position <= POSITION_STRAIGHT:
                    if self._total_calls % 100 == 0:
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} <= {POSITION_STRAIGHT} | power={self.straight_power} (STRAIGHT)")
                    selected_power = self.straight_power
                elif abs_position >= 16000:
                    # 16000以降（POSITION_CROSS3含む）は低速で安定制御
                    if self._total_calls % 100 == 0:
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} >= 16000 | power={self.curve_power} (DIFFICULT)")
                    selected_power = self.curve_power
                else:
                    if self._total_calls % 100 == 0:
                        pos_str = f"{position:>6}"
                        print(f"[SPEED_SELECT] pos={pos_str} | theta={theta_deg:>5.1f}deg | abs_pos={abs_position:>3.0f} > {POSITION_STRAIGHT} | power={self.base_power} (BASE)")
                    selected_power = self.base_power
            else:
                if self._total_calls % 100 == 0:
                    print(f"[SPEED_SELECT] pos=  None | theta={theta_deg:>5.1f}deg | power={self.base_power} (BASE_DEFAULT)")
                selected_power = self.base_power
        
        # 急激なパワー変化の検出
        if self._last_power is not None:
            power_change = abs(selected_power - self._last_power)
            if power_change > BOOST_AND_CURVE_THRESHOLD_DEG * 1.25:
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[STABILITY_ALERT] pos={pos_str} | SUDDEN_POWER_CHANGE | {self._last_power:>2} → {selected_power:>2} (Δ{power_change:>2})")
        
        self._last_theta_deg = theta_deg
        self._last_power = selected_power
        
        # 統計情報表示
        current_time = time.time()
        if current_time - self._last_stats_time >= 15.0:
            if self._total_calls > 0:
                boost_rate = (self._boost_count / self._total_calls) * 100
                curve_rate = (self._curve_power_count / self._total_calls) * 100
                print(f"[STATS] calls={self._total_calls} | boost={self._boost_count}({boost_rate:.1f}%) | curve={self._curve_power_count}({curve_rate:.1f}%)")
                self._boost_count = 0
                self._curve_power_count = 0
                self._total_calls = 0
            self._last_stats_time = current_time
        
        return selected_power

    def calculate_power_adjustment(self, pid_corrected_theta, position=None):
        """パワー調整値計算（ブースト機能付き）"""
        if pid_corrected_theta is None:
            pid_corrected_theta = 0.0
        if not isinstance(pid_corrected_theta, (int, float)):
            pid_corrected_theta = 0.0
        if self.max_theta is None or self.max_theta == 0:
            self.max_theta = math.radians(30)
        
        self._current_theta_deg = abs(math.degrees(pid_corrected_theta))
        
        # シンプルなブースト判定
        if self._current_theta_deg >= BOOST_AND_CURVE_THRESHOLD_DEG:
            current_boost_factor = 1.3  # 1.3倍ブースト
        else:
            current_boost_factor = 1.0
        
        # シンプルなブースト制御
        if current_boost_factor > 1.0:
            self._boost_count += 1
            
        # パワー差分計算
        if self.max_theta != 0:
            base = (pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF
        else:
            base = 0.0
        
        if not isinstance(base, (int, float)) or math.isnan(base) or math.isinf(base):
            base = 0.0
        
        boosted_base = base * current_boost_factor
        
        # ブーストログ出力
        current_time = time.time()
        boost_changed = abs(current_boost_factor - self._last_boost_factor) > 0.01
        should_log = (current_time - self._last_boost_log_time) >= 5.0 or boost_changed
        
        if should_log:
            pos_str = f"{position:>6}" if position is not None else "  None"
            boost_status = "BOOST" if current_boost_factor > 1.0 else "NORM "
            factor_change = current_boost_factor - self._last_boost_factor
            change_str = f"({factor_change:+.2f})" if boost_changed else ""
            
            if boost_changed:
                event_str = "BOOST_ON " if current_boost_factor > self._last_boost_factor else "BOOST_OFF"
                print(f"[BOOST] pos={pos_str} | {event_str} | theta={self._current_theta_deg:>5.1f}deg | factor={current_boost_factor:.3f}{change_str} | base={base:>5.1f}→{boosted_base:>5.1f}")
            else:
                if self._total_calls % 50 == 0:
                    print(f"[BOOST] pos={pos_str} | {boost_status} | theta={self._current_theta_deg:>5.1f}deg | factor={current_boost_factor:.3f} | base={base:>5.1f}→{boosted_base:>5.1f}")
            
            self._last_boost_log_time = current_time
        
        self._last_boost_factor = current_boost_factor
        
        # クリップして返す
        if boosted_base > 0:
            power_adj = min(int(boosted_base), MAX_POWER_DIFF)
        else:
            power_adj = max(int(boosted_base), -MAX_POWER_DIFF)
        
        # 急激な調整値変化の検出
        if hasattr(self, '_last_power_adjustment'):
            adj_change = abs(power_adj - self._last_power_adjustment)
            if adj_change > 25:
                pos_str = f"{position:>6}" if position is not None else "  None"
                print(f"[STABILITY_ALERT] pos={pos_str} | SUDDEN_ADJ_CHANGE | {self._last_power_adjustment:>+3} → {power_adj:>+3} (Δ{adj_change:>2})")
        
        self._last_power_adjustment = power_adj
            
        return int(power_adj)
