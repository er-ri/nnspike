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
from collections import deque
import json

# ====【現場でよく調整する推奨パラメータ】====
BASE_POWER = 32             # 通常走行時の基準パワー
STRAIGHT_POWER = 30         # 直線判定時のパワー
CURVE_POWER = 23            # カーブ判定時のパワー（20→25で復帰力UP）
STRAIGHT_THRESHOLD_DEG = 3  # 直線とみなす角度しきい値[deg]
CURVE_THRESHOLD_DEG = 10    # カーブとみなす角度しきい値[deg]
SENSITIVITY = 1.0           # θ感度UP（1.0→1.1）

# ====【通常は触らない高度なパラメータ】====
MAX_THETA_DEG = 30          # θの最大値[deg]（パワー補正の正規化用）
MAX_POWER_DIFF = 22         # PID補正による最大パワー差分  # 15→20
MAX_POWER_ADJ_STEP = 5     # パワー差分の1ステップ最大変化量（暴走抑制用）

class ControlCalculator:
    """
    ラインフォロワー用制御計算クラス。
    """

    def __init__(self, image_width, debug=True):
        """
        ControlCalculatorの初期化。

        引数:
            image_width: int 画像幅
            debug: bool デバッグ出力ON/OFF
        """
        # 制御パラメータの初期化（使用順に並べ替え）
        self.image_width = image_width  # 画像幅
        self.base_power = BASE_POWER  # 通常時の基準パワー
        self.straight_power = STRAIGHT_POWER  # 直線時の推奨パワー
        self.curve_power = CURVE_POWER  # カーブ時の推奨パワー
        self.straight_threshold = math.radians(STRAIGHT_THRESHOLD_DEG)  # 直線判定しきい値[rad]
        self.curve_threshold = math.radians(CURVE_THRESHOLD_DEG)  # カーブ判定しきい値[rad]
        self.max_theta = math.radians(MAX_THETA_DEG)  # θ最大値[rad]
        self.debug = debug  # デバッグ出力ON/OFF
        self.last_power_adj = 0  # 前回のパワー差分（暴走抑制用）
        self.prev_power_adj = 0  # 暴走抑制用: 前回のパワー差分

    def calculate_and_smooth_theta(self, offset_pixels):
        """
        オフセットピクセルからθを計算し、そのまま返す（平滑化なし）。
        デバッグ出力も1回でまとめて行う。

        引数:
            offset_pixels: int オフセットピクセル
        戻り値:
            float θ[rad]
        """
        image_center_x = self.image_width / 2
        max_offset = image_center_x
        normalized_offset = offset_pixels / max_offset
        theta = normalized_offset * SENSITIVITY
        if self.debug:
            debug_data = {
                "offset": offset_pixels,
                "theta": round(theta, 4),
                "smoothed": round(theta, 4)
            }
            print(f"[CONTROL_DEBUG] calculate_and_smooth_theta {json.dumps(debug_data, ensure_ascii=False)}")
        return theta

    def calculate_adaptive_speed(self, smoothed_theta):
        # 平滑化後θに応じて推奨速度（パワー）を自動調整
        abs_theta = abs(smoothed_theta)
        if abs_theta > self.curve_threshold:
            speed = self.curve_power  # カーブ時
        elif abs_theta < self.straight_threshold:
            speed = self.straight_power  # 直線時
        else:
            speed = self.base_power  # 通常時
        if self.debug:
            debug_data = {
                "abs_theta": round(abs_theta, 4),
                "speed": speed
            }
            print(f"[CONTROL_DEBUG] calculate_adaptive_speed {json.dumps(debug_data, ensure_ascii=False)}")
        return speed

    def calculate_power_adjustment(self, pid_corrected_theta):
        # PID補正値（ラジアン）をパワー差分に変換（暴走抑制なし）
        power_adj = int((pid_corrected_theta / self.max_theta) * MAX_POWER_DIFF)
        if self.debug:
            debug_data = {
                "pid_theta": round(pid_corrected_theta, 4),
                "power_adj": power_adj
            }
            print(f"[CONTROL_DEBUG] calculate_power_adjustment {json.dumps(debug_data, ensure_ascii=False)}")
        return power_adj

    def calculate_attitude_angle(
        offset_pixels: float,
        roi_bottom_y: int,
        camera_height: float = 0.20,
        focal_length_pixels: float = 640,
    ) -> float:
        """
        Calculate attitude angle (theta) from pixel offset using camera geometry.

        This function converts the pixel-based offset detected in the camera image
        to a real-world attitude angle that represents the robot's deviation from
        the desired path. This provides more physically meaningful control compared
        to simple pixel-based normalization.

        Args:
            offset_pixels (float): Lateral offset in pixels from image center
            roi_bottom_y (int): Bottom y-coordinate of ROI (closer to robot)
            camera_height (float, optional): Camera height above ground in meters. Defaults to 0.20.
            focal_length_pixels (float, optional): Camera focal length in pixels. Defaults to 640.

        Returns:
            float: Attitude angle (theta) in radians. Positive values indicate rightward deviation,
                negative values indicate leftward deviation.

        Note:
            The camera parameters (height and focal length) should be calibrated for your
            specific robot setup to ensure accurate angle calculations.
        """
        # Calculate ground distance from camera to the line detection point
        # Using similar triangles: ground_distance / camera_height = focal_length / (image_height - roi_bottom_y)
        image_height = 480  # Assuming standard camera resolution
        ground_distance = (
            camera_height * focal_length_pixels / (image_height - roi_bottom_y)
        )

        # Calculate lateral offset in meters
        # Using similar triangles: lateral_offset / ground_distance = offset_pixels / focal_length
        lateral_offset_meters = offset_pixels * ground_distance / focal_length_pixels

        # Calculate attitude angle (theta) using arctangent
        theta = math.atan2(lateral_offset_meters, ground_distance)

        return theta


    def calculate_differential_steering(
        theta: float, base_speed: float, wheelbase: float = 0.15
    ) -> tuple[float, float]:
        """
        Calculate differential steering based on attitude angle and robot geometry.

        This function implements proper differential drive kinematics to calculate
        individual wheel speeds based on the desired turning angle. It uses a
        simplified but robust approach that directly relates steering angle to
        wheel speed difference.

        Args:
            theta (float): Attitude angle in radians (positive = turn right, negative = turn left)
            base_speed (float): Base forward speed for both wheels when going straight
            wheelbase (float, optional): Distance between wheels in meters. Defaults to 0.15.

        Returns:
            tuple[float, float]: A tuple containing (left_speed, right_speed)
                            Individual wheel speeds for differential steering

        Note:
            - For straight line motion (theta ≈ 0), both wheels get the same speed
            - For turning, the inside wheel gets reduced speed, outside wheel gets increased speed
            - Uses simplified differential steering for better stability and control
        """
        # Handle straight line case
        if abs(theta) < 0.001:  # Threshold for "straight enough"
            return base_speed, base_speed

        # Calculate steering factor based on angle
        # This approach provides more intuitive and stable control
        steering_factor = math.sin(theta)  # Normalized steering influence (-1 to 1)

        # Calculate speed adjustment for each wheel
        # The maximum speed difference is proportional to base_speed
        max_speed_diff = base_speed * 0.5  # Maximum 50% speed difference
        speed_adjustment = steering_factor * max_speed_diff

        if theta < 0:  # Turn right (positive theta)
            left_speed = base_speed + abs(speed_adjustment)  # Left wheel speeds up (outer)
            right_speed = base_speed - abs(
                speed_adjustment
            )  # Right wheel slows down (inner)
        else:  # Turn left (negative theta)
            left_speed = base_speed - abs(speed_adjustment)  # Left wheel slows down (inner)
            right_speed = base_speed + abs(
                speed_adjustment
            )  # Right wheel speeds up (outer)

        # Ensure speeds don't go negative
        left_speed = max(0, left_speed)
        right_speed = max(0, right_speed)

        return left_speed, right_speed


    def get_line_edges_at_y(image, roi, target_y, threshold_value=50):
        """
        Get the left and right edge points of a black line at a specific Y coordinate.

        Parameters:
        - image: Input image (BGR or grayscale)
        - roi_coords: Tuple (x, y, width, height) defining the ROI
        - target_y: The Y coordinate where to detect line edges (in original image coordinates)
        - threshold_value: Threshold for binary conversion (default: 50)

        Returns:
        - left_x: X coordinate of left edge (None if not found)
        - right_x: X coordinate of right edge (None if not found)
        - line_width: Width of the line at this Y position (None if not found)
        """

        # Extract ROI coordinates
        x, y, w, h = roi

        # Check if target_y is within ROI
        if target_y < y or target_y >= y + h:
            return None, None, None

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Extract ROI
        roi = gray[y : y + h, x : x + w]

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        # Binary threshold to isolate black line
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Calculate the row within the ROI
        roi_row = target_y - y

        # Get the binary row at target Y
        if roi_row >= 0 and roi_row < h:
            row_data = binary[roi_row, :]

            # Find all white pixels (line pixels) in this row
            white_pixels = np.where(row_data == 255)[0]

            if len(white_pixels) > 0:
                # Find leftmost and rightmost white pixels
                left_x_roi = white_pixels[0]
                right_x_roi = white_pixels[-1]

                # Convert back to original image coordinates
                left_x = x + left_x_roi
                right_x = x + right_x_roi
                line_width = right_x - left_x + 1

                return left_x, right_x, line_width

        return None, None, None