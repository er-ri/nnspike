
import threading
import time

import serial  # type: ignore

from .spike_status import SpikeStatus


class ETRobot(object):

    # Command IDs should be as same as (`spike/slot_prod.py`) the script in LEGO Spike Prime.
    COMMAND_SET_MOTOR_FORWARD_SPEED_ID = 201
    COMMAND_SET_MOTOR_BACKWARD_SPEED_ID = 202
    COMMAND_SET_MOTOR_RELATIVE_POSITION_ID = 203
    COMMAND_STOP_MOTOR_ID = 204
    COMMAND_MOVE_ARM_ID = 205
    COMMAND_SET_MOTOR_MIXED_SPEED_ID = 206

    CMD_FLAG = b"CF:"

    DUMMY = 1  # Dummy value for parameters

    def __init__(self, port="/dev/ttyACM0") -> None:
        self.__serial_port = serial.Serial(port=port, baudrate=115200, timeout=0.05)
        self.__serial_port.reset_input_buffer()
        self.__serial_port.reset_output_buffer()

        self.is_running = True
        # Create a SpikeStatus object for handling data
        self.spike_status = SpikeStatus()

        # Keep track of last valid sensor data to handle missing readings
        self.last_spike_status = SpikeStatus()

        self.__thread = threading.Thread(target=self.__update_status)
        self.__thread.start()
        self.last_update_time = None
        self.update_count = 0

    def __send_command(self, command) -> None:
        """Send a command to the robot via the serial port."""
        self.__serial_port.write(self.CMD_FLAG + command)

    def __update_status(self) -> None:
        """Background thread that continuously receives data from the serial connection and updates gyro integration."""
        while self.is_running:
            self.receive()
            self.update_gyro_integration()

    def receive(self) -> None:
        """
        Update ETRobot motor and sensor status by the received data from the GPIO port.

        Note:
            The update rate should be less than the rate of sending sensor data in LEGO Prime Hub (0.0005 seconds).
        """
        # --- 旧処理（コメントアウト） ---
        # received_data = self.__serial_port.read_until(expected=b"\r")
        # if received_data.startswith(b"{"):
        #     try:
        #         self.spike_status.update(received_data)
        #         self.__update_last_spike_status()
        #     except Exception as e:
        #         print(f"Error processing data: {e}")
        #         print(f"Raw data: {received_data}")
        # time.sleep(0.005)

        # --- 新処理（バッファでJSON終端までためる） ---
        received_data = b""
        while True:
            chunk = self.__serial_port.read_until(expected=b"\r")
            received_data += chunk
            if received_data.strip().endswith(b'}'):
                break

        if received_data.startswith(b"{"):
            try:
                # Update the spike status with the new data
                self.spike_status.update(received_data)
                # Update last_spike_status with valid sensor readings
                self.__update_last_spike_status()

            except Exception as e:
                print(f"Error processing data: {e}")
                print(f"Raw data: {received_data}")

        time.sleep(0.001)

    def __update_last_spike_status(self) -> None:
        """
        Update last_spike_status with current valid sensor readings.
        Only updates values that are not None to preserve last known good values.
        """
        current = self.spike_status
        last = self.last_spike_status

        # Update sensors with valid readings
        # Update motor data (always update as these are more reliable)
        for motor_id in ["A", "B", "C"]:
            if current.motors[motor_id].position is not None:
                last.motors[motor_id].position = current.motors[motor_id].position
            if current.motors[motor_id].relative_position is not None:
                last.motors[motor_id].relative_position = current.motors[motor_id].relative_position
            if current.motors[motor_id].speed is not None:
                last.motors[motor_id].speed = current.motors[motor_id].speed
            if current.motors[motor_id].power is not None:
                last.motors[motor_id].power = current.motors[motor_id].power

        # Distance sensor
        if current.sensors.distance is not None:
            last.sensors.distance = current.sensors.distance
        else:
            last.sensors.distance = 0

        if current.sensors.force is not None:
            last.sensors.force = current.sensors.force

        # Color sensor data
        if current.sensors.color:
            if not last.sensors.color:
                from .spike_status import ColorSensorStatus
                last.sensors.color = ColorSensorStatus()
            if current.sensors.color.reflected is not None:
                last.sensors.color.reflected = current.sensors.color.reflected
            if current.sensors.color.ambient is not None:
                last.sensors.color.ambient = current.sensors.color.ambient
            if current.sensors.color.color is not None:
                last.sensors.color.color = current.sensors.color.color

        # Accelerometer data
        if current.sensors.accelerometer:
            if not last.sensors.accelerometer:
                from .spike_status import VectorStatus
                last.sensors.accelerometer = VectorStatus()
            if current.sensors.accelerometer.x is not None:
                last.sensors.accelerometer.x = current.sensors.accelerometer.x
            if current.sensors.accelerometer.y is not None:
                last.sensors.accelerometer.y = current.sensors.accelerometer.y
            if current.sensors.accelerometer.z is not None:
                last.sensors.accelerometer.z = current.sensors.accelerometer.z

        # Gyroscope data（生値）
        if current.sensors.gyroscope:
            if not last.sensors.gyroscope:
                from .spike_status import VectorStatus
                last.sensors.gyroscope = VectorStatus()
            if current.sensors.gyroscope.x is not None:
                last.sensors.gyroscope.x = current.sensors.gyroscope.x
            if current.sensors.gyroscope.y is not None:
                last.sensors.gyroscope.y = current.sensors.gyroscope.y
            if current.sensors.gyroscope.z is not None:
                last.sensors.gyroscope.z = current.sensors.gyroscope.z

        # YawPitchRoll（ヨー・ピッチ・ロール）データ
        if current.sensors.yaw_pitch_roll:
            if not last.sensors.yaw_pitch_roll:
                from .spike_status import VectorStatus
                last.sensors.yaw_pitch_roll = VectorStatus()
            if current.sensors.yaw_pitch_roll.x is not None:
                last.sensors.yaw_pitch_roll.x = current.sensors.yaw_pitch_roll.x
            if current.sensors.yaw_pitch_roll.y is not None:
                last.sensors.yaw_pitch_roll.y = current.sensors.yaw_pitch_roll.y
            if current.sensors.yaw_pitch_roll.z is not None:
                last.sensors.yaw_pitch_roll.z = current.sensors.yaw_pitch_roll.z

        # Update battery data (for HIGH_SPEED_BASE optimization)
        if current.battery.voltage is not None:
            last.battery.voltage = current.battery.voltage
        if current.battery.percent is not None:
            last.battery.percent = current.battery.percent
        
        # message_typeとraw_dataも更新
        last.message_type = current.message_type
        last.raw_data = current.raw_data

    def get_spike_status(self):
        """
        Get the spike status with last known good sensor values.

        Returns:
            SpikeStatus: Spike status object with consistent sensor data
        """
        return self.last_spike_status

    def get_motor_power(self) -> tuple[int, int]:
        """
        左右モーターの出力（power）をタプルで返す。

        Returns:
            (int, int): (left_power, right_power)
        """
        status = self.get_spike_status()
        left = abs(status.motors["A"].power) if status.motors["A"].power is not None else 0
        right = abs(status.motors["B"].power) if status.motors["B"].power is not None else 0
        return (left, right)

    def get_motor_speed(self) -> tuple[int, int]:
        """
        左右モーターの速度（speed）をタプルで返す。

        Returns:
            (int, int): (left_speed, right_speed)
        """
        status = self.get_spike_status()
        left = abs(status.motors["A"].speed) if status.motors["A"].speed is not None else 0
        right = abs(status.motors["B"].speed) if status.motors["B"].speed is not None else 0
        return (left, right)

    def get_side_adjust_by_speed_diff(self) -> tuple[int, int]:
        """
        モーターspeed差分による補正値を返す。
        Returns:
            (int, int): (左補正, 右補正)
        """
        left_speed, right_speed = self.get_motor_speed()
        speed_diff = right_speed - left_speed
        threshold = 3
        if speed_diff > threshold:
            return (-1, 0)
        elif speed_diff < -threshold:
            return (0, -1)
        else:
            return (0, 0)

    def calc_speed_with_roll_control(
        self,
        speed: int = 100
    ) -> tuple[int, int]:
        """
        ロール補正（z軸）で速度指令値を計算する。
        Args:
            speed (int): 目標速度
        Returns:
            (int, int): (left_speed, right_speed)
        """
        roll_adj_left, roll_adj_right = self.get_side_adjust_by_roll()
        left_cmd = speed + roll_adj_left
        right_cmd = speed + roll_adj_right
        return int(left_cmd), int(right_cmd)

    def calc_speed_with_speed_diff_control(
        self,
        speed: int = 100
    ) -> tuple[int, int]:
        """
        speed差分補正のみで速度指令値を計算する。
        Args:
            speed (int): 目標速度
        Returns:
            (int, int): (left_speed, right_speed)
        """
        speed_adj_left, speed_adj_right = self.get_side_adjust_by_speed_diff()
        left_cmd = speed + speed_adj_left
        right_cmd = speed + speed_adj_right
        return int(left_cmd), int(right_cmd)

    def set_motor_relative_position(self, left_positon: int, right_position: int) -> None:
        id_byte = self.COMMAND_SET_MOTOR_RELATIVE_POSITION_ID.to_bytes(1, "big")
        parameter1_byte = left_positon.to_bytes(1, "big")
        parameter2_byte = right_position.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        start_time = time.time()
        while time.time() - start_time < 0.5:
            self.__send_command(command)
            time.sleep(0.05)

    def get_motor_relative_position(self, side: str) -> int:
        """
        指定したサイド（'right' または 'left'）のモーター相対位置を返す。

        Args:
            side (str): 'right' または 'left'

        Returns:
            int: 指定したモーターの相対位置（Noneの場合は0）
        """
        status = self.get_spike_status()
        if side == 'right':
            return abs(status.motors["B"].relative_position) if status.motors["B"].relative_position is not None else 0
        elif side == 'left':
            return abs(status.motors["A"].relative_position) if status.motors["A"].relative_position is not None else 0
        else:
            return 0

    def get_color_sensor(self) -> tuple[int, str]:
        """
        現在のカラーセンサーのカラータイプ値（color.color）と、値によるカラータイプ（black/white/other）を返す。

        Returns:
            Tuple[Optional[int], str]: (カラータイプ値, カラータイプ名)
        """
        status = self.get_spike_status()
        color_value_raw = status.sensors.color.color
        if color_value_raw is None:
            color_value = 0
            color_type = "unknown"
        else:
            color_value = int(color_value_raw)
            if color_value < 200:
                color_type = "black"
            elif color_value > 900:
                color_type = "white"
            else:
                color_type = "other"
        return (color_value, color_type)

    def set_motor_speed(self, left_speed: int, right_speed: int) -> None:
        """
        左右のモーター速度を個別に正転・逆転（マイナス値）で設定できる（ID=206コマンド送信）。

        Args:
            left_speed (int): 左モーター速度（-100～100、負なら逆転）
            right_speed (int): 右モーター速度（-100～100、負なら逆転）
        """
        if left_speed < -100 or left_speed > 100 or right_speed < -100 or right_speed > 100:
            raise ValueError("Motor speeds must be between -100 and 100.")

        id_byte = self.COMMAND_SET_MOTOR_MIXED_SPEED_ID.to_bytes(1, "big")
        # -100～+100 → 0～200 に変換して送信
        parameter1_byte = (left_speed + 100).to_bytes(1, "big", signed=False)
        parameter2_byte = (right_speed + 100).to_bytes(1, "big", signed=False)
        command = id_byte + parameter1_byte + parameter2_byte
        self.__send_command(command)

    def set_motor_forward_speed(self, left_speed: int, right_speed: int) -> None:
        """
        Set the ETRobot motor's speed.

        Args:
            left_speed (int): Left motor speed (0-100).
            right_speed (int): Right motor speed (0-100)."""
        id_byte = self.COMMAND_SET_MOTOR_FORWARD_SPEED_ID.to_bytes(1, "big")
        parameter1_byte = left_speed.to_bytes(1, "big")
        parameter2_byte = right_speed.to_bytes(1, "big")
        command = id_byte + parameter1_byte + parameter2_byte
        self.__send_command(command)

    def set_motor_backward_speed(self, left_speed: int, right_speed: int) -> None:
        """
        Set the ETRobot motor's speed in reverse direction.

        Args:
            left_speed (int): Left motor speed (0-100).
            right_speed (int): Right motor speed (0-100).
        """
        id_byte = self.COMMAND_SET_MOTOR_BACKWARD_SPEED_ID.to_bytes(1, "big")
        parameter1_byte = left_speed.to_bytes(1, "big")
        parameter2_byte = right_speed.to_bytes(1, "big")
        command = id_byte + parameter1_byte + parameter2_byte
        self.__send_command(command)

    def brake(self) -> None:
        """Brake the motors of the ETRobot."""
        id_byte = self.COMMAND_STOP_MOTOR_ID.to_bytes(1, "big")
        parameter1_byte = self.DUMMY.to_bytes(1, "big")
        parameter2_byte = self.DUMMY.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        # Continuously send stop commands for 0.5 seconds to ensure reliability
        start_time = time.time()
        while time.time() - start_time < 0.5:
            self.__send_command(command)
            time.sleep(0.05)

    def move_arm(self, action: int, duration: float = 0.5) -> None:
        """
        Move the arm up or down.

        Args:
            action (int): Action to perform (0 = move down, 1 = move up, 2 = stop arm)。
                0: アームを下げる (move down)
                1: アームを上げる (move up)
                2: アームを止める (stop arm)
        """
        id_byte = self.COMMAND_MOVE_ARM_ID.to_bytes(1, "big")
        parameter1_byte = action.to_bytes(1, "big")
        parameter2_byte = self.DUMMY.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        # Continuously send stop commands for 0.5 seconds to ensure reliability
        start_time = time.time()
        while time.time() - start_time < duration:
            self.__send_command(command)
            time.sleep(0.05)

    def stop(self) -> None:
        """Stop the robot and close the serial port."""
        self.is_running = False
        self.brake()
        self.__thread.join()
        self.__serial_port.close()

    def get_gyro_angle_z(self) -> float:
        """
        SpikeStatusからgyroscope_z角度（度）を取得
        Returns:
            float: z軸角度（度）
        """
        status = self.get_spike_status()
        if status.sensors.gyroscope:
            return status.sensors.gyroscope.z
        return 0.0

    def start_gyro_integration(self):
        """
        ジャイロ積分の開始（基準値・時刻を記録）
        """
        status = self.get_spike_status()
        self._gyro_integrated_z = 0.0
        self._gyro_integration_start_time = time.time()
        self._gyro_integration_last_time = self._gyro_integration_start_time  # 追加
        self._gyro_integration_start_z = status.sensors.gyroscope.z if status.sensors.gyroscope else 0.0
        self._gyro_integration_active = True

    def update_gyro_integration(self):
        """
        ジャイロ積分値を最新値で加算（ループ内で呼ぶ）
        dt（ms）と_gyro_integrated_zをprintデバッグ出力
        """
        if not getattr(self, '_gyro_integration_active', False):
            return
        status = self.get_spike_status()
        now = time.time()
        dt = now - getattr(self, '_gyro_integration_start_time', now)
        dt_ms = int(dt * 1000)
        if status.sensors.gyroscope:
            # 角速度（deg/s）× dt（s）で積分（スタートからの累積）
            self._gyro_integrated_z = status.sensors.gyroscope.z * dt
            self._gyro_integration_last_time = now
            print(f"[GyroIntegration] dt={dt_ms}ms, gyro_z={status.sensors.gyroscope.z:.2f}, integrated_z={self._gyro_integrated_z:.2f}")

    def get_gyro_integrated_z(self) -> float:
        """
        積分したgyro_z角度（度）を返す
        """
        return getattr(self, '_gyro_integrated_z', 0.0)


    def reset_gyro_integration(self):
        """
        ジャイロ積分値をリセット
        """
        self._gyro_integrated_z = 0.0
        self._gyro_integration_active = False
        self._gyro_integration_last_time = None  # 追加


    def is_gyro_integrated_rotation_exceeded_z(self, threshold: float, direction: str) -> bool:
        """
        積分加算したジャイロz回転角度が指定した方向・閾値を超えたか判定する。
        Args:
            threshold (float): 閾値（度）
            direction (str): 'left' or 'right'
        Returns:
            bool: 条件を満たせばTrue
        """
        integrated = self.get_gyro_integrated_z()
        print(f"[GyroZThreshold] integrated={integrated:.2f}, threshold={threshold}, direction={direction}")
        if direction == 'right':
            return integrated <= -threshold
        elif direction == 'left':
            return integrated >= threshold
        raise ValueError("direction must be 'left' or 'right'")

    def get_yaw(self) -> float:
        """
        ヨー角（x軸）を一発取得。Noneは許さず必ずfloat型で返す（未取得時は0.0）。
        """
        status = self.get_spike_status()
        val = getattr(getattr(status.sensors, "yaw_pitch_roll", None), "x", None)
        return float(val) if val is not None else 0.0

    def set_start_yaw(self):
        """
        現在のヨー角をstart_yawとして記録する（直線安定化・旋回開始時などで使用）
        """
        self._start_yaw = self.get_yaw()

    def get_start_yaw(self) -> float:
        """
        記録済みのstart_yaw（開始時ヨー角）を取得。未設定時は0.0。
        """
        return getattr(self, '_start_yaw', 0.0)

    def is_yaw_turn_finished(self, side: str = "right", threshold_deg: float = 90.0) -> bool:
        """
        ヨー角による片側旋回の停止判定のみ返す。
        Args:
            side (str): 'left' または 'right'（旋回方向）
            threshold_deg (float): 停止判定の閾値（度、絶対値で指定）
        Returns:
            bool: 停止すべきならTrue
        """
        yaw_val = self.get_yaw()
        yaw_start_val = self.get_start_yaw()
        diff = yaw_val - yaw_start_val
        if side == "left":
            return diff <= -abs(threshold_deg)
        elif side == "right":
            return diff >= abs(threshold_deg)
        else:
            raise ValueError("side must be 'left' or 'right'")

    def yaw_straight_control(self, base_speed: int = 80, kp: float = 1.0, deadband: float = 2.0) -> tuple[int, int]:
        """
        ヨー角による直線安定化制御（P制御、内部start_yaw基準）。
        Args:
            base_speed (int): 基本速度
            kp (float): 比例ゲイン
            deadband (float): デッドバンド幅
        Returns:
            (left_speed, right_speed): 補正後の左右速度
        """
        yaw = self.get_yaw()
        start_yaw = self.get_start_yaw()
        error = yaw - start_yaw
        pid_output = kp * error
        if abs(error) <= deadband:
            left_speed = base_speed
            right_speed = base_speed
        else:
            left_speed = base_speed
            right_speed = base_speed
            if pid_output > 0:
                left_speed = base_speed - abs(pid_output)
            elif pid_output < 0:
                right_speed = base_speed - abs(pid_output)
            left_speed = int(max(min(left_speed, base_speed), base_speed-2))
            right_speed = int(max(min(right_speed, base_speed), base_speed-2))
        return left_speed, right_speed

