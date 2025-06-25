import time
import serial
import threading
from .spike_status import SpikeStatus


class ETRobot(
    object
):  # Command IDs should be as same as (`spike/main.py`) the script in LEGO Spike Prime.
    COMMAND_SET_MOTOR_FORWARD_POWER_ID = 201
    COMMAND_SET_MOTOR_BACKWARD_POWER_ID = 202
    COMMAND_SET_MOTOR_RELATIVE_POSITION_ID = 203
    COMMAND_STOP_MOTOR_ID = 204
    COMMAND_MOVE_ARM_ID = 205
    COMMAND_SET_MOTOR_DEGREES_ID = 206

    CMD_FLAG = b"CF:"

    DUMMY = 1  # Dummy value for parameters

    def __init__(self, port="/dev/ttyACM0") -> None:
        self.__serial_port = serial.Serial(port=port, baudrate=115200, timeout=2)
        self.__serial_port.reset_input_buffer()
        self.__serial_port.reset_output_buffer()

        self.is_running = True
        # Create a SpikeStatus object for handling data
        self.spike_status = SpikeStatus()        # Keep track of last valid sensor data to handle missing readings
        self.last_spike_status = SpikeStatus()
        
        # 動作開始時間を記録
        self.start_time = time.time()

        self.__thread = threading.Thread(target=self.__update_status)
        self.__thread.start()

    def __send_command(self, command) -> None:
        """Send a command to the robot via the serial port."""
        # デバッグ出力なし
        self.__serial_port.write(self.CMD_FLAG + command)

    def __update_status(self) -> None:
        """Background thread that continuously receives data from the serial connection."""
        while self.is_running:
            self.receive()

    def receive(self) -> None:
        """
        Update ETRobot motor and sensor status by the received data from the GPIO port.

        Note:
            The update rate should be less than the rate of sending sensor data in LEGO Prime Hub (0.0005 seconds).        """
        received_data = self.__serial_port.read_until(expected=b"\r")        # Skip empty data
        if not received_data or len(received_data.strip()) == 0:
            return

        if received_data.startswith(b"{"):
            try:
                self.spike_status.update(received_data)
                self._update_last_spike_status()
            except Exception as e:
                # Only print error if it's not a simple JSON parsing issue
                if "syntax error in JSON" not in str(e):
                    pass
        elif received_data.startswith(b'{"i":null,"e":'):
            # Handle error messages from Spike (ignore these for now)
            pass

        time.sleep(0.0001)  # Value should be less than '0.0005' seconds

    def _update_last_spike_status(self) -> None:
        """
        Update last_spike_status with current valid sensor readings.
        Only updates values that are not None to preserve last known good values.
        """
        current = self.spike_status
        last = self.last_spike_status        # Always update timestamp and message type
        last.timestamp = current.timestamp
        last.message_type = current.message_type
        last.raw_data = current.raw_data        # Update sensors with valid readings
        if current.sensors.distance is not None and isinstance(current.sensors.distance, (int, float)):
            if hasattr(self, '_ultrasonic_invalid_counter'):
                self._ultrasonic_invalid_counter = 0
            last.sensors.distance = current.sensors.distance
            if not hasattr(self, '_ultrasonic_debug_counter'):
                self._ultrasonic_debug_counter = 0
            self._ultrasonic_debug_counter += 1
        else:
            if not hasattr(self, '_ultrasonic_invalid_counter'):
                self._ultrasonic_invalid_counter = 0
            self._ultrasonic_invalid_counter += 1
            if self._ultrasonic_invalid_counter == 10:
                old_value = last.sensors.distance
                last.sensors.distance = None
            elif self._ultrasonic_invalid_counter > 10:
                self._ultrasonic_invalid_counter = 10
        if current.sensors.force is not None and isinstance(current.sensors.force, (int, float)):
            last.sensors.force = current.sensors.force# Update color sensor data
        if current.sensors.color:
            if not last.sensors.color:
                from .spike_status import ColorSensorStatus
                last.sensors.color = ColorSensorStatus()
            try:
                if current.sensors.color.reflected is not None and isinstance(current.sensors.color.reflected, (int, float)):
                    last.sensors.color.reflected = current.sensors.color.reflected
                if current.sensors.color.ambient is not None and isinstance(current.sensors.color.ambient, (int, float)):
                    last.sensors.color.ambient = current.sensors.color.ambient
                if current.sensors.color.color is not None and isinstance(current.sensors.color.color, (int, float)):
                    last.sensors.color.color = current.sensors.color.color
            except AttributeError:
                pass
        # Update gyro data
        if current.sensors.gyro:
            if not last.sensors.gyro:
                from .spike_status import VectorStatus
                last.sensors.gyro = VectorStatus()
            try:
                if isinstance(current.sensors.gyro.x, (int, float)):
                    last.sensors.gyro.x = current.sensors.gyro.x
                if isinstance(current.sensors.gyro.y, (int, float)):
                    last.sensors.gyro.y = current.sensors.gyro.y
                if isinstance(current.sensors.gyro.z, (int, float)):
                    last.sensors.gyro.z = current.sensors.gyro.z
            except AttributeError:
                pass
        # Update accelerometer data
        if current.sensors.accelerometer:
            if not last.sensors.accelerometer:
                from .spike_status import VectorStatus
                last.sensors.accelerometer = VectorStatus()
            try:
                if isinstance(current.sensors.accelerometer.x, (int, float)):
                    last.sensors.accelerometer.x = current.sensors.accelerometer.x
                if isinstance(current.sensors.accelerometer.y, (int, float)):
                    last.sensors.accelerometer.y = current.sensors.accelerometer.y
                if isinstance(current.sensors.accelerometer.z, (int, float)):
                    last.sensors.accelerometer.z = current.sensors.accelerometer.z
            except AttributeError:
                pass        # Update position data
        if current.sensors.position:
            if not last.sensors.position:
                from .spike_status import Position
                last.sensors.position = Position()
            last.sensors.position.x = current.sensors.position.x
            last.sensors.position.y = current.sensors.position.y
        # Update motor data (always update as these are more reliable)
        for motor_id in ["A", "B", "C"]:
            if motor_id in current.motors and motor_id in last.motors:
                if current.motors[motor_id].position is not None:
                    last.motors[motor_id].position = current.motors[motor_id].position
                if current.motors[motor_id].relative_position is not None:
                    last.motors[motor_id].relative_position = current.motors[motor_id].relative_position
                if current.motors[motor_id].speed is not None:
                    last.motors[motor_id].speed = current.motors[motor_id].speed
                if current.motors[motor_id].power is not None:
                    last.motors[motor_id].power = current.motors[motor_id].power
        # Update battery data
        if current.battery:
            if current.battery.voltage is not None:
                last.battery.voltage = current.battery.voltage
            if current.battery.percent is not None:
                last.battery.percent = current.battery.percent

    def get_spike_status(self):
        """
        Get the spike status with last known good sensor values.

        Returns:
            SpikeStatus: Spike status object with consistent sensor data
        """
        # 直近のType:0（センサーデータ）だけを返すように修正
        while True:
            received_data = self.__serial_port.read_until(expected=b"\r")
            if not received_data or len(received_data.strip()) == 0:
                continue
            # print(f"[DEBUG][get_spike_status] raw bytes: {repr(received_data)}")  # デバッグ用
            # 複数JSONが連結している場合に分割
            for chunk in received_data.split(b'}{'):
                if not chunk:
                    continue
                if not chunk.startswith(b'{'):
                    chunk = b'{' + chunk
                if not chunk.endswith(b'}'):
                    chunk = chunk + b'}'
                try:
                    from nnspike.unit.spike_status import SpikeStatus
                    status = SpikeStatus(chunk)
                    # print(f"[DEBUG][get_spike_status] message_type: {status.message_type}")  # デバッグ用
                    if status.message_type == 0:
                        self.last_spike_status = status
                        return status
                except Exception as e:
                    # print(f"[DEBUG][get_spike_status] parse error: {e}")  # デバッグ用
                    continue

    def set_motor_relative_position(
        self, left_position: int, right_position: int
    ) -> None:
        id_byte = self.COMMAND_SET_MOTOR_RELATIVE_POSITION_ID.to_bytes(1, "big")
        parameter1_byte = left_position.to_bytes(1, "big")
        parameter2_byte = right_position.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        self.__send_command(command)

    def set_motor_forward_power(self, left_power: int, right_power: int) -> None:
        """
        Set the ETRobot motor's power.

        Args:
            left_power (int): Left motor power (0-100).
            right_power (int): Right motor power (0-100).
        """
        # 送信最適化: 値が変化しない場合は1000msごとに1回のみ送信
        now = time.time()
        if not hasattr(self, '_last_forward_power'):
            self._last_forward_power = (None, None)
            self._last_forward_power_time = 0.0
        # 値が変化した場合は即送信
        if self._last_forward_power != (left_power, right_power):
            self._last_forward_power = (left_power, right_power)
            self._last_forward_power_time = now
        # 値が変化していない場合は、前回送信から1秒(1000ms)経過していれば送信
        elif now - self._last_forward_power_time < 1.0:
            return
        else:
            self._last_forward_power_time = now
        id_byte = self.COMMAND_SET_MOTOR_FORWARD_POWER_ID.to_bytes(1, "big")
        parameter1_byte = left_power.to_bytes(1, "big")
        parameter2_byte = right_power.to_bytes(1, "big")
        command = id_byte + parameter1_byte + parameter2_byte
        self.__send_command(command)

    def set_motor_backward_power(self, left_power: int, right_power: int) -> None:
        """
        Set the ETRobot motor's power in reverse direction.

        Args:
            left_power (int): Left motor power (0-100).
            right_power (int): Right motor power (0-100).
        """
        id_byte = self.COMMAND_SET_MOTOR_BACKWARD_POWER_ID.to_bytes(1, "big")
        parameter1_byte = left_power.to_bytes(1, "big")
        parameter2_byte = right_power.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        self.__send_command(command)

    def brake(self) -> None:
        """Brake the motors of the ETRobot."""
        id_byte = self.COMMAND_STOP_MOTOR_ID.to_bytes(1, "big")
        parameter1_byte = self.DUMMY.to_bytes(1, "big")
        parameter2_byte = self.DUMMY.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        self.__send_command(command)

    def move_arm(self, action: int) -> None:
        """
        Move the arm up or down.

        Args:
            action (int): Action to perform (0 = move down, 1 = move up).
        """
        id_byte = self.COMMAND_MOVE_ARM_ID.to_bytes(1, "big")
        parameter1_byte = action.to_bytes(1, "big")
        parameter2_byte = self.DUMMY.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        self.__send_command(command)

    def set_motor_degrees(self, left_degrees: int, right_degrees: int) -> None:
        """
        左右モーターを指定した角度（degree）だけ回転させる（符号付き2バイト値、符号反転なし）。
        Args:
            left_degrees (int): 左モーターの回転角度（degree, 負値で逆転）
            right_degrees (int): 右モーターの回転角度（degree, 負値で逆転）
        """
        id_byte = self.COMMAND_SET_MOTOR_DEGREES_ID.to_bytes(1, "big")
        parameter1_bytes = int(left_degrees).to_bytes(2, "big", signed=True)
        parameter2_bytes = int(right_degrees).to_bytes(2, "big", signed=True)
        command = id_byte + parameter1_bytes + parameter2_bytes
        self.__send_command(command)

    def stop(self) -> None:
        """Stop the robot and close the serial port. (何もしない: 強制停止・再起動禁止)"""
        pass

    def turn_left(self, degree, power):
        """
        左に指定角度だけ回転する（degree単位、powerは回転速度）。
        呼び出し側でdegree, powerを必ず指定すること。
        """
        time_per_degree = 0.5 / 90
        self.set_motor_forward_power(left_power=0, right_power=power)
        time.sleep(abs(degree) * time_per_degree)
        self.brake()

    def turn_right(self, degree, power):
        """
        右に指定角度だけ回転する（degree単位、powerは回転速度）。
        呼び出し側でdegree, powerを必ず指定すること。
        """
        time_per_degree = 0.5 / 90
        self.set_motor_forward_power(left_power=power, right_power=0)
        time.sleep(abs(degree) * time_per_degree)
        self.brake()

    def move_forward(self, duration, power):
        """
        指定時間だけ前進する。
        呼び出し側でduration, powerを必ず指定すること。
        """
        self.set_motor_forward_power(left_power=power, right_power=power)
        time.sleep(duration)
        self.brake()

    def move_backward(self, duration, power):
        """
        指定時間だけ後退する。
        呼び出し側でduration, powerを必ず指定すること。
        """
        self.set_motor_backward_power(left_power=power, right_power=power)
        time.sleep(duration)
        self.brake()

    def move_left_arc(self, duration, power):
        """
        左カーブで前進（左モーター弱・右モーター強）。
        呼び出し側でduration, powerを必ず指定すること。
        """
        self.set_motor_forward_power(left_power=int(power*0.8), right_power=power)
        time.sleep(duration)
        self.brake()

    def move_right_arc(self, duration, power):
        """
        右カーブで前進（右モーター弱・左モーター強）。
        呼び出し側でduration, powerを必ず指定すること。
        """
        self.set_motor_forward_power(left_power=power, right_power=int(power*0.8))
        time.sleep(duration)
        self.brake()
