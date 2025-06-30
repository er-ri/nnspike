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
        last = self.last_spike_status
        # --- シンプル同期: Noneでなければ上書きするだけ ---
        if current.sensors.distance is not None:
            last.sensors.distance = current.sensors.distance
        if current.sensors.force is not None:
            last.sensors.force = current.sensors.force
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
        if current.sensors.gyro:
            if not last.sensors.gyro:
                from .spike_status import VectorStatus
                last.sensors.gyro = VectorStatus()
            if current.sensors.gyro.x is not None:
                last.sensors.gyro.x = current.sensors.gyro.x
            if current.sensors.gyro.y is not None:
                last.sensors.gyro.y = current.sensors.gyro.y
            if current.sensors.gyro.z is not None:
                last.sensors.gyro.z = current.sensors.gyro.z
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
        if current.sensors.position:
            if not last.sensors.position:
                from .spike_status import Position
                last.sensors.position = Position()
            if current.sensors.position.x is not None:
                last.sensors.position.x = current.sensors.position.x
            if current.sensors.position.y is not None:
                last.sensors.position.y = current.sensors.position.y
        for motor_id in ["A", "B", "C"]:
            if current.motors[motor_id].position is not None:
                last.motors[motor_id].position = current.motors[motor_id].position
            if current.motors[motor_id].relative_position is not None:
                last.motors[motor_id].relative_position = current.motors[motor_id].relative_position
            if current.motors[motor_id].speed is not None:
                last.motors[motor_id].speed = current.motors[motor_id].speed
            if current.motors[motor_id].power is not None:
                last.motors[motor_id].power = current.motors[motor_id].power
        # --- バッテリー情報の更新は維持 ---
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
        return self.last_spike_status

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
        """Stop the robot and close the serial port. (スレッド安全停止)"""
        self.is_running = False
        if hasattr(self, "__thread") and self.__thread.is_alive():
            self.__thread.join(timeout=2.0)
        if hasattr(self, "__serial_port") and self.__serial_port.is_open:
            try:
                self.__serial_port.close()
            except Exception as e:
                pass
