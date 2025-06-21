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
                # デバッグ: 受信した生データを表示（一定間隔で）
                if not hasattr(self, '_debug_counter'):
                    self._debug_counter = 0
                self._debug_counter += 1
                
                if self._debug_counter % 100 == 0:  # 100回に1回
                    print(f"ETRobot受信生データ: {received_data.decode('utf-8', errors='ignore')[:200]}")  # 先頭200文字のみ
                
                # 50回ごと（約1秒ごと）に受信頻度確認
                if self._debug_counter % 50 == 0:
                    print(f"ETRobot: 受信カウント={self._debug_counter}, データ長={len(received_data)}")
                
                # Update the spike status with the new data
                self.spike_status.update(received_data)
                # Update last_spike_status with valid sensor readings
                self._update_last_spike_status()
                
                # Debug: Print update confirmation (uncomment for debugging)
                # print(f"Updated spike_status - Distance: {self.spike_status.sensors.distance}, Color: {self.spike_status.sensors.color}")

            except Exception as e:
                # Only print error if it's not a simple JSON parsing issue
                if "syntax error in JSON" not in str(e):
                    print(f"JSON parsing error: {e}")
                    print(f"Raw data: {received_data}")
                    print(f"Data length: {len(received_data)}")
                    print(f"Data type: {type(received_data)}")
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
            # 有効な超音波センサー値の場合
            # 無効値カウンターをリセット
            if hasattr(self, '_ultrasonic_invalid_counter'):
                if self._ultrasonic_invalid_counter > 0:
                    elapsed_time = time.time() - self.start_time
                    print(f"ETRobot: 超音波センサー復帰 [{elapsed_time:.1f}s] - 無効値カウンター{self._ultrasonic_invalid_counter}をリセット")
                self._ultrasonic_invalid_counter = 0
            
            # 超音波センサーの値が変化したときの詳細ログ
            if hasattr(last.sensors, 'distance') and last.sensors.distance != current.sensors.distance:
                elapsed_time = time.time() - self.start_time
                print(f"ETRobot: 超音波センサー値変化検知 [{elapsed_time:.1f}s] - {last.sensors.distance} → {current.sensors.distance}")
            
            last.sensors.distance = current.sensors.distance
              # 定期的な超音波センサー状態確認
            if not hasattr(self, '_ultrasonic_debug_counter'):
                self._ultrasonic_debug_counter = 0
            self._ultrasonic_debug_counter += 1
            
            if self._ultrasonic_debug_counter % 100 == 0:  # 100回に1回
                elapsed_time = time.time() - self.start_time
                print(f"ETRobot: 超音波センサー定期確認 [{elapsed_time:.1f}s] - 現在値={current.sensors.distance}, 最後の有効値={last.sensors.distance}")
        else:
            # 超音波センサーが無効な値の場合
            if not hasattr(self, '_ultrasonic_invalid_counter'):
                self._ultrasonic_invalid_counter = 0
            self._ultrasonic_invalid_counter += 1
            
            # 5回ごとに状況を確認（より頻繁に監視）
            if self._ultrasonic_invalid_counter % 5 == 0:
                elapsed_time = time.time() - self.start_time
                print(f"ETRobot: 超音波センサー無効値検知 [{elapsed_time:.1f}s] - current.sensors.distance={current.sensors.distance}, type={type(current.sensors.distance)}, 連続回数={self._ultrasonic_invalid_counter}")
              # 10回連続でNoneの場合、last_spike_statusもNoneにリセット
            if self._ultrasonic_invalid_counter == 10:  # ちょうど10回目でリセット実行
                old_value = last.sensors.distance
                last.sensors.distance = None
                elapsed_time = time.time() - self.start_time
                if old_value is not None:
                    print(f"ETRobot: 超音波センサー値をリセット [{elapsed_time:.1f}s] - {old_value} → None （10回連続無効値のため）")
                else:
                    print(f"ETRobot: 超音波センサー値リセット確認 [{elapsed_time:.1f}s] - 既にNone状態 （10回連続無効値のため）")
            elif self._ultrasonic_invalid_counter > 10:
                # 10回を超えた場合はカウンターを10に固定（リセット済み状態を維持）
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
                # Skip update if color sensor data is malformed
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
        return self.last_spike_status

    def set_motor_relative_position(
        self, left_positon: int, right_position: int
    ) -> None:
        id_byte = self.COMMAND_SET_MOTOR_RELATIVE_POSITION_ID.to_bytes(1, "big")
        parameter1_byte = left_positon.to_bytes(1, "big")
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

    def stop(self) -> None:
        """Stop the robot and close the serial port."""
        self.is_running = False
        self.brake()
        self.__thread.join()
        self.__serial_port.close()
