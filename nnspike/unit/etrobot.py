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

    CMD_FLAG = b"CF:"

    DUMMY = 1  # Dummy value for parameters

    def __init__(self, port="/dev/ttyACM0") -> None:
        self.__serial_port = serial.Serial(port=port, baudrate=115200, timeout=2)
        self.__serial_port.reset_input_buffer()
        self.__serial_port.reset_output_buffer()

        self.is_running = True
        # Create a SpikeStatus object for handling data
        self.spike_status = SpikeStatus()

        # Keep track of last valid sensor data to handle missing readings
        self.last_spike_status = SpikeStatus()

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
            The update rate should be less than the rate of sending sensor data in LEGO Prime Hub (0.0005 seconds).
        """
        received_data = self.__serial_port.read_until(expected=b"\r")

        if received_data.startswith(b"{"):
            try:
                # Update the spike status with the new data
                self.spike_status.update(received_data)
                # Update last_spike_status with valid sensor readings
                self.__update_last_spike_status()

            except Exception as e:
                print(f"Error processing data: {e}")
                print(f"Raw data: {received_data}")

        time.sleep(0.01)

    def __update_last_spike_status(self) -> None:
        """
        Update last_spike_status with current valid sensor readings.
        Only updates values that are not None to preserve last known good values.
        """
        current = self.spike_status
        last = self.last_spike_status

        # Update sensors with valid readings
        # PERFORMANCE OPTIMIZATION: Comment out unused sensors
        # Distance sensor (UNUSED - disabled for performance)
        # if current.sensors.distance is not None:
        #     last.sensors.distance = current.sensors.distance
        # else:
        #     last.sensors.distance = 0

        if current.sensors.force is not None:
            last.sensors.force = current.sensors.force

        # Color sensor data (UNUSED - disabled for performance)
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

        # Gyro data (UNUSED - disabled for performance)
        # if current.sensors.gyro:
        #     if not last.sensors.gyro:
        #         from .spike_status import VectorStatus
        # 
        #         last.sensors.gyro = VectorStatus()
        #     if current.sensors.gyro.x is not None:
        #         last.sensors.gyro.x = current.sensors.gyro.x
        #     if current.sensors.gyro.y is not None:
        #         last.sensors.gyro.y = current.sensors.gyro.y
        #     if current.sensors.gyro.z is not None:
        #         last.sensors.gyro.z = current.sensors.gyro.z

        # Accelerometer data (UNUSED - disabled for performance)
        # if current.sensors.accelerometer:
        #     if not last.sensors.accelerometer:
        #         from .spike_status import VectorStatus
        # 
        #         last.sensors.accelerometer = VectorStatus()
        #     if current.sensors.accelerometer.x is not None:
        #         last.sensors.accelerometer.x = current.sensors.accelerometer.x
        #     if current.sensors.accelerometer.y is not None:
        #         last.sensors.accelerometer.y = current.sensors.accelerometer.y
        #     if current.sensors.accelerometer.z is not None:
        #         last.sensors.accelerometer.z = current.sensors.accelerometer.z

        # Position data (UNUSED - disabled for performance)
        # if current.sensors.position:
        #     if not last.sensors.position:
        #         from .spike_status import Position
        # 
        #         last.sensors.position = Position()
        #     if current.sensors.position.x is not None:
        #         last.sensors.position.x = current.sensors.position.x
        #     if current.sensors.position.y is not None:
        #         last.sensors.position.y = current.sensors.position.y

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

        # Update battery data (for HIGH_SPEED_BASE optimization)
        # コメントアウト：get_spike_statusの負荷軽減のため
        # if current.battery.voltage is not None:
        #     last.battery.voltage = current.battery.voltage
        # if current.battery.percent is not None:
        #     last.battery.percent = current.battery.percent
        
        # Also update message_type and raw_data for debugging
        last.message_type = current.message_type
        last.raw_data = current.raw_data

    def get_spike_status(self):
        """
        Get the spike status with last known good sensor values.

        Returns:
            SpikeStatus: Spike status object with consistent sensor data
        """
        return self.last_spike_status

    def set_motor_relative_position(self, left_positon: int, right_position: int) -> None:
        id_byte = self.COMMAND_SET_MOTOR_RELATIVE_POSITION_ID.to_bytes(1, "big")
        parameter1_byte = left_positon.to_bytes(1, "big")
        parameter2_byte = right_position.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        start_time = time.time()
        while time.time() - start_time < 0.5:
            self.__send_command(command)
            time.sleep(0.05)

    def retrieve_motors_relative_position(self) -> int:
        """
        Retrieve the relative positions of the motors.

        Returns:
            int: The sum of the absolute values of the relative positions of both motors.
        """
        status = self.get_spike_status()

        if status.motors["A"].relative_position is not None:
            motor_a_position = abs(status.motors["A"].relative_position)
        else:
            motor_a_position = 0

        if status.motors["B"].relative_position is not None:
            motor_b_position = abs(status.motors["B"].relative_position)
        else:
            motor_b_position = 0

        return motor_a_position + motor_b_position

    def set_motor_speed(self, left_speed: int, right_speed: int) -> None:
        """
        Set the ETRobot motor's speed.

        Args:
            left_speed (int): Left motor speed (-100-100).
            right_speed (int): Right motor speed (-100-100).
        """
        if left_speed < -100 or left_speed > 100 or right_speed < -100 or right_speed > 100:
            raise ValueError("Motor speeds must be between -100 and 100.")

        if left_speed < 0 or right_speed < 0:
            self.set_motor_backward_speed(-left_speed, -right_speed)
        else:
            self.set_motor_forward_speed(left_speed, right_speed)

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
