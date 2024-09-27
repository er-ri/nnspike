import time
import serial
import threading


class ETRobot(object):

    # Command IDs should be as same as (`spike/main.py`) the script in LEGO Spike Prime.
    COMMAND_MOTOR_ID = 201
    COMMAND_ARM_ID = 202
    COMMAND_RESET_MOTOR_ID = 203

    def __init__(self) -> None:
        self.__serial_port = serial.Serial(
            port="/dev/ttyAMA5", baudrate=115200, timeout=2
        )
        self.__serial_port.reset_input_buffer()
        self.__serial_port.reset_output_buffer()

        self.color_sensor = None
        self.motor_count = None
        self.ultrasonic_sensor = None
        self.is_running = True

        self.__thread = threading.Thread(target=self.__update_status)
        self.__thread.start()

    def __send_command(self, command) -> None:
        """Send a command to the robot via the serial port."""
        self.__serial_port.write(command)

    def __update_status(self) -> None:
        """
        Update ETRobot motor and sensor status by the received data from the GPIO port.

        Note:
            The update rate should be less than the rate of sending sensor data in LEGO Prime Hub (0.0005 seconds).
        """
        while self.is_running:
            status_data = self.__serial_port.read(4)

            if status_data != b"":
                color_sensor = int.from_bytes(status_data[0:1], "big")
                motor_count = int.from_bytes(status_data[1:3], "big")
                ultrasonic_sensor = int.from_bytes(status_data[3:4], "big")

                self.color_sensor = color_sensor
                self.motor_count = motor_count
                self.ultrasonic_sensor = ultrasonic_sensor
            time.sleep(0.0001)  # Value should be less than '0.0005' seconds

    def reset_motor(self) -> None:
        id_byte = self.COMMAND_RESET_MOTOR_ID.to_bytes(1, "big")
        dummy = 1
        parameter1_byte = dummy.to_bytes(1, "big")
        parameter2_byte = dummy.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        self.__send_command(command)

    def set_motor_power(self, left_power: int, right_power: int) -> None:
        """
        Set the ETRobot motor's power.

        Args:
            left_power (int): Left motor power (0-100).
            right_power (int): Right motor power (0-100).
        """
        id_byte = self.COMMAND_MOTOR_ID.to_bytes(1, "big")
        parameter1_byte = left_power.to_bytes(1, "big")
        parameter2_byte = right_power.to_bytes(1, "big")

        command = id_byte + parameter1_byte + parameter2_byte

        self.__send_command(command)

    def stop(self) -> None:
        """Stop the robot and close the serial port."""
        self.is_running = False
        self.set_motor_power(0, 0)
        self.__thread.join()
        self.__serial_port.close()
