"""Main controlling program for LEGO Spike Prime Hub."""

import gc
import time

import hub  # type: ignore
import uasyncio  # type: ignore

# Command ID list
COMMAND_SET_MOTOR_FORWARD_SPEED_ID = 201
COMMAND_SET_MOTOR_BACKWARD_SPEED_ID = 202
COMMAND_SET_MOTOR_RELATIVE_POSITION_ID = 203
COMMAND_STOP_MOTOR_ID = 204
COMMAND_MOVE_ARM_ID = 205

CMD_FLAG = b"CF:"

PORT_MAP = {
    "motor_right": "A",
    "motor_left": "B",
    "motor_arm": "C",
    "force_sensor": "D",
    "color_sensor": "E",
    "ultrasonic_sensor": "F",
}


class LegoSpike:
    """Class to control LEGO Spike Prime Hub.

    This class initializes the hub, sets up motors and sensors, and provides methods to read commands
    from USB and execute them.
    It also includes methods to control the motors and arm, and to handle the command execution logic.
    """

    def __init__(self) -> None:
        """Initialize the LEGO Spike Prime Hub and its components."""
        hub.display.show(
            hub.Image.ALL_CLOCKS, delay=400, clear=True, wait=False, loop=True, fade=0
        )
        hub.motion.align_to_model(hub.TOP, hub.FRONT)  # GYRO, orientation
        hub.motion.yaw_pitch_roll(0)  # yaw, pitch and roll

        # Set ports
        self.motor_arm = getattr(hub.port, PORT_MAP["motor_arm"]).motor
        self.motor_right = getattr(hub.port, PORT_MAP["motor_right"]).motor
        self.motor_left = getattr(hub.port, PORT_MAP["motor_left"]).motor
        self.color_sensor = getattr(hub.port, PORT_MAP["color_sensor"]).device
        self.force_sensor = getattr(hub.port, PORT_MAP["force_sensor"]).device
        self.ultrasonic_sensor = getattr(hub.port, PORT_MAP["ultrasonic_sensor"]).device

        self.usb = hub.USB_VCP()
        self.usb.setinterrupt(-1)

        # Setup the serial port(take 1 second), and transferring a maximum of 115200 bits per second.
        time.sleep(1)

        # Set motors mode to measure its relative position on boot
        self._set_motor_relative_position(left_position=0, right_position=0)

        hub.display.show(hub.Image.HAPPY)

    def read_command(self) -> tuple[int | None, int | None, int | None]:
        command_id = None
        command_parameter1 = None
        command_parameter2 = None

        """Read command from USB and return the command ID and parameters."""
        if self.usb.any():
            data = self.usb.read(6)  # Read "CF:" + 3 bytes command data

            flag_pos = data.find(CMD_FLAG)

            if (
                flag_pos >= 0 and len(data) >= flag_pos + 6
            ):  # Ensure we have enough bytes
                raw_bytes = data[
                    flag_pos + 3 : flag_pos + 6
                ]  # Extract the 3 bytes after "CF:"
                command_id = int.from_bytes(raw_bytes[0:1], "big")
                command_parameter1 = int.from_bytes(raw_bytes[1:2], "big")
                command_parameter2 = int.from_bytes(raw_bytes[2:3], "big")

                return command_id, command_parameter1, command_parameter2

        # If no command is read, return None values
        return None, None, None

    def execute_command(
        self, command_id: int, command_parameter1: int, command_parameter2: int
    ) -> None:
        if command_id == COMMAND_SET_MOTOR_FORWARD_SPEED_ID:
            self._set_motor_speed(command_parameter1, command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_BACKWARD_SPEED_ID:
            self._set_motor_speed(-command_parameter1, -command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_RELATIVE_POSITION_ID:
            self._set_motor_relative_position(command_parameter1, command_parameter2)
        elif command_id == COMMAND_STOP_MOTOR_ID:
            self.motor_left.brake()
            self.motor_right.brake()
        elif command_id == COMMAND_MOVE_ARM_ID:
            self._move_arm(command_parameter1)

    def _set_motor_speed(self, left_speed: int, right_speed: int) -> None:
        """Method to control the steering wheel angle.

        Args:
            left_speed: Left wheel speed(0~100)
            right_speed: Right wheel speed(0~100)
        """
        self.motor_left.run_at_speed(-int(left_speed))
        self.motor_right.run_at_speed(int(right_speed))

    def _set_motor_relative_position(
        self, left_position: int, right_position: int
    ) -> None:
        self.motor_left.preset(-int(left_position))
        self.motor_right.preset(int(right_position))

    def _move_arm(self, action: int) -> None:
        """Method to move the arm motor up or down and set its current position using preset.

        Args:
            action: Action to perform (0 = move up, 1 = move down)
        """
        if action == 0:  # Move down
            self.motor_arm.run_at_speed(
                40
            )  # Encapulate int() to ensure speed is an integer
        elif action == 1:  # Move up
            self.motor_arm.run_at_speed(
                -40
            )  # Encapulate int() to ensure speed is an integer
        elif action == 2:  # Stop arm
            self.motor_arm.brake()


async def receiver() -> None:
    while True:
        try:
            command_id, command_parameter1, command_parameter2 = (
                lego_spike.read_command()
            )
        except Exception:
            command_id = None
            command_parameter1 = None
            command_parameter2 = None

        if (
            command_id != None
            and command_parameter1 != None
            and command_parameter2 != None
        ):
            lego_spike.execute_command(
                command_id, command_parameter1, command_parameter2
            )

        await uasyncio.sleep(0.01)  # Sleep for 10ms to reduce CPU usage


async def main_task() -> None:
    tasks = []

    receiver_task = uasyncio.create_task(receiver())
    tasks.append(receiver_task)

    # Run indefinitely - let the receiver task handle commands continuously
    await receiver_task


# Trigger a garbage collection cycle
gc.collect()

print("Starting LEGO Prime Hub..")

try:
    lego_spike = LegoSpike()
    uasyncio.run(main_task())
except SystemExit as e:
    print(e)
