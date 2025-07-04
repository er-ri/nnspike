# LEGO type:standard slot:2 autostart
"""Main controlling program for LEGO Spike Prime Hub"""
import gc
import hub# type: ignore
import time
import uasyncio# type: ignore

MAX_IDLE_TIME = 120000# Maximum idle time, unit: millisecond
MAX_RUN_TIME = 600# Maximum running time, unit: second

# Command ID list
COMMAND_SET_MOTOR_FORWARD_POWER_ID = 201
COMMAND_SET_MOTOR_BACKWARD_POWER_ID = 202
COMMAND_SET_MOTOR_RELATIVE_POSITION_ID = 203
COMMAND_SET_MOTOR_DEGREES_ID = 206
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


class LegoSpike(object):
    """LEGO Spike Prime Hub

    Class for controlling all the devices in the Spike car by receiving
    commands from Raspberry Pi. Every command is made up of 2 bytes, the
    first byte indicates the command id while the second byte represents
    the corresponding parameters as shown below.

    | Device | Command Id | Parameter1 | Parameter2 |
    | Motor | 0 | Power: 0~180 | Steering: -90 ~ 90|

    """

    def __init__(self) -> None:
        # Initialization
        hub.display.show(
            hub.Image.ALL_CLOCKS, delay=400, clear=True, wait=False, loop=True, fade=0
        )
        hub.motion.align_to_model(hub.TOP, hub.FRONT)# GYRO, orientation
        hub.motion.yaw_pitch_roll(0)# yaw, pitch and roll

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

        # Millisecond counter for record the latest command executed time, maximum idle time
        self.command_counter = time.ticks_ms()

        hub.display.show(hub.Image.YES)

    def read_command(self):
        command_id = None
        command_parameter1 = None
        command_parameter2 = None

        """Read command from USB and return the command ID and parameters."""
        if self.usb.any():
            data = self.usb.read(6)# Increased to read "ET:" + 3 bytes command data

            flag_pos = data.find(CMD_FLAG)

            if (
                flag_pos >= 0 and len(data) >= flag_pos + 6
            ):# Ensure we have enough bytes
                raw_bytes = data[
                    flag_pos + 3 : flag_pos + 6
                ]# Extract the 3 bytes after "CF:"
                command_id = int.from_bytes(raw_bytes[0:1], "big")
                command_parameter1 = int.from_bytes(raw_bytes[1:2], "big")
                command_parameter2 = int.from_bytes(raw_bytes[2:3], "big")
                self.command_counter = time.ticks_ms()

        return command_id, command_parameter1, command_parameter2

    def execute_command(self, command_id, command_parameter1, command_parameter2):
        if command_id == COMMAND_SET_MOTOR_FORWARD_POWER_ID:
            self._set_motor_speed(command_parameter1, command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_BACKWARD_POWER_ID:
            self._set_motor_speed(-command_parameter1, -command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_RELATIVE_POSITION_ID:
            self._set_motor_relative_position(command_parameter1, command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_DEGREES_ID:
            self._set_motor_degrees(command_parameter1, command_parameter2)
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
        self.command_counter = time.ticks_ms()

        self.motor_left.run_at_speed(-int(left_speed))
        self.motor_right.run_at_speed(int(right_speed))

    def _set_motor_relative_position(
        self, left_position: int, right_position: int
    ) -> None:
        self.command_counter = time.ticks_ms()

        self.motor_left.preset(left_position)
        self.motor_right.preset(right_position)

    def _move_arm(self, action: int) -> None:
        """Method to move the arm motor up or down and set its current position using preset.

        Args:
            action: Action to perform (0 = move down, 1 = move up)
        """
        self.command_counter = time.ticks_ms()

        if action == 0:# Move down
            # アームを下げる: 速度50で回転（角度指定なし、連続動作）
            self.motor_arm.run_at_speed(50)
        elif action == 1:# Move up
            # アームを上げる: 速度-50で回転（角度指定なし、連続動作）
            self.motor_arm.run_at_speed(-50)

    def _set_motor_degrees(self, left_degrees: int, right_degrees: int) -> None:
        """
        左右モーターを指定された角度だけ動かす（度数単位）。
        Args:
            left_degrees: 左モーターの回転角度（正負で方向指定）
            right_degrees: 右モーターの回転角度（正負で方向指定）
        """
        self.command_counter = time.ticks_ms()
        self.motor_left.run_for_degrees(-int(left_degrees), 50)# 左はマイナス値で反転
        self.motor_right.run_for_degrees(int(right_degrees), 50)

def receiver():
    # バッファを空にして最新コマンドだけ取得
    latest_command = None
    while True:
        command = lego_spike.read_command()
        if command[0] is not None:
            latest_command = command
        else:
            break
    if latest_command is not None:
        command_id, command_parameter1, command_parameter2 = latest_command
        lego_spike.execute_command(command_id, command_parameter1, command_parameter2)

    if time.ticks_ms() - lego_spike.command_counter > MAX_IDLE_TIME:
        raise SystemExit("Maximum idle time reached, terminate lego spike.")

    time.sleep(0.01)


def main_task():
    start_time = time.time()
    while time.time() - start_time < MAX_RUN_TIME:
        receiver()


# Trigger a garbage collection cycle
gc.collect()

print("Starting LEGO Prime Hub..")

try:
    lego_spike = LegoSpike()
    main_task()
except SystemExit as e:
    print(e)

lego_spike.motor_left.brake()
lego_spike.motor_right.brake()

hub.display.show(hub.Image.ASLEEP)

print("Ended")
