
"""Main controlling program for LEGO Spike Prime Hub"""
import gc
import time
import json
import hub  # type: ignore
import uasyncio  # type: ignore

# Command ID list
COMMAND_SET_MOTOR_FORWARD_SPEED_ID = 201
COMMAND_SET_MOTOR_BACKWARD_SPEED_ID = 202
COMMAND_SET_MOTOR_RELATIVE_POSITION_ID = 203
COMMAND_STOP_MOTOR_ID = 204
COMMAND_MOVE_ARM_ID = 205
COMMAND_SET_MOTOR_MIXED_SPEED_ID = 206

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
    """Class to control LEGO Spike Prime Hub.
    This class initializes the hub, sets up motors and sensors, and provides methods to read commands
    from USB and execute them.
    It also includes methods to control the motors and arm, and to handle the command execution logic.
    """

    def __init__(self) -> None:
        # Initialization
        hub.display.show(hub.Image.ALL_CLOCKS, delay=400, clear=True, wait=False, loop=True, fade=0)
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

    def read_command(self):
        command_id = None
        command_parameter1 = None
        command_parameter2 = None

        """Read command from USB and return the command ID and parameters."""
        if self.usb.any():
            data = self.usb.read(6)  # Read "CF:" + 3 bytes command data

            flag_pos = data.find(CMD_FLAG)

            if flag_pos >= 0 and len(data) >= flag_pos + 6:  # Ensure we have enough bytes
                raw_bytes = data[flag_pos + 3 : flag_pos + 6]  # Extract the 3 bytes after "CF:"
                command_id = int.from_bytes(raw_bytes[0:1], "big")
                command_parameter1 = int.from_bytes(raw_bytes[1:2], "big")
                command_parameter2 = int.from_bytes(raw_bytes[2:3], "big")

                return command_id, command_parameter1, command_parameter2

        # If no command is read, return None values
        return None, None, None

    def execute_command(self, command_id, command_parameter1, command_parameter2):
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
        elif command_id == COMMAND_SET_MOTOR_MIXED_SPEED_ID:
            # 正負値をそのまま左右に適用
            self._set_motor_mixed_speed(command_parameter1, command_parameter2)

    def _set_motor_mixed_speed(self, left_speed: int, right_speed: int) -> None:
        """
        左右のモーターに正負値を適用する（右前進・左バック等の個別制御用）
        Args:
            left_speed: 左モーター速度（0～200で受信、-100～+100に復元）
            right_speed: 右モーター速度（0～200で受信、-100～+100に復元）
        """
        left = int(left_speed) - 100
        right = int(right_speed) - 100
        self.motor_left.run_at_speed(-left)
        self.motor_right.run_at_speed(right)

    def _set_motor_speed(self, left_speed: int, right_speed: int) -> None:
        """Method to control the steering wheel angle.

        Args:
            left_speed: Left wheel speed(0~100)
            right_speed: Right wheel speed(0~100)
        """
        self.motor_left.run_at_speed(-int(left_speed))
        self.motor_right.run_at_speed(int(right_speed))

    def _set_motor_relative_position(self, left_position: int, right_position: int) -> None:
        self.motor_left.preset(-int(left_position))
        self.motor_right.preset(int(right_position))

    def _move_arm(self, action: int) -> None:
        """Method to move the arm motor up or down and set its current position using preset.

        Args:
            action: Action to perform (0 = move down, 1 = move up)
                0: アームを下げる (move down)
                1: アームを上げる (move up)
                2: アームを止める (stop arm)
        """
        if action == 0:  # Move down
            self.motor_arm.run_at_speed(int(40))  # Encapulate int() to ensure speed is an integer
        elif action == 1:  # Move up
            self.motor_arm.run_at_speed(-int(40))  # Encapulate int() to ensure speed is an integer
        elif action == 2:  # Stop arm
            self.motor_arm.brake()



# None安全化関数
def safe_get(val, default=0):
    return val if val is not None else default

# カラーセンサー用
def safe_color_get(val, idx, default=0):
    try:
        v = val[idx]
        return v if v is not None else default
    except Exception:
        return default

# 受信タスク
async def receiver_task():
    while True:
        try:
            command_id, command_parameter1, command_parameter2 = lego_spike.read_command()
        except Exception:
            command_id = None
            command_parameter1 = None
            command_parameter2 = None
        if command_id is not None:
            lego_spike.execute_command(command_id, command_parameter1, command_parameter2)
        await uasyncio.sleep(0.01)

# 送信タスク
async def sender_task():
    while True:
        try:
            mr = lego_spike.motor_right.get()
            ml = lego_spike.motor_left.get()
            ma = lego_spike.motor_arm.get()
            fs = lego_spike.force_sensor.get()
            us = lego_spike.ultrasonic_sensor.get()
            cs = lego_spike.color_sensor.get()
            gyro = hub.motion.gyro()
            accel = hub.motion.accelerometer()
            pos = hub.motion.position()
            # last値保持用のstatic変数
            if not hasattr(sender_task, "last_values"):
                sender_task.last_values = {
                    "motors": {"A": [0,0,0,0], "B": [0,0,0,0], "C": [0,0,0,0]},
                    "force": 0,
                    "color": [0,0,0],
                    "gyro": [0,0,0],
                }
            lv = sender_task.last_values
            # 値取得（Noneならlast値、last値もNoneなら0）
            def get_with_last(val, last):
                return val if val is not None else (last if last is not None else 0)
            motors_data = {}
            for key, arr in zip(["A","B","C"], [mr, ml, ma]):
                motors_data[key] = {}
                for i, field in enumerate(["speed","relative_position","position","power"]):
                    v = arr[i] if i < len(arr) else 0  # 要素不足時は必ず0
                    motors_data[key][field] = get_with_last(v, lv["motors"][key][i])
                    lv["motors"][key][i] = motors_data[key][field]
            force_val = get_with_last(fs[1], lv["force"])
            lv["force"] = force_val
            # distanceは不要
            color_data = []
            for i, idx in enumerate([2,3,4]):
                v = safe_color_get(cs, idx)
                color_data.append(get_with_last(v, lv["color"][i]))
                lv["color"][i] = color_data[i]
            gyro_data = []
            for i in range(3):
                v = gyro[i] if i < len(gyro) else None
                gyro_data.append(get_with_last(v, lv["gyro"][i]))
                lv["gyro"][i] = gyro_data[i]
            # accel/positionは不要
            data = {
                "message_type": 0,
                "motors": motors_data,
                "sensors": {
                    "force": force_val,
                    "color": {
                        "reflected": color_data[0],
                        "ambient": color_data[1],
                        "color": color_data[2],
                    },
                    "gyro": {
                        "x": gyro_data[0],
                        "y": gyro_data[1],
                        "z": gyro_data[2],
                    },
                },
            }
            send_str = json.dumps(data) + "\r"
            lego_spike.usb.write(send_str.encode())
            await uasyncio.sleep(0)  # 送信直後にyieldでバッファ安定化
        except Exception:
            pass
    await uasyncio.sleep(0.05)

async def main_task():
    recv_task = uasyncio.create_task(receiver_task())
    send_task = uasyncio.create_task(sender_task())
    await uasyncio.gather(recv_task, send_task)


# Trigger a garbage collection cycle
gc.collect()

print("Starting LEGO Prime Hub..")

try:
    lego_spike = LegoSpike()
    uasyncio.run(main_task())
except SystemExit as e:
    print(e)
