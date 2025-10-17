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

sender_task_last_values = {
    "motors": {"A": [0,0,0,0], "B": [0,0,0,0], "C": [0,0,0,0]},
    "force": 0,
    "color": [0,0,0],
    "gyro": [0,0,0],
}

# 送信タスク
async def sender_task():
    while True:
        try:
            # 全センサー値取得（Noneも含めて送信）
            mr = lego_spike.motor_right.get() if hasattr(lego_spike.motor_right, 'get') else [None]*4
            ml = lego_spike.motor_left.get() if hasattr(lego_spike.motor_left, 'get') else [None]*4
            ma = lego_spike.motor_arm.get() if hasattr(lego_spike.motor_arm, 'get') else [None]*4
            fs = lego_spike.force_sensor.get() if hasattr(lego_spike.force_sensor, 'get') else [None]*2
            cs = lego_spike.color_sensor.get() if hasattr(lego_spike.color_sensor, 'get') else [None]*5
            us = lego_spike.ultrasonic_sensor.get() if hasattr(lego_spike.ultrasonic_sensor, 'get') else [None]*2
            # --- センサー値取得 ---
            try:
                accel = hub.motion.accelerometer()
            except Exception:
                accel = [0, 0, 0]
            try:
                gyro_accel = hub.motion.gyroscope()
            except Exception:
                gyro_accel = [0, 0, 0]
            try:
                yaw, pitch, roll = hub.motion.yaw_pitch_roll()
            except Exception:
                yaw, pitch, roll = 0, 0, 0
            # モータ値
            motors_data = {}
            for key, arr in zip(["A","B","C"], [mr, ml, ma]):
                motors_data[key] = {}
                for i, field in enumerate(["speed","relative_position","position","power"]):
                    v = arr[i] if arr and i < len(arr) else None
                    motors_data[key][field] = safe_get(v)
            # Force sensor値
            force_val = safe_get(fs[2] if fs and len(fs) > 2 else None)
            # Color sensor値
            color_data = []
            for idx in [2,3,4]:
                v = cs[idx] if cs and len(cs) > idx else None
                color_data.append(safe_get(v))
            # Ultrasonic sensor値
            us_val = safe_get(us[0] if us and len(us) > 0 else None)
            # 送信データ構築
            p = []
            p.append([48, [motors_data["A"]["speed"], motors_data["A"]["relative_position"], motors_data["A"]["position"], motors_data["A"]["power"]]])
            p.append([48, [motors_data["B"]["speed"], motors_data["B"]["relative_position"], motors_data["B"]["position"], motors_data["B"]["power"]]])
            p.append([49, [motors_data["C"]["speed"], motors_data["C"]["relative_position"], motors_data["C"]["position"], motors_data["C"]["power"]]])
            p.append([63, [0, 0, force_val]])
            p.append([61, [0, safe_get(cs[1]), color_data[0], color_data[1], color_data[2]]])
            p.append([62, [us_val]])
            p.append([safe_get(accel[0]), safe_get(accel[1]), safe_get(accel[2])])  # 加速度
            p.append([safe_get(gyro_accel[0]), safe_get(gyro_accel[1]), safe_get(gyro_accel[2])])  # ジャイロ（角速度）
            p.append([safe_get(yaw), safe_get(pitch), safe_get(roll)])  # ヨー・ピッチ・ロール
            p.append("")
            p.append(0)
            data = {"m": 0, "p": p}
            send_str = json.dumps(data) + "\r"
            sent_bytes = lego_spike.usb.write(send_str.encode())
            # print("USB write bytes:", sent_bytes, "| USB write content:", send_str)
            await uasyncio.sleep(0.005)  # 送信直後にバッファ安定化
        except Exception as e:
            print("[SEND ERROR]", repr(e))
        await uasyncio.sleep(0.005)  # 送信間隔厳守

async def main_task():
    recv_task = uasyncio.create_task(receiver_task())
    send_task = uasyncio.create_task(sender_task())
    await uasyncio.gather(recv_task, send_task)


# Trigger a garbage collection cycle
gc.collect()

print("Starting LEGO Prime Hub..")
lego_spike = LegoSpike()
# print("motor_right:", lego_spike.motor_right.get())
# print("motor_left:", lego_spike.motor_left.get())
# print("motor_arm:", lego_spike.motor_arm.get())
# print("force_sensor:", lego_spike.force_sensor.get())
# print("color_sensor:", lego_spike.color_sensor.get())
# print("ultrasonic_sensor:", lego_spike.ultrasonic_sensor.get())
# print("accelerometer:", hub.motion.accelerometer())
# print("gyroscope:", hub.motion.gyroscope())
# print("yaw_pitch_roll:", hub.motion.yaw_pitch_roll())

try:
    lego_spike = LegoSpike()
    uasyncio.run(main_task())
except SystemExit as e:
    print(e)
