# LEGO type:standard slot:2 autostart
"""Main controlling program for LEGO Spike Prime Hub"""
import hub  # type: ignore
import time
import ujson  # type: ignore

MAX_IDLE_TIME = 120000  # Maximum idle time, unit: millisecond
MAX_RUN_TIME = 600  # Maximum running time, unit: second

# Command ID list                                                                               
COMMAND_SET_MOTOR_FORWARD_POWER_ID = 201
COMMAND_SET_MOTOR_BACKWARD_POWER_ID = 202
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
        # 停止フラグを追加
        self.stop_requested = False
        # Initialization
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
          # Millisecond counter for record the latest command executed time, maximum idle time
        self.command_counter = time.ticks_ms()

        hub.display.show(hub.Image.YES)

    def read_command(self):
        command_id = None
        command_parameter1 = None
        command_parameter2 = None

        """Read command from USB and return the command ID and parameters."""
        if self.usb.any():
            data = self.usb.read(6)  # Increased to read "ET:" + 3 bytes command data

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
                self.command_counter = time.ticks_ms()

        return command_id, command_parameter1, command_parameter2

    def execute_command(self, command_id, command_parameter1, command_parameter2):
        if command_id == COMMAND_SET_MOTOR_FORWARD_POWER_ID:
            self._set_motor_speed(command_parameter1, command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_BACKWARD_POWER_ID:
            self._set_motor_speed(-command_parameter1, -command_parameter2)
        elif command_id == COMMAND_SET_MOTOR_RELATIVE_POSITION_ID:
            self._set_motor_relative_position(command_parameter1, command_parameter2)
        elif command_id == COMMAND_STOP_MOTOR_ID:
            self.stop_all()
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
        if action == 0:  # Move down
            self.motor_arm.run_at_speed(50)
        elif action == 1:  # Move up
            self.motor_arm.run_at_speed(-50)

    def get_sensor_data(self):
        """モーターA/B・カラー・超音波センサーのみ取得し、他はコメントアウト"""
        try:
            # self.motor_left, self.motor_rightは__init__で取得済みインスタンスを必ず利用
            # モーター情報取得
            # motor_left = self.motor_left
            # motor_right = self.motor_right
            # motor_left_speed = motor_left.speed() if hasattr(motor_left, 'speed') else 0
            # motor_left_position = motor_left.absolute_position() if hasattr(motor_left, 'absolute_position') else 0
            # motor_left_relative = motor_left.relative_position() if hasattr(motor_left, 'relative_position') else 0
            # motor_left_power = motor_left.power() if hasattr(motor_left, 'power') else 0
            # motor_right_speed = motor_right.speed() if hasattr(motor_right, 'speed') else 0
            # motor_right_position = motor_right.absolute_position() if hasattr(motor_right, 'absolute_position') else 0
            # motor_right_relative = motor_right.relative_position() if hasattr(motor_right, 'relative_position') else 0
            # motor_right_power = motor_right.power() if hasattr(motor_right, 'power') else 0
            
            # motor_arm, force_sensor, accel, gyro, 位置情報はコメントアウト
            # motor_arm_speed = self.motor_arm.speed() if hasattr(self.motor_arm, 'speed') else 0
            # motor_arm_position = self.motor_arm.absolute_position() if hasattr(self.motor_arm, 'absolute_position') else 0
            # motor_arm_relative = self.motor_arm.relative_position() if hasattr(self.motor_arm, 'relative_position') else 0
            # motor_arm_power = self.motor_arm.power() if hasattr(self.motor_arm, 'power') else 0
            # force_data = self.force_sensor.get() if self.force_sensor else [0]
            color_data = self.color_sensor.get() if self.color_sensor else [0, 0, 0, 0]
            ultrasonic_data = self.ultrasonic_sensor.get() if self.ultrasonic_sensor else [0]
            # try:
            #     accel = hub.motion.accelerometer()
            #     gyro = hub.motion.gyroscope()
            # except:
            #     accel = [0, 0, 0]
            #     gyro = [0, 0, 0]
            # spike_status.pyが期待するJSON形式でデータ構築
            sensor_data = {
                "m": 0,  # message_type: 0 = sensor data
                "p": [
                    # [48, [motor_left_speed, motor_left_relative, motor_left_position, motor_left_power]],  # モーターA (左) - Port 48
                    # [48, [motor_right_speed, motor_right_relative, motor_right_position, motor_right_power]],  # モーターB (右) - Port 48
                    # # モーターC (アーム) - Port 49
                    # [49, [motor_arm_speed, motor_arm_relative, motor_arm_position, motor_arm_power]],
                    # # フォースセンサー - Port 63
                    # [63, [0, 0, force_data[0] if len(force_data) > 0 else 0]],
                    # カラーセンサー - Port 61
                    [61, [
                        0, 0,  # 不明な値（プレースホルダー）
                        color_data[0] if len(color_data) > 0 else 0,  # reflected
                        color_data[1] if len(color_data) > 1 else 0,  # ambient
                        color_data[2] if len(color_data) > 2 else 0   # color
                    ]],
                    # 超音波センサー - Port 62
                    [62, [ultrasonic_data[0] if len(ultrasonic_data) > 0 else 0]],
                    # # 加速度センサー
                    # accel,
                    # # ジャイロセンサー
                    # gyro,
                    # # 位置情報（プレースホルダー）
                    # [0, 0, 0],
                    # # 予備フィールド
                    # "",
                    # 0
                ]
            }
            return sensor_data
        except Exception as e:
            # エラー時はデフォルトデータを返す（同様に必要部分のみ）
            return {
                "m": 0,
                "p": [
                    # [48, [0, 0, 0, 0]],  # モーターA
                    # [48, [0, 0, 0, 0]],  # モーターB
                    # [49, [0, 0, 0, 0]],  # モーターC
                    # [63, [0, 0, 0]],     # フォースセンサー
                    [61, [0, 0, 0, 0, 0]], # カラーセンサー
                    [62, [0]],           # 超音波センサー
                    # [0, 0, 0],           # 加速度センサー
                    # [0, 0, 0],           # ジャイロセンサー
                    # [0, 0, 0],           # 位置情報
                    # "",                  # 予備
                    # 0                    # 予備
                ]
            }

    def send_sensor_data(self):
        """センサーデータをJSON形式でUSB経由で送信"""
        sensor_data = self.get_sensor_data()
        json_string = ujson.dumps(sensor_data)
        bytes_written = self.usb.write(json_string + '\r\n')
        if bytes_written is None or bytes_written == 0:
            raise Exception("USB write failed")
    
    def safe_stop_all_motors(self):
        """すべてのモーターを安全に停止する（1回のみ、0.01秒待機）"""
        repeat = 1
        wait = 0.01
        for _ in range(repeat):
            try:
                self.motor_left.brake()
                self.motor_right.brake()
                self.motor_arm.brake()
                self.motor_left.stop()
                self.motor_right.stop()
                self.motor_arm.stop()
                if wait > 0:
                    time.sleep(wait)
            except:
                pass

    def stop_all(self):
        """全停止処理（フラグ・モーター・ディスプレイ・音）をまとめて実行"""
        self.stop_requested = True
        self.safe_stop_all_motors()
        try:
            hub.display.show(hub.Image.ASLEEP)
        except:
            pass
        try:
            hub.speaker.beep(60, 100)
        except:
            pass

def main_loop():
    """メインループ - センサーデータ送信とコマンド受信を順次処理"""
    start_time = time.time()
    # ループ外で一度だけ取得できるものはここで取得
    send_sensor_data = lego_spike.send_sensor_data
    read_command = lego_spike.read_command
    execute_command = lego_spike.execute_command

    while not lego_spike.stop_requested and (time.time() - start_time) < MAX_RUN_TIME:
        try:
            # コマンド受信処理
            command_id, command_parameter1, command_parameter2 = read_command()
            if command_id is not None:
                execute_command(command_id, command_parameter1, command_parameter2)
                # STOPコマンド受信時は即座に全処理を停止
                if command_id == COMMAND_STOP_MOTOR_ID:
                    lego_spike.stop_all()
                    break
            # センサーデータ送信処理（毎ループ送信、30ms間隔）
            try:
                send_sensor_data()
            except:
                pass
            # アイドルタイムチェック
            if time.ticks_ms() - lego_spike.command_counter > MAX_IDLE_TIME:
                lego_spike.stop_requested = True
                break
            # 動作頻度を30msに統一
            time.sleep(0.03)  # 30ms
        except:
            # エラー時は停止フラグのみ設定し、重い処理やprintはしない
            # lego_spike.stop_requested = True
            # break
            pass
        

print("Starting LEGO Prime Hub..")

try:
    lego_spike = LegoSpike()
    main_loop()
finally:
    try:
        lego_spike.stop_all(repeat=5, wait=0.1)
    except:
        pass
