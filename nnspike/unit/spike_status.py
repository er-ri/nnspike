import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class MotorStatus:
    """Status information for a motor connected to the Spike Prime hub."""

    position: Optional[int] = None
    power: Optional[int] = None
    relative_position: Optional[int] = None
    speed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotorStatus":
        return cls(
            position=data.get("position"),
            power=data.get("power"),
            relative_position=data.get("relative_position"),
            speed=data.get("speed"),
        )



@dataclass
class ColorSensorStatus:
    """Status information for a color sensor connected to the Spike Prime hub."""

    reflected: Optional[int] = None
    ambient: Optional[int] = None
    color: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorSensorStatus":
        return cls(
            reflected=data.get("reflected"),
            ambient=data.get("ambient"),
            color=data.get("color"),
        )


@dataclass
class VectorStatus:
    """Status information for vector-based sensors (gyro, accelerometer)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorStatus":
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0), z=data.get("z", 0.0))


@dataclass
class Position:
    """Position information from the Spike Prime hub."""

    x: float = 0.0
    y: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0))



@dataclass
class BatteryStatus:
    """Battery status information from the Spike Prime hub."""

    voltage: Optional[float] = None
    percent: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatteryStatus":
        return cls(voltage=data.get("voltage"), percent=data.get("percent"))


@dataclass
class SensorStatus:
    """Status information for all sensors connected to the Spike Prime hub."""

    distance: Optional[int] = None
    force: Optional[int] = None
    color: ColorSensorStatus = field(default_factory=lambda: ColorSensorStatus(None, None, None))
    gyro: Optional[VectorStatus] = None
    accelerometer: Optional[VectorStatus] = None
    position: Optional[Position] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorStatus":
        color = data.get("color")
        if isinstance(color, dict):
            color = ColorSensorStatus.from_dict(color)
        elif not isinstance(color, ColorSensorStatus):
            color = ColorSensorStatus(None, None, None)
        return cls(
            distance=data.get("distance"),
            force=data.get("force"),
            color=color,
            gyro=(VectorStatus.from_dict(data.get("gyro", {})) if data.get("gyro") else None),
            accelerometer=(VectorStatus.from_dict(data.get("accelerometer", {})) if data.get("accelerometer") else None),
            position=(Position.from_dict(data.get("position", {})) if data.get("position") else None),
        )



class SpikeStatus:
    """
    Class to represent and access the status of a Lego Spike Prime hub.

    This class provides a structured way to access the data received from the
    Spike Prime, including sensor readings, motor positions, and battery status.
    """


    def __init__(self, raw_data: Optional[Union[str, bytes, Dict]] = None):
        """
        Initialize the SpikeStatus object.

        Args:
            raw_data: Optional raw data from the Spike Prime to parse
        """
        self.timestamp: float = time.time()
        self.message_type: int = -1
        self.motors: Dict[str, MotorStatus] = {
            "A": MotorStatus(),
            "B": MotorStatus(),
            "C": MotorStatus(),  # Add motor arm (port C)
        }
        self.sensors: SensorStatus = SensorStatus()
        self.battery: BatteryStatus = BatteryStatus()
        self.raw_data: Dict = {}

        # --- gyro積分角度用 ---
        self._gyro_angle_x: float = 0.0
        self._gyro_angle_y: float = 0.0
        self._gyro_angle_z: float = 0.0
        self._last_gyro_update_time: Optional[float] = None
        self._last_gyro_x: Optional[float] = None
        self._last_gyro_y: Optional[float] = None
        self._last_gyro_z: Optional[float] = None

        if raw_data is not None:
            self.update(raw_data)

    def reset_gyro_angle(self) -> None:
        """
        積分したジャイロ角度のみをリセットする。
        """
        self._gyro_angle_x = 0.0
        self._gyro_angle_y = 0.0
        self._gyro_angle_z = 0.0
        self._last_gyro_update_time = None
        self._last_gyro_x = None
        self._last_gyro_y = None
        self._last_gyro_z = None

    def update(self, data: Union[str, bytes, Dict]) -> None:
        # update呼び出し間隔（ms）を計算（m=0時のみprint）
        now = time.time()
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = now
        dt = (now - self._last_update_time) * 1000
        self._last_update_time = now
        """
        Update the status with new data from the Spike Prime.

        Args:
            data: Raw data from the Spike Prime (string, bytes, or dictionary)
        """
        import traceback
        try:
            parsed_data = self._parse_data(data)
        except Exception as e:
            print(f"Exception in _parse_data: {e}")
            print(f"RAW (error): {data}")
            traceback.print_exc()
            return

        # Gyro値は[Yaw, Pitch, Roll]で統一して表示
        if 'sensors' in parsed_data and 'gyro' in parsed_data['sensors']:
            gyro = parsed_data['sensors']['gyro']
            gyro_list = [gyro.get('x', 0), gyro.get('y', 0), gyro.get('z', 0)]
            print(f"[SpikeStatus] parsed_data: ... gyro: [Yaw={gyro_list[0]}, Pitch={gyro_list[1]}, Roll={gyro_list[2]}] ...")
        else:
            print(f"[SpikeStatus] parsed_data: {parsed_data}")
        # Update basic metadata
        self.timestamp = parsed_data.get("timestamp", time.time())
        self.message_type = parsed_data.get("message_type", -1)
        self.raw_data = parsed_data.get("raw", {})

        # m=0（センサーデータ）以外は積分・センサー・モーター処理を完全にスキップ
        if self.message_type != 0:
            # バッテリー情報のみ更新
            battery_data = parsed_data.get("battery", {})
            self.battery = BatteryStatus.from_dict(battery_data)
            print(f"m={self.message_type} skip")
            return
        # m=0のとき、受信間隔と各センサー値を個別にprint
        print(f"[SpikeStatus][m=0] interval: {dt:.1f}ms")
        motors_data = parsed_data.get("motors", {})
        sensors_data = parsed_data.get("sensors", {})
        force = sensors_data.get("force", None)
        color = sensors_data.get("color", {})
        color_list = [color.get("reflected", 0), color.get("ambient", 0), color.get("color", 0)] if color else None
        gyro = sensors_data.get("gyro", {})
        # slot_prod.pyの送信順（x=Yaw, y=Pitch, z=Roll）に合わせる
        # 表示も[Yaw, Pitch, Roll]の順で統一
        if gyro:
            gyro_list = [gyro.get("x", 0), gyro.get("y", 0), gyro.get("z", 0)]
            print(f"[SpikeStatus][m=0] motors: {motors_data}")
            print(f"[SpikeStatus][m=0] force: {force}")
            print(f"[SpikeStatus][m=0] color: {color_list}")
            print(f"[SpikeStatus][m=0] gyro: [Yaw={gyro_list[0]}, Pitch={gyro_list[1]}, Roll={gyro_list[2]}]  # [Yaw, Pitch, Roll]")
        else:
            print(f"[SpikeStatus][m=0] motors: {motors_data}")
            print(f"[SpikeStatus][m=0] force: {force}")
            print(f"[SpikeStatus][m=0] color: {color_list}")
            print(f"[SpikeStatus][m=0] gyro: None  # [Yaw, Pitch, Roll]")

        # Update motors
        motors_data = parsed_data.get("motors", {})
        for motor_id, motor_data in motors_data.items():
            if motor_id in self.motors:
                self.motors[motor_id] = MotorStatus.from_dict(motor_data)

        # Update sensors
        sensors_data = parsed_data.get("sensors", {})
        self.sensors = SensorStatus.from_dict(sensors_data)

        # --- gyro積分角度の更新 ---
        now = self.timestamp
        gyro = self.sensors.gyro
        if gyro is not None:
            # 前回値があればdtを計算
            if self._last_gyro_update_time is not None:
                dt = now - self._last_gyro_update_time
                # 積分（台形則: 前回と今回の平均 × dt）
                if self._last_gyro_x is not None:
                    self._gyro_angle_x += ((self._last_gyro_x + gyro.x) / 2.0) * dt
                if self._last_gyro_y is not None:
                    self._gyro_angle_y += ((self._last_gyro_y + gyro.y) / 2.0) * dt
                if self._last_gyro_z is not None:
                    self._gyro_angle_z += ((self._last_gyro_z + gyro.z) / 2.0) * dt
            # 値を保存
            self._last_gyro_update_time = now
            self._last_gyro_x = gyro.x
            self._last_gyro_y = gyro.y
            self._last_gyro_z = gyro.z
        else:
            # gyroがNoneなら前回値はクリアしない（センサ未接続時など）
            pass

        # Update battery
        battery_data = parsed_data.get("battery", {})
        self.battery = BatteryStatus.from_dict(battery_data)

    def get_gyro_angle_x(self) -> float:
        """
        積分したgyro_x角度（度）を返す。
        Returns:
            float: x軸（ロール）角度（度）
        """
        return self._gyro_angle_x

    def get_gyro_angle_y(self) -> float:
        """
        積分したgyro_y角度（度）を返す。
        Returns:
            float: y軸（ピッチ or ロール）角度（度）
        """
        return self._gyro_angle_y

    def get_gyro_angle_z(self) -> float:
        """
        積分したgyro_z角度（度）を返す。
        Returns:
            float: z軸（ヨー）角度（度）
        """
        return self._gyro_angle_z

    @staticmethod
    def _parse_data(data: Union[str, bytes, Dict]) -> Dict:
        """
        Parse raw data from Spike Prime into a dictionary.

        Args:
            data: Raw data from the Spike Prime (string, bytes, or dictionary)

        Returns:
            Dict: Parsed data as a dictionary
        """
        # Port definitions
        # 48: motor pair (A + B)
        # 49: motor arm (C)
        # 61: color sensor (E)
        # 63: force sensor (D)
        # 62: distance sensor (F)

        # Store original data for error reporting
        original_data = data

        # If already a dictionary, return as is
        if isinstance(data, dict):
            return data

        # Convert from bytes to string if needed
        if isinstance(data, bytes):
            data_str = data.decode("utf-8").strip()
        else:
            data_str = str(data).strip()

        # Remove trailing carriage return if present
        if data_str.endswith("\r"):
            data_str = data_str[:-1]

        try:
            # Parse JSON
            json_data = json.loads(data_str)

            # Extract key information
            message_type = json_data.get("m", -1)
            payload = json_data.get("p", [])

            # Initialize result structure
            result = {
                "message_type": message_type,
                "timestamp": time.time(),
                "raw": json_data,
                "sensors": {},
                "motors": {},
                "battery": {},
            }

            # Process the payload based on message type
            if message_type == 0:  # Sensor data message                # Motor A and B position - Port 48
                motor_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 48]
                if len(motor_entries) >= 2:
                    result["motors"]["A"] = {
                        "speed": (motor_entries[1][1][0] if len(motor_entries[1][1]) > 0 else None),
                        "relative_position": (motor_entries[1][1][1] if len(motor_entries[1][1]) > 2 else None),
                        "position": (motor_entries[1][1][2] if len(motor_entries[1][1]) > 2 else None),
                        "power": (motor_entries[1][1][3] if len(motor_entries[1][1]) > 3 else None),
                    }
                    result["motors"]["B"] = {
                        "speed": (motor_entries[0][1][0] if len(motor_entries[0][1]) > 0 else None),
                        "relative_position": (motor_entries[0][1][1] if len(motor_entries[0][1]) > 2 else None),
                        "position": (motor_entries[0][1][2] if len(motor_entries[0][1]) > 2 else None),
                        "power": (motor_entries[0][1][3] if len(motor_entries[0][1]) > 3 else None),
                    }

                # Motor arm (C) - Port 49
                motor_arm_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 49]
                if motor_arm_entries:
                    result["motors"]["C"] = {
                        "speed": (motor_arm_entries[0][1][0] if len(motor_arm_entries[0][1]) > 0 else None),
                        "relative_position": (motor_arm_entries[0][1][1] if len(motor_arm_entries[0][1]) > 2 else None),
                        "position": (motor_arm_entries[0][1][2] if len(motor_arm_entries[0][1]) > 2 else None),
                        "power": (motor_arm_entries[0][1][3] if len(motor_arm_entries[0][1]) > 3 else None),
                    }

                # Force sensor - Port 63 (ESSENTIAL - USED)
                force_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 63]
                if force_entries:
                    result["sensors"]["force"] = force_entries[0][1][1] if len(force_entries[0][1]) > 2 else None

                # Distance sensor - Port 62
                distance_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 62]
                if distance_entries:
                    result["sensors"]["distance"] = distance_entries[0][1][0] if len(distance_entries[0][1]) > 0 else None

                # Color sensor - Port 61
                color_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 61]
                if color_entries and len(color_entries[0][1]) > 4:
                    result["sensors"]["color"] = {
                        "reflected": (color_entries[0][1][2] if len(color_entries[0][1]) > 2 else None),
                        "ambient": (color_entries[0][1][3] if len(color_entries[0][1]) > 3 else None),
                        "color": (color_entries[0][1][4] if len(color_entries[0][1]) > 4 else None),
                    }

                # Gyro sensor information
                if len(payload) > 7 and isinstance(payload[7], list) and len(payload[7]) >= 3:
                    result["sensors"]["gyro"] = {
                        "x": payload[7][0],
                        "y": payload[7][1],
                        "z": payload[7][2],
                    }

                # Accelerometer information
                if len(payload) > 8 and isinstance(payload[8], list) and len(payload[8]) >= 3:
                    result["sensors"]["accelerometer"] = {
                        "x": payload[8][0],
                        "y": payload[8][1],
                        "z": payload[8][2],
                    }

                # Position from sensors
                if len(payload) > 6 and isinstance(payload[6], list) and len(payload[6]) >= 3:
                    result["sensors"]["position"] = {
                        "x": payload[6][1],
                        "y": payload[6][2],
                    }

            elif message_type == 2:  # Battery status message
                if len(payload) > 1:
                    result["battery"] = {
                        "voltage": payload[0] if len(payload) > 0 else None,
                        "percent": payload[1] if len(payload) > 1 else None,
                    }

            return result

        except json.JSONDecodeError as e:
            display_data = original_data.decode("utf-8") if isinstance(original_data, bytes) else original_data
            print(f"JSON parsing error: {e}, raw data: {display_data}")
            return {"error": "json_parse_error", "raw": data}
        except Exception as e:
            display_data = original_data.decode("utf-8") if isinstance(original_data, bytes) else original_data
            print(f"General parsing error: {e}, raw data: {display_data}")
            return {"error": "parsing_error", "raw": data}

    def __str__(self) -> str:
        """Return a string representation of the status."""
        lines = [
            f"Spike Status - Message Type: {self.message_type}, Time: {self.timestamp}",
            "Motors:",
        ]

        for motor_id, motor in self.motors.items():
            if motor and motor.position is not None:
                lines.append(f"  Motor {motor_id}: Position: {motor.position}, Power: {motor.power}")

        lines.append("Sensors:")
        if self.sensors.distance is not None:
            lines.append(f"  Distance: {self.sensors.distance}mm")
        if self.sensors.force is not None:
            lines.append(f"  Force: {self.sensors.force}")
        if self.sensors.color:
            lines.append(f"  Color - Reflected: {self.sensors.color.reflected}, Ambient: {self.sensors.color.ambient}, Color: {self.sensors.color.color}")
        if self.sensors.gyro:
            # 表示順を[Yaw, Pitch, Roll]で統一
            lines.append(f"  Gyro - Yaw: {self.sensors.gyro.x}, Pitch: {self.sensors.gyro.y}, Roll: {self.sensors.gyro.z}")
        if self.sensors.accelerometer:
            lines.append(f"  Accel - X: {self.sensors.accelerometer.x}, Y: {self.sensors.accelerometer.y}, Z: {self.sensors.accelerometer.z}")
        if self.sensors.position:
            lines.append(f"  Position - X: {self.sensors.position.x}, Y: {self.sensors.position.y}")
        if self.battery and self.battery.percent is not None:
            lines.append(f"Battery: {self.battery.percent}% ({self.battery.voltage}V)")

        return "\n".join(lines)
