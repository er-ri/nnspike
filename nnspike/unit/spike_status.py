import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class MotorStatus:
    """Status information for a motor connected to the Spike Prime hub."""

    position: Optional[int] = None  # モーター用positionは残す（A/B/C用）
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

    def __init__(self, distance=None, force=None, color=None, accelerometer=None, gyroscope=None, yaw_pitch_roll=None):
        self.distance = distance
        self.force = force
        self.color = color if color is not None else ColorSensorStatus(None, None, None)
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.yaw_pitch_roll = yaw_pitch_roll

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorStatus":
        color = data.get("color")
        if isinstance(color, dict):
            color = ColorSensorStatus.from_dict(color)
        elif not isinstance(color, ColorSensorStatus):
            color = ColorSensorStatus(None, None, None)
        accelerometer = VectorStatus.from_dict(data.get("accelerometer", {})) if data.get("accelerometer") else None
        gyroscope = VectorStatus.from_dict(data.get("gyroscope", {})) if data.get("gyroscope") else None
        yaw_pitch_roll = data.get("yaw_pitch_roll")
        return cls(
            distance=data.get("distance"),
            force=data.get("force"),
            color=color,
            accelerometer=accelerometer,
            gyroscope=gyroscope,
            yaw_pitch_roll=yaw_pitch_roll,
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

        if raw_data is not None:
            self.update(raw_data)

    def update(self, data: Union[str, bytes, Dict]) -> None:
        """
        Update the status with new data from the Spike Prime.
        Args:
            data: Raw data from the Spike Prime (string, bytes, or dictionary)
        """
        try:
            parsed_data = self._parse_data(data)
        except Exception:
            return

        self.timestamp = parsed_data.get("timestamp", time.time())
        self.message_type = parsed_data.get("message_type", -1)
        self.raw_data = parsed_data.get("raw", {})
        # type: -1のときrawデータを表示して原因調査
        if self.message_type == -1:
            print("[type:-1 raw]", self.raw_data)

        # message_typeによる分岐・returnを廃止。常に全データを更新。

        # 全てのmessage_typeでインターバル計算
        now = self.timestamp
        dt = None
        if hasattr(self, '_last_motor_update_time') and self._last_motor_update_time is not None:
            dt = now - self._last_motor_update_time
            dt_ms = dt * 1000 if dt is not None else None
            print(f"interval: {dt_ms:06.2f} ms | type: {self.message_type} | raw: {self.raw_data}")
        self._last_motor_update_time = now

        # Update motors
        for motor_id, motor_data in parsed_data.get("motors", {}).items():
            if motor_id in self.motors:
                self.motors[motor_id] = MotorStatus.from_dict(motor_data)

        # Update sensors
        self.sensors = SensorStatus.from_dict(parsed_data.get("sensors", {}))

        # Update battery
        self.battery = BatteryStatus.from_dict(parsed_data.get("battery", {}))

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
            if message_type == 0:  # Sensor data message
                # 順番通りに処理
                # 0: Motor A (port 48)
                if len(payload) > 0 and isinstance(payload[0], list) and payload[0][0] == 48:
                    result["motors"]["A"] = {
                        "speed": (payload[0][1][0] if len(payload[0][1]) > 0 else None),
                        "relative_position": (payload[0][1][1] if len(payload[0][1]) > 2 else None),
                        "position": (payload[0][1][2] if len(payload[0][1]) > 2 else None),
                        "power": (payload[0][1][3] if len(payload[0][1]) > 3 else None),
                    }
                # 1: Motor B (port 48)
                if len(payload) > 1 and isinstance(payload[1], list) and payload[1][0] == 48:
                    result["motors"]["B"] = {
                        "speed": (payload[1][1][0] if len(payload[1][1]) > 0 else None),
                        "relative_position": (payload[1][1][1] if len(payload[1][1]) > 2 else None),
                        "position": (payload[1][1][2] if len(payload[1][1]) > 2 else None),
                        "power": (payload[1][1][3] if len(payload[1][1]) > 3 else None),
                    }
                # 2: Motor C (port 49)
                if len(payload) > 2 and isinstance(payload[2], list) and payload[2][0] == 49:
                    result["motors"]["C"] = {
                        "speed": (payload[2][1][0] if len(payload[2][1]) > 0 else None),
                        "relative_position": (payload[2][1][1] if len(payload[2][1]) > 2 else None),
                        "position": (payload[2][1][2] if len(payload[2][1]) > 2 else None),
                        "power": (payload[2][1][3] if len(payload[2][1]) > 3 else None),
                    }
                # 3: Force sensor (port 63)
                if len(payload) > 3 and isinstance(payload[3], list) and payload[3][0] == 63:
                    result["sensors"]["force"] = payload[3][1][2] if len(payload[3][1]) > 2 else None
                # 4: Color sensor (port 61)
                if len(payload) > 4 and isinstance(payload[4], list) and payload[4][0] == 61:
                    result["sensors"]["color"] = {
                        "reflected": (payload[4][1][2] if len(payload[4][1]) > 2 else None),
                        "ambient": (payload[4][1][3] if len(payload[4][1]) > 3 else None),
                        "color": (payload[4][1][4] if len(payload[4][1]) > 4 else None),
                    }
                # 5: Distance sensor (port 62)
                if len(payload) > 5 and isinstance(payload[5], list) and payload[5][0] == 62:
                    result["sensors"]["distance"] = payload[5][1][0] if len(payload[5][1]) > 0 else None
                # 6: 加速度（IDではなく値）
                if len(payload) > 6 and isinstance(payload[6], list):
                    # [x, y, z] の場合
                    if len(payload[6]) == 3:
                        result["sensors"]["accelerometer"] = {
                            "x": payload[6][0],
                            "y": payload[6][1],
                            "z": payload[6][2],
                        }
                # 7: gyroscope (x/y/z)
                if len(payload) > 7 and isinstance(payload[7], list):
                    if len(payload[7]) == 3:
                        result["sensors"]["gyroscope"] = {
                            "x": payload[7][0],
                            "y": payload[7][1],
                            "z": payload[7][2],
                        }
                # 8: yaw_pitch_roll (yaw, pitch, roll)
                if len(payload) > 8 and isinstance(payload[8], list):
                    if len(payload[8]) == 3:
                        result["sensors"]["yaw_pitch_roll"] = {
                            "yaw": payload[8][0],
                            "pitch": payload[8][1],
                            "roll": payload[8][2],
                        }

                # Position from sensors
                # 位置情報は送信されないため、ここは削除

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
            if motor and motor.relative_position is not None:
                lines.append(f"  Motor {motor_id}: Relative Position: {motor.relative_position}, Power: {motor.power}")

        lines.append("Sensors:")
        if self.sensors.distance is not None:
            lines.append(f"  Distance: {self.sensors.distance}mm")
        if self.sensors.force is not None:
            lines.append(f"  Force: {self.sensors.force}")
        if self.sensors.color:
            lines.append(f"  Color - Reflected: {self.sensors.color.reflected}, Ambient: {self.sensors.color.ambient}, Color: {self.sensors.color.color}")
        if self.sensors.accelerometer:
            lines.append(f"  Accelerometer - x: {self.sensors.accelerometer.x}, y: {self.sensors.accelerometer.y}, z: {self.sensors.accelerometer.z}")
        if self.sensors.gyroscope:
            lines.append(f"  Gyroscope - x: {self.sensors.gyroscope.x}, y: {self.sensors.gyroscope.y}, z: {self.sensors.gyroscope.z}")
        if self.sensors.yaw_pitch_roll:
            lines.append(f"  YawPitchRoll - yaw: {self.sensors.yaw_pitch_roll.get('yaw')}, pitch: {self.sensors.yaw_pitch_roll.get('pitch')}, roll: {self.sensors.yaw_pitch_roll.get('roll')}")
        if self.battery and self.battery.percent is not None:
            lines.append(f"Battery: {self.battery.percent}% ({self.battery.voltage}V)")

        return "\n".join(lines)
