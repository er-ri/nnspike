import json
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any
import time


@dataclass
class MotorStatus:
    """Status information for a motor connected to the Spike Prime hub."""
    position: Optional[int] = None
    power: Optional[int] = None
    relative_position: Optional[int] = None
    speed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MotorStatus':
        return cls(
            position=data.get('position'),
            power=data.get('power'),
            relative_position=data.get('relative_position'),
            speed=data.get('speed')
        )


@dataclass
class ColorSensorStatus:
    """Status information for a color sensor connected to the Spike Prime hub."""
    reflected: Optional[int] = None
    ambient: Optional[int] = None
    color: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorSensorStatus':
        return cls(
            reflected=data.get('reflected'),
            ambient=data.get('ambient'),
            color=data.get('color')
        )


@dataclass
class VectorStatus:
    """Status information for vector-based sensors (gyro, accelerometer)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorStatus':
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0)
        )


@dataclass
class Position:
    """Position information from the Spike Prime hub."""
    x: float = 0.0
    y: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0)
        )


@dataclass
class BatteryStatus:
    """Battery status information from the Spike Prime hub."""
    voltage: Optional[float] = None
    percent: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatteryStatus':
        return cls(
            voltage=data.get('voltage'),
            percent=data.get('percent')
        )


@dataclass
class SensorStatus:
    """Status information for all sensors connected to the Spike Prime hub."""
    distance: Optional[int] = None
    force: Optional[int] = None
    color: Optional[ColorSensorStatus] = None
    gyro: Optional[VectorStatus] = None
    accelerometer: Optional[VectorStatus] = None
    position: Optional[Position] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorStatus':
        return cls(
            distance=data.get('distance'),
            force=data.get('force'),
            color=ColorSensorStatus.from_dict(data.get('color', {})) if data.get('color') else None,
            gyro=VectorStatus.from_dict(data.get('gyro', {})) if data.get('gyro') else None,
            accelerometer=VectorStatus.from_dict(data.get('accelerometer', {})) if data.get('accelerometer') else None,
            position=Position.from_dict(data.get('position', {})) if data.get('position') else None
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
            'A': MotorStatus(),
            'B': MotorStatus(),
            'C': MotorStatus()  # Add motor arm (port C)
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
        parsed_data = self._parse_data(data)
        
        # Update basic metadata
        self.timestamp = parsed_data.get('timestamp', time.time())
        self.message_type = parsed_data.get('message_type', -1)
        self.raw_data = parsed_data.get('raw', {})
        
        # Update motors
        motors_data = parsed_data.get('motors', {})
        for motor_id, motor_data in motors_data.items():
            if motor_id in self.motors:
                self.motors[motor_id] = MotorStatus.from_dict(motor_data)
        
        # Update sensors
        sensors_data = parsed_data.get('sensors', {})
        self.sensors = SensorStatus.from_dict(sensors_data)
        
        # Update battery
        battery_data = parsed_data.get('battery', {})
        self.battery = BatteryStatus.from_dict(battery_data)
    
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
        
        # If already a dictionary, return as is
        if isinstance(data, dict):
            return data
              # Convert from bytes to string if needed
        if isinstance(data, bytes):
            data_str = data.decode('utf-8').strip()
        else:
            data_str = str(data).strip()
            
        # Remove trailing carriage return if present
        if data_str.endswith('\r'):
            data_str = data_str[:-1]
        
        # Handle multiple JSON messages concatenated together
        # Split by '}{ pattern and process each valid JSON separately
        json_messages = []
        current_pos = 0
        
        while current_pos < len(data_str):
            # Find the start of a JSON object
            start_pos = data_str.find('{', current_pos)
            if start_pos == -1:
                break
                
            # Find the matching closing brace
            brace_count = 0
            end_pos = start_pos
            
            for i in range(start_pos, len(data_str)):
                if data_str[i] == '{':
                    brace_count += 1
                elif data_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
              # Extract this JSON message
            if brace_count == 0:  # Found complete JSON
                json_message = data_str[start_pos:end_pos + 1]
                json_messages.append(json_message)
                current_pos = end_pos + 1
            else:
                # Incomplete JSON, skip it
                break
        
        # Process the last complete JSON message (most recent data)
        if not json_messages:
            return {"error": "no_valid_json", "raw": data, "processed": data_str}
        
        # Use the last (most recent) complete JSON message
        data_str = json_messages[-1]
        
        try:
            # Parse JSON
            json_data = json.loads(data_str)

            # Extract key information
            message_type = json_data.get('m', -1)
            payload = json_data.get('p', [])
            
            # Initialize result structure
            result = {
                "message_type": message_type,
                "timestamp": time.time(),
                "raw": json_data,
                "sensors": {},
                "motors": {},
                "battery": {}
            }
            
            # Process the payload based on message type
            if message_type == 0:  # Sensor data message
                # Motor A and B position - Port 48
                motor_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 48]
                if len(motor_entries) >= 2:
                    result["motors"]["A"] = {
                        "speed": motor_entries[0][1][0] if len(motor_entries[0][1]) > 0 else None,
                        "relative_position": motor_entries[0][1][1] if len(motor_entries[0][1]) > 1 else None,
                        "position": motor_entries[0][1][2] if len(motor_entries[0][1]) > 2 else None,
                        "power": motor_entries[0][1][3] if len(motor_entries[0][1]) > 3 else None
                    }
                    result["motors"]["B"] = {
                        "speed": motor_entries[1][1][0] if len(motor_entries[1][1]) > 0 else None,
                        "relative_position": motor_entries[1][1][1] if len(motor_entries[1][1]) > 1 else None,
                        "position": motor_entries[1][1][2] if len(motor_entries[1][1]) > 2 else None,
                        "power": motor_entries[1][1][3] if len(motor_entries[1][1]) > 3 else None
                    }
                
                # Motor arm (C) - Port 49
                motor_arm_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 49]
                if motor_arm_entries:
                    result["motors"]["C"] = {
                        "speed": motor_arm_entries[0][1][0] if len(motor_arm_entries[0][1]) > 0 else None,
                        "relative_position": motor_arm_entries[0][1][1] if len(motor_arm_entries[0][1]) > 1 else None,
                        "position": motor_arm_entries[0][1][2] if len(motor_arm_entries[0][1]) > 2 else None,
                        "power": motor_arm_entries[0][1][3] if len(motor_arm_entries[0][1]) > 3 else None
                    }
                
                # Force sensor - Port 63
                force_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 63]
                if force_entries:
                    result["sensors"]["force"] = force_entries[0][1][2] if len(force_entries[0][1]) > 2 else None                # Distance sensor - Port 62
                distance_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 62]
                if distance_entries:
                    # 超音波センサーの生データを詳細にログ出力
                    distance_raw = distance_entries[0][1][0] if len(distance_entries[0][1]) > 0 else None
                    
                    # グローバル変数でデバッグカウンタを管理
                    global _distance_debug_counter, _last_distance_value
                    if '_distance_debug_counter' not in globals():
                        _distance_debug_counter = 0
                        _last_distance_value = None
                    
                    _distance_debug_counter += 1
                    
                    # 値が変化したときのログ
                    if distance_raw != _last_distance_value:
                        print(f"SpikeStatus: 超音波センサー生データ変化 - {_last_distance_value} → {distance_raw}")
                        print(f"SpikeStatus: 超音波センサー生配列 - {distance_entries[0][1]}")
                        _last_distance_value = distance_raw
                    
                    # 定期的な状態確認（100回に1回）
                    if _distance_debug_counter % 100 == 0:
                        print(f"SpikeStatus: 超音波センサー定期確認 - 生値={distance_raw}, 配列={distance_entries[0][1]}")
                        print(f"SpikeStatus: Port 62エントリ数={len(distance_entries)}, 第1エントリ={distance_entries[0]}")
                    
                    result["sensors"]["distance"] = distance_raw
                else:
                    # Port 62のデータが見つからない場合のログ
                    global _distance_missing_counter
                    if '_distance_missing_counter' not in globals():
                        _distance_missing_counter = 0
                    _distance_missing_counter += 1
                    
                    if _distance_missing_counter % 50 == 0:  # 50回に1回
                        port_info = [f"Port {p[0]}" for p in payload if p and isinstance(p, list) and len(p) > 0]
                        print(f"SpikeStatus: Port 62（超音波センサー）が見つかりません - 利用可能ポート: {port_info}")
                        print(f"SpikeStatus: ペイロード詳細 - {payload[:3]}...")  # 最初の3要素のみ表示
                
                # Color sensor - Port 61
                color_entries = [p for p in payload if p and isinstance(p, list) and p[0] == 61]
                if color_entries and len(color_entries[0][1]) > 4:
                    result["sensors"]["color"] = {
                        "reflected": color_entries[0][1][2] if len(color_entries[0][1]) > 2 else None,
                        "ambient": color_entries[0][1][3] if len(color_entries[0][1]) > 3 else None,
                        "color": color_entries[0][1][4] if len(color_entries[0][1]) > 4 else None,
                    }
                  # Accelerometer information (typically index 6 in payload)
                if len(payload) > 6 and isinstance(payload[6], list) and len(payload[6]) >= 3:
                    result["sensors"]["accelerometer"] = {
                        "x": payload[6][0],
                        "y": payload[6][1],
                        "z": payload[6][2]
                    }
                
                # Gyro sensor information (typically index 7 in payload)
                if len(payload) > 7 and isinstance(payload[7], list) and len(payload[7]) >= 3:
                    result["sensors"]["gyro"] = {
                        "x": payload[7][0],
                        "y": payload[7][1],
                        "z": payload[7][2]
                    }
                    
                # Position from sensors (derived from payload[8] for coordinates)
                if len(payload) > 8 and isinstance(payload[8], list) and len(payload[8]) >= 3:
                    result["sensors"]["position"] = {
                        "x": payload[8][1],
                        "y": payload[8][2]
                    }
                
            elif message_type == 2:  # Battery status message
                # Extract battery information if available
                if len(payload) > 1:
                    result["battery"] = {
                        "voltage": payload[0] if len(payload) > 0 else None,
                        "percent": payload[1] if len(payload) > 1 else None
                    }
            
            return result
            
        except json.JSONDecodeError as e:
            return {"error": "json_parse_error", "raw": data, "processed": data_str}
        except Exception as e:
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
            lines.append(f"  Distance: {self.sensors.distance}cm")
        if self.sensors.force is not None:
            lines.append(f"  Force: {self.sensors.force}")
        
        if self.sensors.color:
            lines.append(f"  Color - Reflected: {self.sensors.color.reflected}, " 
                        f"Ambient: {self.sensors.color.ambient}, Color: {self.sensors.color.color}")
        
        if self.sensors.gyro:
            lines.append(f"  Gyro - X: {self.sensors.gyro.x}, Y: {self.sensors.gyro.y}, Z: {self.sensors.gyro.z}")
        
        if self.sensors.accelerometer:
            lines.append(f"  Accel - X: {self.sensors.accelerometer.x}, "
                        f"Y: {self.sensors.accelerometer.y}, Z: {self.sensors.accelerometer.z}")
        
        if self.sensors.position:
            lines.append(f"  Position - X: {self.sensors.position.x}, Y: {self.sensors.position.y}")
        
        if self.battery and self.battery.percent is not None:
            lines.append(f"Battery: {self.battery.percent}% ({self.battery.voltage}V)")
        
        return "\n".join(lines)

