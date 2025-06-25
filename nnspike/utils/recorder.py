"""
Sensor Data Recorder Module

This module provides a class for recording ETRobot sensor status and control data to CSV files.
It's designed for high-performance logging during robot operation without impacting frame rates.
"""

import csv
import os
import time
import atexit
from typing import Optional, Any


class SensorRecorder:
    """
    High-performance CSV recorder for ETRobot sensor data and control information.
    
    This class manages CSV file creation, writing, and cleanup with optimizations for
    real-time robot operation at high frame rates (30fps+).
    
    Features:
    - Single file open/close for entire session
    - Buffered writing for performance
    - Periodic flushing for data safety
    - Automatic cleanup on program exit
    - Comprehensive sensor and control data logging
    """
    
    def __init__(self, output_dir: str = "storage/sensor_data", timestamp: Optional[str] = None):
        """
        Initialize the sensor recorder.
        
        Args:
            output_dir: Directory to store CSV files
            timestamp: Custom timestamp for filename, auto-generated if None
        """
        self.output_dir = output_dir
        self.timestamp = timestamp or time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.csv_filename = os.path.join(output_dir, f"{self.timestamp}_sensor_log.csv")
        
        self.csv_file = None
        self.csv_writer = None
        self.is_recording = False
        self.frame_count = 0
        
        # CSV headers for sensor data only
        self.headers = [
            'timestamp',
            'frame_number',
            'distance_sensor',
            'force_sensor',
            'color_reflected',
            'color_ambient', 
            'color_value',
            'gyro_x',
            'gyro_y', 
            'gyro_z',
            'accelerometer_x',
            'accelerometer_y',
            'accelerometer_z',
            'position_x',
            'position_y',
            'motor_a_position',
            'motor_a_relative_position',
            'motor_a_speed',
            'motor_a_power',
            'motor_b_position',
            'motor_b_relative_position',
            'motor_b_speed', 
            'motor_b_power',
            'motor_c_position',
            'motor_c_relative_position',
            'motor_c_speed',
            'motor_c_power'
        ]
    
    def start_recording(self) -> None:
        """
        Start CSV recording session.
        
        Creates output directory, opens CSV file, writes headers, and sets up cleanup.
        """
        print(f"[DEBUG][SensorRecorder] start_recording() called. Output: {self.csv_filename}")
        if self.is_recording:
            print(f"[DEBUG][SensorRecorder] Recording already in progress. is_recording={self.is_recording}")
            return
        
        # Ensure directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Open file with buffering for performance
        self.csv_file = open(self.csv_filename, 'w', newline='', buffering=8192)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.headers)
        
        # Register cleanup function
        atexit.register(self.stop_recording)
        
        self.is_recording = True
        self.frame_count = 0
        
        print(f"CSV logging started: {self.csv_filename}")
        print(f"[DEBUG][SensorRecorder] is_recording={self.is_recording}, frame_count={self.frame_count}")
    
    def log_frame_data(self, spike_status) -> None:
        """
        Log sensor data for a single frame.
        
        Args:
            spike_status: SpikeStatus object with sensor data
        """
        if not self.is_recording or self.csv_writer is None:
            print(f"[DEBUG][SensorRecorder] log_frame_data() skipped: is_recording={self.is_recording}, csv_writer={self.csv_writer}")
            return
        
        self.frame_count += 1
        current_time = time.time()
        
        # Extract sensor data from spike_status
        sensors = spike_status.sensors
        motors = spike_status.motors
        
        row_data = [
            current_time,
            self.frame_count,
            self._safe_get(sensors.distance, 0),
            self._safe_get(sensors.force, 0),
            self._safe_get(sensors.color.reflected if sensors.color else None, 0),
            self._safe_get(sensors.color.ambient if sensors.color else None, 0),
            self._safe_get(sensors.color.color if sensors.color else None, 0),
            self._safe_get(sensors.gyro.x if sensors.gyro else None, 0.0),
            self._safe_get(sensors.gyro.y if sensors.gyro else None, 0.0),
            self._safe_get(sensors.gyro.z if sensors.gyro else None, 0.0),
            self._safe_get(sensors.accelerometer.x if sensors.accelerometer else None, 0.0),
            self._safe_get(sensors.accelerometer.y if sensors.accelerometer else None, 0.0),
            self._safe_get(sensors.accelerometer.z if sensors.accelerometer else None, 0.0),
            self._safe_get(sensors.position.x if sensors.position else None, 0.0),
            self._safe_get(sensors.position.y if sensors.position else None, 0.0),
            self._safe_get(motors['A'].position, 0),
            self._safe_get(motors['A'].relative_position, 0),
            self._safe_get(motors['A'].speed, 0),
            self._safe_get(motors['A'].power, 0),
            self._safe_get(motors['B'].position, 0),
            self._safe_get(motors['B'].relative_position, 0),
            self._safe_get(motors['B'].speed, 0),
            self._safe_get(motors['B'].power, 0),
            self._safe_get(motors['C'].position, 0),
            self._safe_get(motors['C'].relative_position, 0),
            self._safe_get(motors['C'].speed, 0),
            self._safe_get(motors['C'].power, 0)
        ]
        
        try:
            self.csv_writer.writerow(row_data)
            
            # Flush periodically to ensure data is saved (every 30 frames ≈ 1 second at 30fps)
            if self.frame_count % 30 == 0:
                print(f"[DEBUG][SensorRecorder] Flushing at frame {self.frame_count}")
                self.csv_file.flush()
                
        except Exception as e:
            print(f"[DEBUG][SensorRecorder] Error writing to CSV: {e}")
            import traceback
            traceback.print_exc()
            raise

    def stop_recording(self) -> None:
        """
        Stop CSV recording session and close file properly.
        """
        print(f"[DEBUG][SensorRecorder] stop_recording() called. is_recording={self.is_recording}")
        if not self.is_recording:
            print(f"[DEBUG][SensorRecorder] Recording not in progress. is_recording={self.is_recording}")
            return
        
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()  # Ensure all data is written
            self.csv_file.close()
            print(f"CSV recording stopped. Data saved to: {self.csv_filename}")
        
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        print(f"[DEBUG][SensorRecorder] stop_recording() finished. frame_count={self.frame_count}")

    def get_filename(self) -> str:
        """
        Get the current CSV filename.
        
        Returns:
            str: Full path to the CSV file
        """
        return self.csv_filename
    
    def get_frame_count(self) -> int:
        """
        Get the current frame count.
        
        Returns:
            int: Number of frames recorded
        """
        print(f"[DEBUG][SensorRecorder] get_frame_count() called. frame_count={self.frame_count}")
        return self.frame_count
    
    @staticmethod
    def _safe_get(value: Any, default: Any = 0) -> Any:
        """
        Safely extract value, returning default if None.
        
        Args:
            value: Any value that might be None
            default: Default value to return if value is None
            
        Returns:
            The value if not None, otherwise the default
        """
        return value if value is not None else default
    
    def __enter__(self):
        """Context manager entry."""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_recording()