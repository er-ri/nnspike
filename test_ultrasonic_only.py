#!/usr/bin/env python3
"""
Ultrasonic Sensor Only Test Program (Synchronous Version)

This script tests ONLY the ultrasonic sensor functionality.
No arm movement - just sensor readings.
"""
import time
from nnspike.unit import ETRobot


def display_spike_status(et, interval=0.02, duration=20.0):
    """Display spike status at regular intervals for specified duration"""
    status_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            status_count += 1
            elapsed = time.time() - start_time
            spike_status = et.get_spike_status()
            
            if spike_status and spike_status.sensors and spike_status.sensors.distance is not None:
                distance_raw = spike_status.sensors.distance
                distance_cm = distance_raw  # Raw value is actually in cm
                distance_mm = distance_raw * 10.0  # Convert cm to mm
                print(f"[{elapsed:.3f}s] Status #{status_count}: Distance={distance_cm}cm ({distance_mm}mm)")
            else:
                print(f"[{elapsed:.3f}s] Status #{status_count}: No ultrasonic sensor data")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.3f}s] Status #{status_count}: Error - {e}")
        
        # Ensure consistent timing regardless of processing time
        next_time = start_time + (status_count * interval)
        current_time = time.time()
        sleep_time = max(0, next_time - current_time)
        time.sleep(sleep_time)


def main():
    """Synchronous main function"""
    print("=== Ultrasonic Sensor Only Test (Sync Version) ===")
    print("This program ONLY reads ultrasonic sensor values via spike_status")
    print("Continuous reading for 20 seconds")
    print("Displaying values from spike_status every 0.02 seconds (20ms)")
    print()
    
    # Initialize robot with retry logic
    et = None
    for attempt in range(3):
        try:
            print(f"Attempting to connect to robot (attempt {attempt + 1}/3)...")
            et = ETRobot()
            time.sleep(1.0)
            print("✓ Robot initialized")
            break
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                print("Failed to connect after 3 attempts. Exiting.")
                return
            time.sleep(2.0)
    
    print("Starting 20-second continuous reading from spike_status...")
    print("Status will be displayed every 0.02 seconds (20ms)")
    print()
    
    try:
        start_time = time.time()
        display_spike_status(et, interval=0.02, duration=20.0)
        elapsed_time = time.time() - start_time
        print(f"\n✓ Test completed in {elapsed_time:.3f} seconds")
        print(f"✓ Used spike_status for ultrasonic sensor reading")
        print(f"✓ Interval: 0.02 seconds (20ms)")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if et:
            # 明示的にSTOPコマンドを送信
            try:
                id_byte = et.COMMAND_STOP_MOTOR_ID.to_bytes(1, "big")
                dummy1 = (0).to_bytes(1, "big")
                dummy2 = (0).to_bytes(1, "big")
                command = id_byte + dummy1 + dummy2
                et._ETRobot__send_command(command)
            except Exception as e:
                print(f"Error sending STOP command: {e}")
            et.stop()
        print("Program finished")


if __name__ == "__main__":
    main()
