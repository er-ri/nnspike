#!/usr/bin/env python3
"""
Color Sensor Only Test Program (Synchronous Version)

This script tests ONLY the color sensor functionality.
No arm movement - just sensor readings.
"""
import time
from nnspike.unit import ETRobot


def display_spike_status(et, interval=0.02, duration=5.0):
    """Display spike status at regular intervals for specified duration"""
    status_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            status_count += 1
            elapsed = time.time() - start_time
            spike_status = et.get_spike_status()
            
            if spike_status and spike_status.sensors and spike_status.sensors.color:
                color_sensor = spike_status.sensors.color
                print(f"[{elapsed:.3f}s] Status #{status_count}: R={color_sensor.reflected}, A={color_sensor.ambient}, C={color_sensor.color}")
            else:
                print(f"[{elapsed:.3f}s] Status #{status_count}: No color sensor data")
                
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
    print("=== Color Sensor Only Test (Sync Version) ===")
    print("This program ONLY reads color sensor values synchronously")
    print("Continuous reading for 5 seconds")
    print("Display interval: 20ms (matches Spike sensor broadcast interval)")
    print("Arm will be moved up then down before test starts")
    print()
    
    # Initialize robot with retry logic
    et = None
    for attempt in range(3):
        try:
            print(f"Attempting to connect to robot (attempt {attempt + 1}/3)...")
            et = ETRobot()
            time.sleep(1.0)
            print("✓ Robot initialized")
            # Move arm up and then down before starting the test
            print("Moving arm up and then down before test...")
            et.move_arm(1)  # 1 = move up
            time.sleep(2.0)  # Wait 1 second
            print("✓ Arm moved up, now moving down...")
            et.move_arm(0)  # 0 = move down
            time.sleep(2.0)  # Wait 1 second
            print("✓ Arm moved down")
            break
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                print("Failed to connect after 3 attempts. Exiting.")
                return
            time.sleep(2.0)
    print("Starting 5-second continuous reading from spike_status...")
    print("Status will be displayed every 0.02 seconds (20ms)")
    print()
    
    try:
        start_time = time.time()
        display_spike_status(et, interval=0.02, duration=5.0)
        elapsed_time = time.time() - start_time
        print(f"\n✓ Test completed in {elapsed_time:.3f} seconds")
        print(f"✓ Used spike_status for color sensor reading")
        print(f"✓ Interval: 0.02 seconds (20ms)")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if et:
            et.stop()
        print("Program finished")


if __name__ == "__main__":
    main()
