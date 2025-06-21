#!/usr/bin/env python3
"""
Ultrasonic Sensor Only Test Program (Asynchronous Version)

This script tests ONLY the ultrasonic sensor functionality using async/await.
No arm movement - just sensor readings.
"""
import time
import asyncio
import argparse
from nnspike.unit import ETRobot


async def display_spike_status_async(et, interval=0.05, duration=5.0):
    """Asynchronously display spike status at regular intervals for specified duration"""
    status_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            status_count += 1
            elapsed = time.time() - start_time
            spike_status = et.get_spike_status()
            
            if spike_status and spike_status.sensors and spike_status.sensors.distance is not None:
                distance_raw = spike_status.sensors.distance
                # The raw value appears to be in cm, not mm as originally assumed
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
        await asyncio.sleep(sleep_time)


async def main_async():
    """Asynchronous main function"""
    print("=== Ultrasonic Sensor Only Test (Async Version) ===")
    print("This program ONLY reads ultrasonic sensor values via spike_status")
    print("Continuous reading for 5 seconds with async processing")
    print("Displaying values from spike_status every 0.05 seconds")
    print()
    
    # Initialize robot with retry logic
    et = None
    for attempt in range(3):
        try:
            print(f"Attempting to connect to robot (attempt {attempt + 1}/3)...")
            et = ETRobot()
            await asyncio.sleep(1.0)  # Async sleep
            print("✓ Robot initialized")
            break
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                print("Failed to connect after 3 attempts. Exiting.")
                return            await asyncio.sleep(2.0)  # Async sleep    
    print("Starting 5-second async continuous reading from spike_status...")
    print("Status will be displayed every 0.05 seconds (50ms)")
    print()
    
    try:
        # Record start time
        start_time = time.time()        # Only run spike status display task (no separate sensor reading)
        status_task = asyncio.create_task(display_spike_status_async(et, interval=0.05, duration=5.0))
        
        # Wait for the status task to complete
        await status_task
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Async test completed in {elapsed_time:.3f} seconds")
        print(f"✓ Used spike_status for ultrasonic sensor reading")
        print(f"✓ Interval: 0.05 seconds (50ms)")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if et:
            et.stop()
        print("Program finished")


def main():
    """Synchronous wrapper for async main"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
