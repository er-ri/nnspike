#!/usr/bin/env python3
"""
Right Motor Only Test Program (Synchronous Version)

このスクリプトは右モーターのみを一定時間動かすテストを行います。
"""
import time
from nnspike.unit import ETRobot

def main():
    print("=== Right Motor Only Test (Sync Version) ===")
    print("右モーターのみを5秒間動かします")
    print()
    
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
    
    try:
        print("右モーターを50%パワーで5秒間動かします...")
        et.set_motor_forward_power(left_power=0, right_power=50)
        time.sleep(5.0)
        et.set_motor_forward_power(left_power=0, right_power=0)
        print("✓ テスト完了: 右モーター停止")
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if et:
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
