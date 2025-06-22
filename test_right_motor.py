#!/usr/bin/env python3
"""
Right/Left Motor Alternating Test Program (Async Version)

このスクリプトは右・左モーターを1秒ごとに交互に動かすテストを行います。
"""
import asyncio
from nnspike.unit import ETRobot

async def main():
    print("=== Right/Left Motor Alternating Test (Async Version) ===")
    print("右・左モーターを1秒ごとに交互に動かします（合計10秒）")
    print()
    
    et = None
    for attempt in range(3):
        try:
            print(f"Attempting to connect to robot (attempt {attempt + 1}/3)...")
            et = ETRobot()
            await asyncio.sleep(1.0)
            print("✓ Robot initialized")
            break
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                print("Failed to connect after 3 attempts. Exiting.")
                return
            await asyncio.sleep(2.0)
    
    try:
        for i in range(10):
            et.set_motor_forward_power(left_power=0 if i % 2 == 0 else 50, right_power=50 if i % 2 == 0 else 0)
            await asyncio.sleep(0.5)  # 少し待ってから速度取得（応答遅延対策）
            status = et.get_spike_status()
            left_speed = status.motors['B'].speed  # B=左
            right_speed = status.motors['A'].speed # A=右
            if i % 2 == 0:
                print(f"{i+1}回目: 右モーターON, 左モーターOFF | 指令値: L=0, R=50 | 実測速度: L={left_speed}, R={right_speed}")
            else:
                print(f"{i+1}回目: 左モーターON, 右モーターOFF | 指令値: L=50, R=0 | 実測速度: L={left_speed}, R={right_speed}")
            await asyncio.sleep(2.5)  # 残りの2.5秒
        et.set_motor_forward_power(left_power=0, right_power=0)
        print("✓ テスト完了: 両モーター停止")
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
    asyncio.run(main())
