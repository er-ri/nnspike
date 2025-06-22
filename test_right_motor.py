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
        print("右・左モーターを1秒ごとに交互に動かします（合計10秒）")
        print("  回数 | 左モーター | 右モーター | 指令値 (L, R)")
        print("------------------------------------------")
        for i in range(10):
            if i % 2 == 0:
                left_cmd, right_cmd = 0, 50
                left_disp, right_disp = "OFF", "ON "
            else:
                left_cmd, right_cmd = 50, 0
                left_disp, right_disp = "ON ", "OFF"
            # 1秒間、30msごとに同じ命令を送信
            t_start = asyncio.get_event_loop().time()
            count = 0
            while asyncio.get_event_loop().time() - t_start < 1.0:
                et.set_motor_forward_power(left_power=left_cmd, right_power=right_cmd)
                if count == 0:
                    print(f" {i+1:2d}回 | {left_disp:^7} | {right_disp:^7} | L={left_cmd:2d}, R={right_cmd:2d}")
                count += 1
                await asyncio.sleep(0.03)  # 30msごと
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
