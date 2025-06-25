#!/usr/bin/env python3
"""
Color Sensor Only Test Program (Synchronous Version)

このスクリプトはカラーセンサーの動作のみをテストします。
- アームの動作は開始時のみ（上げてから下げる）。
- 以降はカラーセンサー値の取得・表示のみを行います。
- 取得間隔は30ms（スパイク側のセンサーブロードキャスト間隔に合わせる）。
- 例外発生時は自動リカバリ（ETRobot再初期化）を最大3回まで試みます。
- ログはflush=Trueでリアルタイム表示されます。
"""
import time
from nnspike.unit import ETRobot


# def display_spike_status(et, interval=0.03, duration=5.0, max_recover=3):
#     """指定した間隔・時間だけスパイクのステータスを表示する（同期版）
#     例外発生時は自動リカバリを試みる
#     スレッド・シリアルポートの多重アクセスを防ぐため、再初期化時は必ずstop()してから新インスタンスを返す。
#     """
#     status_count = 0
#     start_time = time.time()
#     recover_count = 0
#     while time.time() - start_time < duration:
#         try:
#             status_count += 1
#             elapsed = time.time() - start_time
#             spike_status = et.get_spike_status()
#             if spike_status and spike_status.sensors and spike_status.sensors.color:
#                 color_sensor = spike_status.sensors.color
#                 print(f"[{elapsed:.3f}s] ColorSensor: Reflected={color_sensor.reflected}, Ambient={color_sensor.ambient}, Color={color_sensor.color}", flush=True)
#             elif spike_status and spike_status.sensors:
#                 print(f"[{elapsed:.3f}s] Status #{status_count}: No color sensor data (sensors.color=None)")
#             else:
#                 print(f"[{elapsed:.3f}s] Status #{status_count}: No color sensor data (sensors=None)")
#             recover_count = 0  # 成功したらリカバリカウントリセット
#         except Exception as e:
#             elapsed = time.time() - start_time
#             print(f"[{elapsed:.3f}s] Status #{status_count}: Error - {e}")
#             recover_count += 1
#             if recover_count > max_recover:
#                 print(f"リカバリ試行{max_recover}回失敗。処理を中断します。")
#                 break
#             # ETRobot再初期化を試みる
#             try:
#                 print("ETRobotを再初期化してリカバリを試みます...（stop→新インスタンス）")
#                 et.stop()
#             except Exception as e_stop:
#                 print(f"et.stop()失敗: {e_stop}")
#             try:
#                 et = ETRobot()
#                 time.sleep(1.0)
#                 print("✓ ETRobot再初期化成功")
#             except Exception as e2:
#                 print(f"ETRobot再初期化失敗: {e2}")
#                 time.sleep(2.0)
#         # 一定間隔で実行
#         next_time = start_time + (status_count * interval)
#         current_time = time.time()
#         sleep_time = max(0, next_time - current_time)
#         time.sleep(sleep_time)
#     return et


def main():
    et = ETRobot()
    try:
        # アームを上げてから下げる
        et.move_arm(1)  # 上げる
        time.sleep(1.0)
        et.move_arm(0)  # 下げる
        time.sleep(1.0)
        for _ in range(int(10/0.03)):
            try:
                s = et.get_spike_status()
                if s and s.sensors and s.sensors.color:
                    c = s.sensors.color
                    print(f"Reflected={c.reflected}, Ambient={c.ambient}, Color={c.color}", flush=True)
                else:
                    print("No color sensor data", flush=True)
            except Exception as e:
                print(f"Error: {e}", flush=True)
            time.sleep(0.03)
    finally:
        et.stop()


if __name__ == "__main__":
    main()
