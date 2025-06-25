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
import sys
import time
from nnspike.unit import ETRobot


def display_spike_status(et, interval=0.03, duration=5.0, max_recover=3):
    """指定した間隔・時間だけスパイクのステータスを表示する（同期版）
    例外発生時は自動リカバリを試みる
    スレッド・シリアルポートの多重アクセスを防ぐため、再初期化時は必ずstop()してから新インスタンスを返す。
    """
    status_count = 0
    start_time = time.time()
    recover_count = 0
    while time.time() - start_time < duration:
        try:
            status_count += 1
            elapsed = time.time() - start_time
            spike_status = et.get_spike_status()
            if spike_status and spike_status.sensors and spike_status.sensors.color:
                color_sensor = spike_status.sensors.color
                print(f"[{elapsed:.3f}s] ColorSensor: Reflected={color_sensor.reflected}, Ambient={color_sensor.ambient}, Color={color_sensor.color}", flush=True)
            elif spike_status and spike_status.sensors:
                print(f"[{elapsed:.3f}s] Status #{status_count}: No color sensor data (sensors.color=None)", flush=True)
            else:
                print(f"[{elapsed:.3f}s] Status #{status_count}: No color sensor data (sensors=None)", flush=True)
            recover_count = 0  # 成功したらリカバリカウントリセット
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.3f}s] Status #{status_count}: Error - {e}", flush=True)
            recover_count += 1
            if recover_count > max_recover:
                print(f"リカバリ試行{max_recover}回失敗。処理を中断します。", flush=True)
                break
            # ETRobot再初期化を試みる
            try:
                print("ETRobotを再初期化してリカバリを試みます...（stop→新インスタンス）", flush=True)
                et.stop()
            except Exception as e_stop:
                print(f"et.stop()失敗: {e_stop}", flush=True)
            try:
                et = ETRobot()
                time.sleep(1.0)
                print("✓ ETRobot再初期化成功", flush=True)
            except Exception as e2:
                print(f"ETRobot再初期化失敗: {e2}", flush=True)
                time.sleep(2.0)
        # 一定間隔で実行
        next_time = start_time + (status_count * interval)
        current_time = time.time()
        sleep_time = max(0, next_time - current_time)
        time.sleep(sleep_time)
    return et


def main():
    print("=== Color Sensor Only Test (Sync Version) ===", flush=True)
    print("このプログラムはカラーセンサー値のみを同期的に取得します", flush=True)
    print("10秒間連続で読み取りを行います", flush=True)
    print("表示間隔: 30ms (スパイクのセンサーブロードキャスト間隔に合わせる)", flush=True)
    print("テスト開始前にアームを上げてから下げます", flush=True)
    print()
    et = None
    for attempt in range(3):
        try:
            print(f"ロボットに接続中... (試行 {attempt + 1}/3)", flush=True)
            et = ETRobot()
            time.sleep(1.0)
            print("✓ ロボット初期化完了", flush=True)
            print("テスト前にアームを上げてから下げます...", flush=True)
            et.move_arm(1)
            time.sleep(1.0)
            print("✓ アームを上げました。次に下げます...", flush=True)
            et.move_arm(0)
            time.sleep(1.0)
            print("✓ アームを下げました", flush=True)
            break
        except Exception as e:
            print(f"接続試行 {attempt + 1} 失敗: {e}", flush=True)
            if attempt == 2:
                print("3回接続に失敗しました。終了します。", flush=True)
                return
            time.sleep(2.0)
    print("10秒間スパイクのステータスを連続取得します...", flush=True)
    print("0.03秒(30ms)ごとに表示します", flush=True)
    print()
    try:
        start_time = time.time()
        et = display_spike_status(et, interval=0.03, duration=10.0)
        elapsed_time = time.time() - start_time
        print(f"\n✓ テスト完了: {elapsed_time:.3f}秒", flush=True)
        print(f"✓ spike_statusを用いてカラーセンサー値を取得", flush=True)
        print(f"✓ 表示間隔: 0.03秒 (30ms)", flush=True)
    except KeyboardInterrupt:
        print("\nユーザーによってテストが中断されました", flush=True)
    except Exception as e:
        print(f"エラー: {e}", flush=True)
    finally:
        if et:
            try:
                et.stop()
            except Exception as e:
                print(f"et.stop() 実行時に例外: {e}", flush=True)
        print("プログラム終了", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
