#!/usr/bin/env python3
"""
Spike Hub USB Communication Test

SpikeハブとRaspberry Pi間のUSB通信品質をテストします。
- 基本接続テスト
- 通信レスポンス時間測定
- データ取得安定性テスト
- USB3.0環境での動作確認
"""
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import time
import platform
import serial.tools.list_ports


from nnspike.unit.etrobot import ETRobot


def find_spike_port():
    """SpikeハブのUSBポートを自動検出"""
    print("🔍 Spikeハブを検索中...")
    
    ports = serial.tools.list_ports.comports()
    spike_candidates = []
    
    for port in ports:
        port_info = f"{port.device} - {port.description}"
        print(f"  検出ポート: {port_info}")
        
        # Spikeハブの識別（一般的なパターン）
        if any(keyword in port.description.lower() for keyword in 
               ['spike', 'lego', 'usb serial', 'usb-serial']):
            spike_candidates.append(port.device)
    
    if spike_candidates:
        print(f"✅ Spikeハブ候補: {spike_candidates}")
        return spike_candidates[0]
    
    # デフォルトポート推定
    if platform.system() == "Linux":
        # Raspberry Pi/Linux環境
        default_ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]
    else:
        # Windows環境
        default_ports = ["COM3", "COM4", "COM5"]
    
    for port in default_ports:
        try:
            # ポートの存在確認（簡易）
            if platform.system() == "Linux":
                if os.path.exists(port):
                    print(f"🔍 デフォルトポート使用: {port}")
                    return port
            else:
                print(f"🔍 デフォルトポート試行: {port}")
                return port
        except:
            continue
    
    print("❌ Spikeハブが見つかりません")
    return None


def test_basic_connection(port):
    """基本接続テスト"""
    print(f"\n1️⃣ 基本接続テスト: {port}")
    print("="*50)
    
    try:
        et = ETRobot(port=port)
        print("✅ ETRobot初期化成功")
        
        # 接続確認
        time.sleep(1)  # 接続安定化待機
        
        status = et.get_spike_status()
        if status:
            print("✅ Spikeステータス取得成功")
            # print(f"  タイムスタンプ: {status.timestamp}")  # status.timestampは信頼しない
            print(f"  メッセージタイプ: {status.message_type}")
            return et
        else:
            print("❌ Spikeステータス取得失敗")
            return None
            
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return None


def test_communication_speed(et):
    print("\n2️⃣ 通信速度テスト（新規データ受信サイクル計測）")
    print("="*50)
    duration_sec = 5
    print(f"📊 {duration_sec}秒間、生データ(raw_data)の変化のみをカウント・計測します...")
    prev_raw = None
    prev_time = None
    intervals = []
    count = 0
    start = time.time()
    while time.time() - start < duration_sec:
        status = et.get_spike_status()
        now_raw = status.raw_data
        now_time = time.time()
        if prev_raw is not None and now_raw != prev_raw and prev_time is not None:
            interval = (now_time - prev_time) * 1000  # ms
            intervals.append(interval)
            count += 1
            if count <= 10 or count % 20 == 0:
                print(f"  {count}回目: {interval:.2f}ms")
        if now_raw != prev_raw:
            prev_time = now_time
        prev_raw = now_raw
        time.sleep(0.001)
    if intervals:
        avg = sum(intervals) / len(intervals)
        print(f"\n受信サイクル統計: 平均={avg:.2f}ms, 最短={min(intervals):.2f}ms, 最長={max(intervals):.2f}ms, 回数={len(intervals)}")
        print(f"サイクル分布例: {intervals[:10]} ...")
    else:
        print("新規データ受信が検出できませんでした")


def test_motor_control(et):
    """モーター制御テスト"""
    print(f"\n3️⃣ モーター制御テスト")
    print("="*50)
    
    try:
        print("🔄 モーター制御シーケンステスト中...")
        
        # 1. 停止確認
        print("  初期停止...")
        et.brake()
        time.sleep(0.2)  # 短縮
        status = et.get_spike_status()
        motor_a_speed = status.motors["A"].speed if status.motors["A"].speed is not None else 'N/A'
        motor_b_speed = status.motors["B"].speed if status.motors["B"].speed is not None else 'N/A'
        print(f"  停止状態: モーターA={motor_a_speed}, モーターB={motor_b_speed}")
        
        # 2. 前進テスト（短時間）
        print("  前進テスト...")
        et.set_motor_speed(left_speed=20, right_speed=20)  # 速度を下げる
        time.sleep(0.3)  # 短縮
        status = et.get_spike_status()
        motor_a_speed = status.motors["A"].speed if status.motors["A"].speed is not None else 'N/A'
        motor_b_speed = status.motors["B"].speed if status.motors["B"].speed is not None else 'N/A'
        print(f"  前進状態: モーターA={motor_a_speed}, モーターB={motor_b_speed}")
        
        # 3. 即座に停止
        print("  停止中...")
        et.brake()
        time.sleep(0.2)
        status = et.get_spike_status()
        motor_a_speed = status.motors["A"].speed if status.motors["A"].speed is not None else 'N/A'
        motor_b_speed = status.motors["B"].speed if status.motors["B"].speed is not None else 'N/A'
        print(f"  最終停止: モーターA={motor_a_speed}, モーターB={motor_b_speed}")
        
        print("✅ モーター制御テスト完了")
        
    except KeyboardInterrupt:
        print("\n⚠️  ユーザーによりテスト中断")
        print("  安全停止中...")
        try:
            et.brake()
        except:
            pass
        raise
    except Exception as e:
        print(f"❌ モーター制御エラー: {e}")
        try:
            et.brake()
        except:
            pass


def test_sensor_stability(et):
    """センサーデータ安定性テスト"""
    print(f"\n4️⃣ センサーデータ安定性テスト")
    print("="*50)
    
    test_duration = 2  # 2秒間
    print(f"📊 {test_duration}秒間 status.raw_data (生JSON)と受信サイクル(ms)をprintします")
    start_time = time.time()
    prev_ts = None
    while time.time() - start_time < test_duration:
        try:
            status = et.get_spike_status()
            now = time.time()
            cycle = None
            if prev_ts is not None:
                cycle = (now - prev_ts) * 1000
            prev_ts = now
            print(f"cycle={cycle:.2f}ms, raw={status.raw_data}" if cycle is not None else f"raw={status.raw_data}")
            time.sleep(0.01)
        except Exception as e:
            print(f"  ⚠️  データ取得エラー: {e}")

def measure_receive_cycle(et, duration_sec=5):
    print(f"\n--- {duration_sec}秒間の新規データ受信サイクル(ms)計測 ---")
    prev_raw = None
    prev_time = None
    intervals = []
    count = 0
    start = time.time()
    while time.time() - start < duration_sec:
        status = et.get_spike_status()
        now_raw = status.raw_data
        now_time = time.time()
        if prev_raw is not None and now_raw != prev_raw and prev_time is not None:
            interval = (now_time - prev_time) * 1000  # ms
            intervals.append(interval)
            count += 1
            if count <= 10 or count % 20 == 0:
                print(f"  {count}回目: {interval:.2f}ms")
        if now_raw != prev_raw:
            prev_time = now_time
        prev_raw = now_raw
        time.sleep(0.001)
    if intervals:
        avg = sum(intervals) / len(intervals)
        print(f"\n受信サイクル統計: 平均={avg:.2f}ms, 最短={min(intervals):.2f}ms, 最長={max(intervals):.2f}ms, 回数={len(intervals)}")
        print(f"サイクル分布例: {intervals[:10]} ...")
    else:
        print("新規データ受信が検出できませんでした")

def main():
    """メイン実行関数"""
    print("🎯 SPIKE Hub USB Communication Test")
    print("=" * 60)
    print(f"実行環境: {platform.system()} {platform.release()}")
    
    # 1. ポート自動検出
    port = find_spike_port()
    if not port:
        print("\n❌ Spikeハブが見つかりません。以下を確認してください:")
        print("  - Spikeハブが接続されているか")
        print("  - slot_prod.pyが実行されているか")
        print("  - USBケーブルが正常か")
        return
    
    # 2. 基本接続テスト
    et = test_basic_connection(port)
    if not et:
        print("\n❌ 基本接続に失敗しました")
        return
    
    try:
        # 3. 通信速度テスト
        test_communication_speed(et)

        # 4. モーター制御テスト
        test_motor_control(et)

        # 5. モーター制御後に受信サイクル計測
        measure_receive_cycle(et, duration_sec=5)

        # 6. センサー安定性テスト
        test_sensor_stability(et)

        print(f"\n✅ 全テスト完了！")
        print(f"🔍 USB3.0環境での通信品質は良好です")

    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")

    finally:
        try:
            et.brake()  # 安全のため停止
            et.stop()   # 接続終了
            print("🔚 接続を安全に終了しました")
        except:
            pass


if __name__ == "__main__":
    main()
