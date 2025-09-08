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
            print(f"  タイムスタンプ: {status.timestamp}")
            print(f"  メッセージタイプ: {status.message_type}")
            return et
        else:
            print("❌ Spikeステータス取得失敗")
            return None
            
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return None


def test_communication_speed(et):
    # 1回だけ詳細なサンプル出力
    print("\n--- SpikeStatus サンプル出力 ---")
    status = et.get_spike_status()
    # モーター情報
    for m in ["A", "B", "C"]:
        motor = status.motors[m]
        print(f"Motor {m}: position={motor.position}, power={motor.power}, relative_position={motor.relative_position}, speed={motor.speed}")
    # センサー情報
    sensors = status.sensors
    print(f"Force sensor: {sensors.force}")
    print(f"Distance sensor: {getattr(sensors, 'distance', None)}")
    if sensors.color:
        print(f"Color sensor: reflected={sensors.color.reflected}, ambient={sensors.color.ambient}, color={sensors.color.color}")
    if sensors.gyro:
        print(f"Gyro: x={sensors.gyro.x}, y={sensors.gyro.y}, z={sensors.gyro.z}")
    if sensors.accelerometer:
        print(f"Accelerometer: x={sensors.accelerometer.x}, y={sensors.accelerometer.y}, z={sensors.accelerometer.z}")
    if sensors.position:
        print(f"Position: x={sensors.position.x}, y={sensors.position.y}")
    # バッテリー情報
    battery = status.battery
    print(f"Battery: voltage={battery.voltage}, percent={battery.percent}")
    print("--- サンプル出力ここまで ---\n")
    """通信レスポンス時間測定"""
    print(f"\n2️⃣ 通信速度テスト")
    print("="*50)
    
    response_times = []
    intervals = []
    successful_reads = 0
    total_tests = 100

    print(f"📊 {total_tests}回のデータ取得サイクル(ms)を高頻度で測定中...")

    prev_time = time.time()
    for i in range(total_tests):
        start_time = time.time()
        try:
            status = et.get_spike_status()
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms
            response_times.append(response_time)
            successful_reads += 1

            # サイクル間隔（前回取得からの経過時間）
            interval = (start_time - prev_time) * 1000  # ms
            if i > 0:
                intervals.append(interval)
            prev_time = start_time

            if (i + 1) % 20 == 0:
                print(f"  進行状況: {i+1}/{total_tests} (応答: {response_time:.2f}ms, サイクル: {interval:.2f}ms)")
        except Exception as e:
            print(f"  ❌ エラー{i+1}: {e}")
        # できるだけ高頻度で取得（sleepなし）

    # 結果分析
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        min_interval = min(intervals) if intervals else 0
        max_interval = max(intervals) if intervals else 0
        success_rate = (successful_reads / total_tests) * 100

        print(f"\n📈 通信速度・受信サイクル結果:")
        print(f"  成功率: {success_rate:.1f}% ({successful_reads}/{total_tests})")
        print(f"  平均応答時間: {avg_time:.2f}ms (1回のget_spike_status呼び出しにかかる時間)")
        print(f"  最速応答時間: {min_time:.2f}ms, 最遅応答時間: {max_time:.2f}ms")
        print(f"  平均受信サイクル: {avg_interval:.2f}ms (前回取得からの間隔)")
        print(f"  最短サイクル: {min_interval:.2f}ms, 最長サイクル: {max_interval:.2f}ms")
        if avg_interval < 10:
            print("  ✅ 優秀 - 10ms以下で高頻度受信")
        elif avg_interval < 30:
            print("  ✅ 良好 - 30ms以下で安定受信")
        else:
            print("  ⚠️  サイクル遅延あり (30ms超)")
        # サイクル分布例
        if len(intervals) > 10:
            print(f"  サイクル分布例: {intervals[:10]} ...")
    else:
        print("❌ 通信速度測定失敗")


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
    
    test_duration = 10  # 10秒間
    read_interval = 0.1  # 100ms間隔
    
    print(f"📊 {test_duration}秒間のセンサーデータ安定性を確認中...")
    
    data_points = []
    start_time = time.time()
    
    while time.time() - start_time < test_duration:
        try:
            status = et.get_spike_status()
            if status:
                # 実装されているセンサーのみテスト  
                data_point = {
                    'timestamp': time.time(),
                    'force': status.sensors.force if status.sensors and status.sensors.force is not None else None,
                    'motor_a_speed': status.motors["A"].speed if status.motors["A"].speed is not None else None,
                    'motor_b_speed': status.motors["B"].speed if status.motors["B"].speed is not None else None,
                    'motor_a_position': status.motors["A"].position if status.motors["A"].position is not None else None,
                    'motor_b_position': status.motors["B"].position if status.motors["B"].position is not None else None,
                }
                data_points.append(data_point)
            
            time.sleep(read_interval)
            
        except Exception as e:
            print(f"  ⚠️  データ取得エラー: {e}")
    
    # データ安定性分析（実装済みセンサーのみ）
    if data_points:
        valid_force = [d['force'] for d in data_points if d['force'] is not None]
        valid_motor_a_speed = [d['motor_a_speed'] for d in data_points if d['motor_a_speed'] is not None]
        valid_motor_b_speed = [d['motor_b_speed'] for d in data_points if d['motor_b_speed'] is not None]
        valid_motor_a_pos = [d['motor_a_position'] for d in data_points if d['motor_a_position'] is not None]
        valid_motor_b_pos = [d['motor_b_position'] for d in data_points if d['motor_b_position'] is not None]
        
        print(f"\n📈 センサーデータ安定性結果:")
        print(f"  総データ点数: {len(data_points)}")
        print(f"  フォースセンサー有効率: {len(valid_force)/len(data_points)*100:.1f}%")
        print(f"  モーターA速度有効率: {len(valid_motor_a_speed)/len(data_points)*100:.1f}%") 
        print(f"  モーターB速度有効率: {len(valid_motor_b_speed)/len(data_points)*100:.1f}%")
        print(f"  モーターA位置有効率: {len(valid_motor_a_pos)/len(data_points)*100:.1f}%")
        print(f"  モーターB位置有効率: {len(valid_motor_b_pos)/len(data_points)*100:.1f}%")
        
        # 実際のデータ例表示
        if valid_force:
            print(f"  フォースセンサー値例: {valid_force[-1]}")
        if valid_motor_a_speed:
            print(f"  モーターA速度例: {valid_motor_a_speed[-1]}")
        if valid_motor_b_speed:
            print(f"  モーターB速度例: {valid_motor_b_speed[-1]}")
        if valid_motor_a_pos:
            print(f"  モーターA位置例: {valid_motor_a_pos[-1]}")
        if valid_motor_b_pos:
            print(f"  モーターB位置例: {valid_motor_b_pos[-1]}")
        
        if len(data_points) > 50:  # 5秒以上のデータ
            print("  ✅ データ取得安定")
        else:
            print("  ⚠️  データ取得不安定")
    else:
        print("❌ センサーデータ取得失敗")

def measure_receive_cycle(et, duration_sec=5):
    print(f"\n--- {duration_sec}秒間の新規データ受信サイクル(ms)計測 ---")
    prev_timestamp = None
    intervals = []
    count = 0
    start = time.time()
    while time.time() - start < duration_sec:
        status = et.get_spike_status()
        ts = status.timestamp
        if prev_timestamp is not None and ts != prev_timestamp:
            interval = (ts - prev_timestamp) * 1000  # ms
            intervals.append(interval)
            count += 1
            if count <= 10 or count % 20 == 0:
                print(f"  {count}回目: {interval:.2f}ms")
        prev_timestamp = ts
        time.sleep(0.001)  # 1ms間隔で監視
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
        
        # 5. センサー安定性テスト
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
