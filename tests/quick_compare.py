#!/usr/bin/env python3
"""
同時実行機体比較ツール

各Raspberry Piで同じプログラムを実行し、IPアドレスで機体を識別。
結果をコピペして比較できる形式で出力します。
"""

import sys
import time
import statistics
import socket
import subprocess
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from nnspike.unit import ETRobot
from nnspike.constants import HIGH_SPEED_BASE

def get_robot_info():
    """機体情報を自動取得"""
    try:
        # IPアドレス取得
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # より正確なIPアドレス取得を試行
        try:
            # 外部接続を試みてローカルIPを取得
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            pass
        
        # MACアドレス取得（追加識別用）
        try:
            mac_result = subprocess.run(['cat', '/sys/class/net/wlan0/address'], 
                                      capture_output=True, text=True)
            mac_address = mac_result.stdout.strip() if mac_result.returncode == 0 else "unknown"
        except:
            mac_address = "unknown"
        
        # 機体名生成
        ip_suffix = local_ip.split('.')[-1]
        robot_name = f"Robot-{ip_suffix}"
        
        return {
            'robot_name': robot_name,
            'ip_address': local_ip,
            'hostname': hostname,
            'mac_address': mac_address[-6:] if mac_address != "unknown" else "unknown"  # 末尾6文字
        }
        
    except Exception as e:
        print(f"機体情報取得エラー: {e}")
        return {
            'robot_name': f"Robot-{time.strftime('%H%M')}",
            'ip_address': "unknown",
            'hostname': "unknown", 
            'mac_address': "unknown"
        }

def single_robot_performance_test():
    """単一機体の性能テスト（同時実行用）"""
    print("=" * 60)
    print("🤖 機体性能テスト (同時実行版)")
    print("=" * 60)
    
    # 機体情報を自動取得
    robot_info = get_robot_info()
    
    print(f"機体識別: {robot_info['robot_name']}")
    print(f"IPアドレス: {robot_info['ip_address']}")
    print(f"ホスト名: {robot_info['hostname']}")
    print(f"MAC(末尾): {robot_info['mac_address']}")
    
    et = ETRobot()
    
    try:
        print("\n🔧 テスト設定:")
        print("- 測定時間: 各3秒 × 3回 = 合計9秒")
        print("- テスト速度: 80, 98, 110")
        print("- 車輪を浮かしてください")
        
        input("\n準備完了? Enterで開始...")
        
        test_speeds = [80, 98, 110]
        all_results = []
        
        for i, speed in enumerate(test_speeds, 1):
            print(f"\n[{i}/3] 速度{speed}テスト開始...")
            result = quick_motor_test_internal(et, speed, duration=3)
            
            if result:
                result.update(robot_info)
                all_results.append(result)
                print(f"  ✅ 速度={result['avg_speed']:.1f}, 乖離={result['speed_deviation']:.1f}, 制御精度={result['control_accuracy']:.3f}")
            else:
                print(f"  ❌ 速度{speed}のテスト失敗")
            
            # 休憩（最後以外）
            if i < len(test_speeds):
                print("  2秒休憩...")
                time.sleep(2)
        
        # 総合結果計算
        if all_results:
            calculate_overall_performance(all_results, robot_info)
            output_copyable_results(all_results, robot_info)
        else:
            print("❌ 有効なテスト結果がありません")
            
    except KeyboardInterrupt:
        print("\n⚠️ テスト中断")
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        et.brake()
        et.stop()

def quick_motor_test_internal(et, test_speed, duration=3):
    """内部用モーターテスト関数"""
    try:
        # データ収集リスト
        speeds_a, speeds_b = [], []
        powers_a, powers_b = [], []
        
        frames = duration * 20  # 20fps（軽量化）
        stabilization = 10  # 0.5秒安定化
        
        print(f"    測定中... ", end="", flush=True)
        
        for i in range(frames):
            et.set_motor_forward_speed(left_speed=test_speed, right_speed=test_speed)
            
            try:
                status = et.get_spike_status()
                motor_a = status.motors.get("A")
                motor_b = status.motors.get("B")
                
                # 安定化期間後のデータを記録
                if i >= stabilization and motor_a and motor_b:
                    speeds_a.append(abs(motor_a.speed) if motor_a.speed else 0)
                    speeds_b.append(abs(motor_b.speed) if motor_b.speed else 0)
                    powers_a.append(abs(motor_a.power) if motor_a.power else 0)
                    powers_b.append(abs(motor_b.power) if motor_b.power else 0)
                
                # 進捗表示
                if i % 20 == 0:
                    print(".", end="", flush=True)
                
            except:
                pass
            
            time.sleep(0.05)  # 20fps
        
        et.brake()
        print(" 完了")
        
        # 結果計算
        if len(speeds_a) >= 10:  # 最低0.5秒分のデータ
            avg_speed = (statistics.mean(speeds_a) + statistics.mean(speeds_b)) / 2
            avg_power = (statistics.mean(powers_a) + statistics.mean(powers_b)) / 2
            speed_stability = (statistics.stdev(speeds_a) + statistics.stdev(speeds_b)) / 2 if len(speeds_a) > 1 else 0
            achievement_rate = avg_speed / test_speed
            efficiency = avg_speed / avg_power if avg_power > 0 else 0
            
            # 🎯 制御精度指標（指定速度との乖離）
            speed_deviation = abs(test_speed - avg_speed)
            control_accuracy = 1.0 - (speed_deviation / test_speed)  # 1.0が完璧
            
            return {
                'test_speed': test_speed,
                'avg_speed': avg_speed,
                'avg_power': avg_power,
                'speed_stability': speed_stability,
                'achievement_rate': achievement_rate,
                'efficiency': efficiency,
                'speed_deviation': speed_deviation,  # 指定速度との乖離
                'control_accuracy': control_accuracy,  # 制御精度
                'samples': len(speeds_a)
            }
        else:
            return None
            
    except Exception as e:
        print(f"\n    テストエラー: {e}")
        return None

def calculate_overall_performance(results, robot_info):
    """総合性能を計算（制御安定性重視）"""
    if not results:
        return
    
    # 各指標の計算
    avg_speeds = [r['avg_speed'] for r in results]
    stabilities = [r['speed_stability'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    achievements = [r['achievement_rate'] for r in results]
    
    # 🎯 制御安定性指標（最重要）: 指定速度との乖離
    control_accuracy_scores = []
    for r in results:
        # 乖離率を計算 (低いほど良い制御)
        deviation_rate = abs(r['test_speed'] - r['avg_speed']) / r['test_speed']
        # 制御精度スコア (高いほど良い: 1.0が完璧、0に近づくほど悪い)
        control_accuracy = 1.0 - deviation_rate
        control_accuracy_scores.append(control_accuracy)
    
    overall = {
        'total_avg_speed': statistics.mean(avg_speeds),
        'total_stability': statistics.mean(stabilities),  # 速度変動
        'total_efficiency': statistics.mean(efficiencies),
        'total_achievement': statistics.mean(achievements),
        'speed_consistency': statistics.stdev(avg_speeds) if len(avg_speeds) > 1 else 0,
        # 🎯 制御精度（最重要指標）
        'control_accuracy': statistics.mean(control_accuracy_scores),  # 指定速度への制御精度
        'control_consistency': statistics.stdev(control_accuracy_scores) if len(control_accuracy_scores) > 1 else 0
    }
    
    # 各速度での乖離を詳細表示
    print(f"\n📊 {robot_info['robot_name']} 制御精度分析:")
    for i, r in enumerate(results):
        deviation = abs(r['test_speed'] - r['avg_speed'])
        deviation_rate = deviation / r['test_speed'] * 100
        control_score = control_accuracy_scores[i]
        print(f"  速度{r['test_speed']:>3}: 実測{r['avg_speed']:>5.1f} (乖離{deviation:>4.1f}, {deviation_rate:>4.1f}%, 制御精度{control_score:>5.3f})")
    
    # 総合スコア計算（制御精度を最重視）
    performance_score = (
        overall['control_accuracy'] * 100 * 0.40 +  # 40% - 制御精度（最重要）
        overall['total_avg_speed'] * 0.25 +         # 25% - 実速度
        (100 / (overall['total_stability'] + 1)) * 0.15 +  # 15% - 速度変動安定性
        overall['total_efficiency'] * 10 * 0.10 +   # 10% - 効率性
        overall['total_achievement'] * 50 * 0.05 +  # 5% - 達成率
        (10 / (overall['control_consistency'] + 0.01)) * 0.05  # 5% - 制御一貫性
    )
    
    overall['performance_score'] = performance_score
    
    print(f"\n📊 {robot_info['robot_name']} 総合性能:")
    print(f"  🎯 制御精度: {overall['control_accuracy']:.3f} (1.000が完璧)")
    print(f"  平均速度: {overall['total_avg_speed']:.1f}")
    print(f"  速度安定性: ±{overall['total_stability']:.1f}")
    print(f"  効率性: {overall['total_efficiency']:.2f}")
    print(f"  達成率: {overall['total_achievement']:.3f} ({overall['total_achievement']*100:.1f}%)")
    print(f"  制御一貫性: ±{overall['control_consistency']:.3f}")
    print(f"  🏆 総合スコア: {performance_score:.1f}")
    
    return overall

def output_copyable_results(results, robot_info):
    """コピペ用の結果を出力"""
    print(f"\n" + "="*60)
    print("📋 結果サマリー (コピペ用)")
    print("="*60)
    
    # 機体情報
    print(f"機体: {robot_info['robot_name']} (IP: {robot_info['ip_address']})")
    
    # 各速度の結果
    print("詳細結果:")
    for r in results:
        deviation_rate = r['speed_deviation'] / r['test_speed'] * 100
        print(f"  速度{r['test_speed']:>3}: 実速度={r['avg_speed']:>5.1f}, 乖離={r['speed_deviation']:>4.1f}({deviation_rate:>4.1f}%), 制御精度={r['control_accuracy']:>5.3f}, 安定性=±{r['speed_stability']:>4.1f}")
    
    # 総合指標
    if results:
        overall = calculate_overall_performance(results, robot_info)
        if overall:
            print(f"🎯 制御精度総合: {overall['control_accuracy']:.3f}")
            print(f"総合スコア: {overall['performance_score']:.1f}")
    
    # 比較用フォーマット（制御精度重視）
    print(f"\n比較用データ:")
    print(f"{robot_info['robot_name']},{robot_info['ip_address']}", end="")
    for r in results:
        print(f",{r['avg_speed']:.1f},{r['speed_deviation']:.1f},{r['control_accuracy']:.3f},{r['speed_stability']:.1f}", end="")
    if results:
        overall = calculate_overall_performance(results, robot_info)
        if overall:
            print(f",{overall['control_accuracy']:.3f},{overall['performance_score']:.1f}")
    print()
    
    print(f"\n⏰ テスト完了時刻: {time.strftime('%H:%M:%S')}")
    print("="*60)

def parse_comparison_data():
    """コピペされた比較データを解析（制御精度重視）"""
    print("=" * 60)
    print("📊 機体比較データ解析 (制御精度重視)")
    print("=" * 60)
    print("各機体の比較用データをペーストしてください")
    print("形式: Robot-XXX,IP,速度80実測,乖離80,制御精度80,安定性80,速度98実測,乖離98,制御精度98,安定性98,速度110実測,乖離110,制御精度110,安定性110,総合制御精度,総合スコア")
    print("例: Robot-100,192.168.1.100,79.5,0.5,0.994,1.2,96.8,1.2,0.988,1.1,98.2,11.8,0.893,1.3,0.958,142.5")
    print("\n1台目のデータを入力してください:")
    
    try:
        data1 = input("> ").strip()
        if not data1:
            print("❌ データが入力されませんでした")
            return
        
        print("\n2台目のデータを入力してください:")
        data2 = input("> ").strip()
        if not data2:
            print("❌ データが入力されませんでした")
            return
        
        # データ解析
        robot1 = parse_robot_data_control_focused(data1)
        robot2 = parse_robot_data_control_focused(data2)
        
        if robot1 and robot2:
            display_comparison_control_focused(robot1, robot2)
        else:
            print("❌ データの解析に失敗しました")
            
    except Exception as e:
        print(f"❌ エラー: {e}")

def parse_robot_data_control_focused(data_line):
    """機体データを解析（制御精度重視）"""
    try:
        parts = data_line.split(',')
        if len(parts) < 16:
            print(f"❌ データが不足しています: {len(parts)}/16項目")
            return None
        
        return {
            'name': parts[0],
            'ip': parts[1],
            # 速度80
            'speed_80': float(parts[2]),
            'deviation_80': float(parts[3]),
            'control_80': float(parts[4]),
            'stability_80': float(parts[5]),
            # 速度98  
            'speed_98': float(parts[6]),
            'deviation_98': float(parts[7]),
            'control_98': float(parts[8]),
            'stability_98': float(parts[9]),
            # 速度110
            'speed_110': float(parts[10]),
            'deviation_110': float(parts[11]),
            'control_110': float(parts[12]),
            'stability_110': float(parts[13]),
            # 総合
            'total_control_accuracy': float(parts[14]),
            'total_score': float(parts[15])
        }
    except (ValueError, IndexError) as e:
        print(f"❌ データ解析エラー: {e}")
        return None

def display_comparison_control_focused(robot1, robot2):
    """比較結果を表示（制御精度重視）"""
    print(f"\n" + "=" * 80)
    print("🏆 機体性能比較結果 (制御精度重視)")
    print("=" * 80)
    
    print(f"\n機体情報:")
    print(f"  {robot1['name']} (IP: {robot1['ip']})")
    print(f"  {robot2['name']} (IP: {robot2['ip']})")
    
    # 🎯 制御精度比較テーブル（最重要）
    print(f"\n{'制御精度項目':>15} {robot1['name']:>15} {robot2['name']:>15} {'優勢':>10}")
    print("-" * 70)
    
    # 制御精度比較（高いほど良い）
    control_items = [
        ('制御精度80', 'control_80', '%.3f'),
        ('制御精度98', 'control_98', '%.3f'),
        ('制御精度110', 'control_110', '%.3f'),
        ('総合制御精度', 'total_control_accuracy', '%.3f')
    ]
    
    control_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in control_items:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 > val2 else robot2['name']
        control_wins[winner] += 1
        diff = abs(val1 - val2)
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10} (差{diff:.3f})")
    
    # 🎯 乖離度比較（低いほど良い）
    print(f"\n{'乖離度項目':>15} {robot1['name']:>15} {robot2['name']:>15} {'優勢':>10}")
    print("-" * 70)
    
    deviation_items = [
        ('乖離度80', 'deviation_80', '%.1f'),
        ('乖離度98', 'deviation_98', '%.1f'),
        ('乖離度110', 'deviation_110', '%.1f')
    ]
    
    deviation_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in deviation_items:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 < val2 else robot2['name']  # 低い方が勝ち
        deviation_wins[winner] += 1
        diff = abs(val1 - val2)
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10} (差{diff:.1f})")
    
    # 実速度比較
    print(f"\n{'実速度項目':>15} {robot1['name']:>15} {robot2['name']:>15} {'優勢':>10}")
    print("-" * 70)
    
    speed_items = [
        ('実速度80', 'speed_80', '%.1f'),
        ('実速度98', 'speed_98', '%.1f'),
        ('実速度110', 'speed_110', '%.1f')
    ]
    
    speed_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in speed_items:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 > val2 else robot2['name']
        speed_wins[winner] += 1
        diff = abs(val1 - val2)
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10} (差{diff:.1f})")
    
    # 安定性比較（低い方が良い）
    print(f"\n{'安定性項目':>15} {robot1['name']:>15} {robot2['name']:>15} {'優勢':>10}")
    print("-" * 70)
    
    stability_items = [
        ('変動±80', 'stability_80', '%.1f'),
        ('変動±98', 'stability_98', '%.1f'),
        ('変動±110', 'stability_110', '%.1f')
    ]
    
    stability_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in stability_items:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 < val2 else robot2['name']  # 低い方が勝ち
        stability_wins[winner] += 1
        diff = abs(val1 - val2)
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10} (差{diff:.1f})")
    
    # 総合スコア
    total_winner = robot1['name'] if robot1['total_score'] > robot2['total_score'] else robot2['name']
    score_diff = abs(robot1['total_score'] - robot2['total_score'])
    
    print(f"{'総合スコア':>15}: {robot1['total_score']:>15.1f} {robot2['total_score']:>15.1f} {total_winner:>10} (差{score_diff:.1f})")
    
    # 🎯 総合判定（制御精度を最重視）
    print(f"\n" + "=" * 80)
    print("🎯 総合判定 (制御精度重視)")
    print("=" * 80)
    
    # 制御精度による優勝判定
    control_leader = robot1['name'] if robot1['total_control_accuracy'] > robot2['total_control_accuracy'] else robot2['name']
    control_diff = abs(robot1['total_control_accuracy'] - robot2['total_control_accuracy'])
    
    print(f"🏆 制御精度チャンピオン: {control_leader}")
    print(f"   制御精度差: {control_diff:.3f} (1.000が完璧)")
    print(f"   制御精度勝利項目: {control_wins[control_leader]}/4項目")
    print(f"   乖離度勝利項目: {deviation_wins[control_leader]}/3項目")
    
    # 各カテゴリーの成績
    print(f"\n📊 カテゴリー別成績:")
    print(f"   制御精度: {robot1['name']} {control_wins[robot1['name']]}-{control_wins[robot2['name']]} {robot2['name']}")
    print(f"   乖離度: {robot1['name']} {deviation_wins[robot1['name']]}-{deviation_wins[robot2['name']]} {robot2['name']}")
    print(f"   実速度: {robot1['name']} {speed_wins[robot1['name']]}-{speed_wins[robot2['name']]} {robot2['name']}")
    print(f"   安定性: {robot1['name']} {stability_wins[robot1['name']]}-{stability_wins[robot2['name']]} {robot2['name']}")
    
    # 制御精度の評価
    if control_diff < 0.005:
        control_evaluation = "制御精度は互角"
    elif control_diff < 0.02:
        control_evaluation = "制御精度に明確な差"
    else:
        control_evaluation = "制御精度に大きな差"
    
    print(f"   評価: {control_evaluation}")
    
    # 🎯 実用的推奨
    print(f"\n💡 実用的推奨:")
    
    # 制御重視の推奨
    print(f"   🎯 制御精度重視: {control_leader} (制御精度 {max(robot1['total_control_accuracy'], robot2['total_control_accuracy']):.3f})")
    
    # 速度重視の推奨
    avg_speed_1 = (robot1['speed_80'] + robot1['speed_98'] + robot1['speed_110']) / 3
    avg_speed_2 = (robot2['speed_80'] + robot2['speed_98'] + robot2['speed_110']) / 3
    speed_leader = robot1['name'] if avg_speed_1 > avg_speed_2 else robot2['name']
    print(f"   🚀 高速性能重視: {speed_leader} (平均実速度 {max(avg_speed_1, avg_speed_2):.1f})")
    
    # 安定性重視の推奨
    avg_stability_1 = (robot1['stability_80'] + robot1['stability_98'] + robot1['stability_110']) / 3
    avg_stability_2 = (robot2['stability_80'] + robot2['stability_98'] + robot2['stability_110']) / 3
    stability_leader = robot1['name'] if avg_stability_1 < avg_stability_2 else robot2['name']
    print(f"   📊 変動安定性重視: {stability_leader} (平均変動 ±{min(avg_stability_1, avg_stability_2):.1f})")
    
    # 最終推奨
    print(f"\n🏆 最終推奨: {control_leader}")
    print(f"   理由: 制御精度が最重要指標であり、指定速度への追従性能が優秀")
    
    return control_leader

def display_comparison(robot1, robot2):
    """比較結果を表示"""
    print(f"\n" + "=" * 80)
    print("🏆 機体性能比較結果")
    print("=" * 80)
    
    print(f"\n機体情報:")
    print(f"  {robot1['name']} (IP: {robot1['ip']})")
    print(f"  {robot2['name']} (IP: {robot2['ip']})")
    
    # 比較テーブル
    print(f"\n{'テスト項目':>15} {robot1['name']:>15} {robot2['name']:>15} {'優勢':>10}")
    print("-" * 70)
    
    # 速度比較
    speeds = [
        ('速度80実測', 'speed_80', '%.1f'),
        ('速度98実測', 'speed_98', '%.1f'), 
        ('速度110実測', 'speed_110', '%.1f')
    ]
    
    speed_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in speeds:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 > val2 else robot2['name']
        speed_wins[winner] += 1
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10}")
    
    # 安定性比較（低い方が良い）
    stabilities = [
        ('安定性80', 'stability_80', '±%.1f'),
        ('安定性98', 'stability_98', '±%.1f'),
        ('安定性110', 'stability_110', '±%.1f')
    ]
    
    stability_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in stabilities:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 < val2 else robot2['name']  # 低い方が勝ち
        stability_wins[winner] += 1
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10}")
    
    # 達成率比較
    achievements = [
        ('達成率80', 'achievement_80', '%.3f'),
        ('達成率98', 'achievement_98', '%.3f'),
        ('達成率110', 'achievement_110', '%.3f')
    ]
    
    achievement_wins = {robot1['name']: 0, robot2['name']: 0}
    
    for name, key, fmt in achievements:
        val1, val2 = robot1[key], robot2[key]
        winner = robot1['name'] if val1 > val2 else robot2['name']
        achievement_wins[winner] += 1
        print(f"{name:>15}: {fmt % val1:>15} {fmt % val2:>15} {winner:>10}")
    
    # 総合スコア
    total_winner = robot1['name'] if robot1['total_score'] > robot2['total_score'] else robot2['name']
    score_diff = abs(robot1['total_score'] - robot2['total_score'])
    
    print(f"{'総合スコア':>15}: {robot1['total_score']:>15.1f} {robot2['total_score']:>15.1f} {total_winner:>10}")
    
    # 総合判定
    print(f"\n" + "=" * 80)
    print("🎯 総合判定")
    print("=" * 80)
    
    total_wins = {robot1['name']: 0, robot2['name']: 0}
    total_wins[speed_wins[robot1['name']] > speed_wins[robot2['name']] and robot1['name'] or robot2['name']] += 1
    total_wins[stability_wins[robot1['name']] > stability_wins[robot2['name']] and robot1['name'] or robot2['name']] += 1
    total_wins[achievement_wins[robot1['name']] > achievement_wins[robot2['name']] and robot1['name'] or robot2['name']] += 1
    total_wins[total_winner] += 1
    
    overall_winner = robot1['name'] if total_wins[robot1['name']] > total_wins[robot2['name']] else robot2['name']
    
    print(f"� 優勝: {overall_winner}")
    print(f"   速度性能: {robot1['name']} {speed_wins[robot1['name']]}-{speed_wins[robot2['name']]} {robot2['name']}")
    print(f"   安定性: {robot1['name']} {stability_wins[robot1['name']]}-{stability_wins[robot2['name']]} {robot2['name']}")
    print(f"   達成率: {robot1['name']} {achievement_wins[robot1['name']]}-{achievement_wins[robot2['name']]} {robot2['name']}")
    print(f"   総合スコア差: {score_diff:.1f}ポイント")
    
    # 性能差評価
    if score_diff < 5:
        print("   評価: 性能は互角")
    elif score_diff < 15:
        print("   評価: 明確な差あり") 
    else:
        print("   評価: 大きな性能差")
    
    # おすすめ用途
    print(f"\n💡 おすすめ用途:")
    
    # 速度重視
    speed_leader = robot1['name'] if (robot1['speed_98'] + robot1['speed_110'])/2 > (robot2['speed_98'] + robot2['speed_110'])/2 else robot2['name']
    print(f"   高速走行: {speed_leader}")
    
    # 安定性重視
    stability_leader = robot1['name'] if (robot1['stability_98'] + robot1['stability_110'])/2 < (robot2['stability_98'] + robot2['stability_110'])/2 else robot2['name']
    print(f"   精密制御: {stability_leader}")
    
    # バランス
    print(f"   総合バランス: {overall_winner}")

def main():
    """メイン関数"""
    print("同時実行機体比較ツール")
    print("1: 機体性能テスト実行 (各Raspberry Piで実行)")
    print("2: 比較データ解析 (結果をコピペして比較)")
    
    choice = input("選択 (1 or 2): ").strip()
    
    if choice == "1":
        single_robot_performance_test()
    elif choice == "2":
        parse_comparison_data()
    else:
        print("1を選択して各Raspberry Piでテストを実行してください")
        print("結果をコピペして、2で比較分析できます")

if __name__ == "__main__":
    main()
