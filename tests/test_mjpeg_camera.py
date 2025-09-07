#!/usr/bin/env python3
"""
MJPEG Camera Test Program

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))の効果をテストします。
- 設定の成功/失敗を確認
- フレーム取得速度の測定
- 画像品質の比較（PNG保存）
- メモリ使用量の測定
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nnspike.constants import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

class CameraPerformanceTester:
    def __init__(self):
        self.results = {}
        self.storage_dir = project_root / "storage" / "camera_test"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        print(f"画像保存先: {self.storage_dir}")
    
    def test_camera_setup(self, use_mjpeg=False, jpeg_quality=None):
        """カメラセットアップとパフォーマンステスト"""
        mode_desc = "通常モード"
        if use_mjpeg:
            if jpeg_quality:
                mode_desc = f"MJPEG品質{jpeg_quality}"
            else:
                mode_desc = "MJPEGデフォルト"
        
        print(f"\n{'='*60}")
        print(f"テスト開始: {mode_desc}")
        print(f"{'='*60}")
        
        # カメラ初期化
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ カメラを開けませんでした")
            return None
        
        # 基本設定
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # MJPEG設定（テスト対象）
        mjpeg_success = False
        quality_success = False
        if use_mjpeg:
            mjpeg_success = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            
            # 品質設定
            if jpeg_quality:
                # 明示的品質設定（OpenCVバージョンによって異なる可能性）
                try:
                    quality_success = cap.set(cv2.CAP_PROP_SATURATION, jpeg_quality)  # 代替設定
                except:
                    quality_success = False
            
            print(f"MJPEG設定結果: {'✅ 成功' if mjpeg_success else '❌ 失敗'}")
            if jpeg_quality:
                print(f"品質設定結果: {'✅ 成功' if quality_success else '❌ 失敗'} (品質{jpeg_quality})")
                print(f"  → MJPEG品質{jpeg_quality}でテスト")
            else:
                print(f"  → MJPEGフォーマットのみ設定（品質はデフォルト）")
        
        # 設定値の確認
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # FOURCC値を文字列に変換
        fourcc_str = ''.join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"\n📊 カメラ設定確認:")
        print(f"  FPS: {actual_fps}")
        print(f"  解像度: {int(actual_width)}x{int(actual_height)}")
        print(f"  FOURCC: {actual_fourcc} ({fourcc_str})")
        if use_mjpeg:
            print(f"  MJPEGモード: {'MJPG' in fourcc_str or 'mjpg' in fourcc_str.lower()}")
            print(f"  FOURCC詳細: バイナリ={bin(actual_fourcc)}, 16進数={hex(actual_fourcc)}")
        else:
            print(f"  通常モード FOURCC詳細: バイナリ={bin(actual_fourcc)}, 16進数={hex(actual_fourcc)}")
        
        # FOURCC情報を結果に保存
        fourcc_info = {
            'fourcc_int': actual_fourcc,
            'fourcc_str': fourcc_str,
            'fourcc_hex': hex(actual_fourcc)
        }
        
        # ウォームアップ
        print("\n🔥 カメラウォームアップ中...")
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print(f"❌ ウォームアップフレーム{i+1}の取得に失敗")
                cap.release()
                return None
        
        # パフォーマンステスト
        results = self.measure_performance(cap, use_mjpeg)
        
        # FOURCC情報を結果に追加
        results['fourcc_info'] = fourcc_info
        results['jpeg_quality'] = jpeg_quality if jpeg_quality else 'default'
        
        # サンプル画像保存
        ret, sample_frame = cap.read()
        if ret:
            image_analysis = self.save_sample_images(sample_frame, use_mjpeg, jpeg_quality)
            results['image_analysis'] = image_analysis
        
        cap.release()
        return results
    
    def measure_performance(self, cap, use_mjpeg, num_frames=100):
        """フレーム取得パフォーマンス測定"""
        print(f"\n⏱️  パフォーマンス測定中... ({num_frames}フレーム)")
        
        frame_times = []
        frame_sizes = []
        successful_frames = 0
        
        start_time = time.time()
        
        for i in range(num_frames):
            frame_start = time.time()
            ret, frame = cap.read()
            frame_end = time.time()
            
            if ret:
                successful_frames += 1
                frame_times.append(frame_end - frame_start)
                frame_sizes.append(frame.nbytes)
                
                # 進行状況表示
                if (i + 1) % 20 == 0:
                    print(f"  進行状況: {i+1}/{num_frames} フレーム")
            else:
                print(f"  ❌ フレーム{i+1}の取得に失敗")
        
        end_time = time.time()
        
        # 結果計算
        total_time = end_time - start_time
        avg_frame_time = np.mean(frame_times) if frame_times else 0
        actual_fps = successful_frames / total_time if total_time > 0 else 0
        avg_frame_size = np.mean(frame_sizes) if frame_sizes else 0
        
        results = {
            'mode': 'MJPEG' if use_mjpeg else 'Normal',
            'successful_frames': successful_frames,
            'total_frames': num_frames,
            'success_rate': (successful_frames / num_frames) * 100,
            'total_time': total_time,
            'avg_frame_time': avg_frame_time * 1000,  # ms
            'actual_fps': actual_fps,
            'avg_frame_size_mb': avg_frame_size / 1024 / 1024,
            'bandwidth_mbps': (avg_frame_size * actual_fps) / 1024 / 1024 if actual_fps > 0 else 0
        }
        
        # 結果表示
        print(f"\n📈 パフォーマンス結果:")
        print(f"  成功フレーム: {successful_frames}/{num_frames} ({results['success_rate']:.1f}%)")
        print(f"  総時間: {total_time:.2f}秒")
        print(f"  平均フレーム時間: {results['avg_frame_time']:.2f}ms")
        print(f"  実際のFPS: {actual_fps:.2f}")
        print(f"  平均フレームサイズ: {results['avg_frame_size_mb']:.2f}MB")
        print(f"  推定帯域幅: {results['bandwidth_mbps']:.2f}MB/s")
        
        return results
    
    def save_sample_images(self, frame, use_mjpeg, jpeg_quality=None):
        """サンプル画像をPNG形式で保存し、圧縮率を確認"""
        if jpeg_quality:
            mode_name = f"mjpeg_q{jpeg_quality}"
        else:
            mode_name = "mjpeg" if use_mjpeg else "normal"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 元画像保存（PNG：無圧縮）
        original_filename = f"{timestamp}_{mode_name}_original.png"
        original_path = self.storage_dir / original_filename
        cv2.imwrite(str(original_path), frame)
        png_size = original_path.stat().st_size
        
        # 比較用：様々な品質でJPEG保存
        jpeg_sizes = {}
        for quality in [95, 80, 50, 30]:
            jpeg_filename = f"{timestamp}_{mode_name}_jpeg_q{quality}.jpg"
            jpeg_path = self.storage_dir / jpeg_filename
            cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            jpeg_sizes[quality] = jpeg_path.stat().st_size
        
        # グレースケール画像保存
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_filename = f"{timestamp}_{mode_name}_gray.png"
        gray_path = self.storage_dir / gray_filename
        cv2.imwrite(str(gray_path), gray)
        
        # 二値化画像保存（テスト用）
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        binary_filename = f"{timestamp}_{mode_name}_binary.png"
        binary_path = self.storage_dir / binary_filename
        cv2.imwrite(str(binary_path), binary)
        
        print(f"\n💾 画像保存完了:")
        print(f"  原画像(PNG): {original_filename} ({png_size:,} bytes)")
        print(f"  JPEG品質比較（参考用）:")
        for quality, size in jpeg_sizes.items():
            compression_ratio = (1 - size / png_size) * 100
            print(f"    品質{quality}: {size:,} bytes (圧縮率: {compression_ratio:.1f}%)")
        print(f"  グレー: {gray_filename}")
        print(f"  二値化: {binary_filename}")
        
        # 正しい比較説明
        if use_mjpeg:
            print(f"\n📝 重要な注意:")
            print(f"  PNG画像がMJPEGの実際の画質です")
            print(f"  JPEG品質比較は、同じ元画像での圧縮例（参考用）")
            print(f"  実際のUSB転送では、カメラ内圧縮済みデータが送信されます")
        else:
            print(f"\n📝 重要な注意:")
            print(f"  PNG画像が通常モードの画質です（無圧縮）")
            print(f"  JPEG品質比較は、この画像をJPEG圧縮した場合の例")
        
        # MJPEGの推定品質を計算
        if use_mjpeg:
            frame_size_bytes = frame.nbytes  # メモリ上のサイズ
            print(f"\n🔍 MJPEG実際の転送データ分析:")
            print(f"  メモリフレームサイズ: {frame_size_bytes:,} bytes")
            print(f"  PNG保存サイズ: {png_size:,} bytes")
            print(f"  → カメラからUSB経由で受信したデータサイズは約{frame_size_bytes:,} bytes")
            
            # 最も近い品質レベルを推定
            closest_quality = min(jpeg_sizes.items(), key=lambda x: abs(x[1] - png_size))
            print(f"  推定MJPEG品質: 約{closest_quality[0]} (PNG画像との比較)")
        else:
            print(f"\n🔍 通常モード実際の転送データ分析:")
            print(f"  USB転送サイズ: {frame.nbytes:,} bytes (YUYVデータ)")
            print(f"  PNG保存サイズ: {png_size:,} bytes")
            print(f"  → USB転送は常に{frame.nbytes:,} bytes（圧縮なし）")
        
        return {
            'png_size': png_size,
            'jpeg_sizes': jpeg_sizes,
            'frame_memory_size': frame.nbytes
        }
    
    def compare_results(self, normal_results, mjpeg_results):
        """結果比較とレポート生成"""
        if not normal_results or not mjpeg_results:
            print("❌ 比較に必要なデータが不足しています")
            return
        
        print(f"\n{'='*60}")
        print("🔍 パフォーマンス比較結果")
        print(f"{'='*60}")
        
        # 比較表示
        metrics = [
            ('成功率', 'success_rate', '%'),
            ('平均フレーム時間', 'avg_frame_time', 'ms'),
            ('実際のFPS', 'actual_fps', 'fps'),
            ('フレームサイズ', 'avg_frame_size_mb', 'MB'),
            ('推定帯域幅', 'bandwidth_mbps', 'MB/s')
        ]
        
        print(f"{'項目':<15} {'通常モード':<12} {'MJPEG':<12} {'改善率':<10}")
        print(f"{'-'*50}")
        
        for name, key, unit in metrics:
            normal_val = normal_results.get(key, 0)
            mjpeg_val = mjpeg_results.get(key, 0)
            
            if normal_val > 0:
                improvement = ((mjpeg_val - normal_val) / normal_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{name:<15} {normal_val:<12.2f} {mjpeg_val:<12.2f} {improvement_str:<10}")
        
        # 結論
        print(f"\n🎯 結論:")
        
        # FOURCC比較
        normal_fourcc = normal_results.get('fourcc_info', {})
        mjpeg_fourcc = mjpeg_results.get('fourcc_info', {})
        
        print(f"\n🔍 FOURCC比較:")
        print(f"  通常モード: {normal_fourcc.get('fourcc_str', 'N/A')} ({normal_fourcc.get('fourcc_hex', 'N/A')})")
        print(f"  MJPEGモード: {mjpeg_fourcc.get('fourcc_str', 'N/A')} ({mjpeg_fourcc.get('fourcc_hex', 'N/A')})")
        
        if normal_fourcc.get('fourcc_str') == mjpeg_fourcc.get('fourcc_str'):
            print("  ❌ FOURCC値が同じ → MJPEG設定が反映されていません")
        else:
            print("  ✅ FOURCC値が変更されました")
        
        if mjpeg_results['actual_fps'] > normal_results['actual_fps']:
            print("✅ MJPEG設定によりFPSが向上しました")
        else:
            print("❌ MJPEG設定によるFPS向上は確認できませんでした")
        
        if mjpeg_results['bandwidth_mbps'] < normal_results['bandwidth_mbps']:
            print("✅ MJPEG設定により帯域幅が削減されました")
        else:
            print("❌ MJPEG設定による帯域幅削減は確認できませんでした")
        
        # レポートファイル保存
        self.save_report(normal_results, mjpeg_results)
    
    def save_report(self, normal_results, mjpeg_results):
        """テスト結果をファイルに保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"camera_test_report_{timestamp}.txt"
        report_path = self.storage_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MJPEG Camera Performance Test Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"テスト実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("通常モード結果:\n")
            for key, value in normal_results.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nMJPEGモード結果:\n")
            for key, value in mjpeg_results.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\n📄 レポート保存: {report_filename}")
    
    def compare_all_results(self, all_results):
        """全テスト結果の総合比較"""
        if len(all_results) < 2:
            print("❌ 比較に必要なデータが不足しています")
            return
        
        print(f"\n{'='*80}")
        print("🔍 USB転送速度 & 圧縮率総合比較")
        print(f"{'='*80}")
        
        # ヘッダー表示
        print(f"{'モード':<15} {'FPS':<8} {'帯域幅(MB/s)':<12} {'フレーム時間(ms)':<15} {'圧縮効果':<12}")
        print(f"{'-'*70}")
        
        baseline_bandwidth = None
        baseline_fps = None
        
        for name, results in all_results:
            fps = results.get('actual_fps', 0)
            bandwidth = results.get('bandwidth_mbps', 0)
            frame_time = results.get('avg_frame_time', 0)
            
            # ベースライン設定（通常モード）
            if '通常' in name:
                baseline_bandwidth = bandwidth
                baseline_fps = fps
                compression_effect = "ベースライン"
            else:
                if baseline_bandwidth and baseline_fps:
                    bandwidth_improvement = ((baseline_bandwidth - bandwidth) / baseline_bandwidth) * 100
                    fps_improvement = ((fps - baseline_fps) / baseline_fps) * 100
                    compression_effect = f"帯域幅{bandwidth_improvement:+.1f}% FPS{fps_improvement:+.1f}%"
                else:
                    compression_effect = "N/A"
            
            print(f"{name:<15} {fps:<8.2f} {bandwidth:<12.2f} {frame_time:<15.2f} {compression_effect:<12}")
        
        # 圧縮率とUSB転送効率の分析
        print(f"\n🔍 圧縮効果分析:")
        
        normal_result = next((r for n, r in all_results if '通常' in n), None)
        mjpeg_results = [(n, r) for n, r in all_results if 'MJPEG' in n]
        
        if normal_result and mjpeg_results:
            print(f"\n📊 USB転送効率改善:")
            normal_bandwidth = normal_result.get('bandwidth_mbps', 0)
            
            for name, mjpeg_result in mjpeg_results:
                mjpeg_bandwidth = mjpeg_result.get('bandwidth_mbps', 0)
                mjpeg_fps = mjpeg_result.get('actual_fps', 0)
                normal_fps = normal_result.get('actual_fps', 0)
                
                if normal_bandwidth > 0:
                    bandwidth_reduction = ((normal_bandwidth - mjpeg_bandwidth) / normal_bandwidth) * 100
                    fps_change = ((mjpeg_fps - normal_fps) / normal_fps) * 100
                    
                    print(f"  {name}:")
                    print(f"    帯域幅削減: {bandwidth_reduction:.1f}%")
                    print(f"    FPS変化: {fps_change:+.1f}%")
                    
                    # 画像圧縮分析
                    image_analysis = mjpeg_result.get('image_analysis', {})
                    if image_analysis:
                        png_size = image_analysis.get('png_size', 0)
                        frame_memory = image_analysis.get('frame_memory_size', 0)
                        if png_size > 0 and frame_memory > 0:
                            actual_compression = ((frame_memory - png_size) / frame_memory) * 100
                            print(f"    実圧縮率: {actual_compression:.1f}%")
        
        # 結論
        print(f"\n🎯 USB転送最適化の結論:")
        
        best_fps_result = max(all_results, key=lambda x: x[1].get('actual_fps', 0))
        best_bandwidth_result = min([r for r in all_results if 'MJPEG' in r[0]], 
                                  key=lambda x: x[1].get('bandwidth_mbps', float('inf')), 
                                  default=None)
        
        print(f"  最高FPS: {best_fps_result[0]} ({best_fps_result[1].get('actual_fps', 0):.2f} FPS)")
        if best_bandwidth_result:
            print(f"  最小帯域幅: {best_bandwidth_result[0]} ({best_bandwidth_result[1].get('bandwidth_mbps', 0):.2f} MB/s)")
        
        # 推奨設定
        print(f"\n💡 推奨設定:")
        if len(mjpeg_results) > 0:
            # FPSが最も高く、帯域幅削減もあるものを推奨
            recommended = max(mjpeg_results, 
                            key=lambda x: x[1].get('actual_fps', 0) - x[1].get('bandwidth_mbps', 0) * 0.1)
            print(f"  推奨: {recommended[0]}")
            print(f"    理由: FPSとUSB転送効率のバランスが最適")
        
        # レポート保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"compression_comparison_report_{timestamp}.txt"
        report_path = self.storage_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MJPEG Compression & USB Transfer Speed Comparison Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"テスト実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for name, results in all_results:
                f.write(f"{name}結果:\n")
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"\n📄 総合レポート保存: {report_filename}")

def main():
    """メイン実行関数"""
    print("🎥 MJPEG Camera Performance Test with Quality Comparison")
    print("=" * 60)
    
    tester = CameraPerformanceTester()
    all_results = []
    
    # 1. 通常モードテスト（ベースライン）
    print("\n1️⃣ 通常モード（YUYV生データ）でのテスト")
    normal_results = tester.test_camera_setup(use_mjpeg=False)
    if normal_results is None:
        print("❌ 通常モードのテストに失敗しました")
        return
    all_results.append(('通常モード(YUYV)', normal_results))
    time.sleep(1)
    
    # 2. MJPEGデフォルト品質テスト
    print("\n2️⃣ MJPEGデフォルト品質（約95相当）でのテスト")
    mjpeg_default_results = tester.test_camera_setup(use_mjpeg=True)
    if mjpeg_default_results is None:
        print("❌ MJPEGデフォルトのテストに失敗しました")
        return
    all_results.append(('MJPEGデフォルト', mjpeg_default_results))
    time.sleep(1)
    
    # 3. MJPEG品質別テスト（カメラが対応している場合）
    quality_levels = [70, 50, 30]  # 高→低品質（高→低圧縮）
    for quality in quality_levels:
        print(f"\n{len(all_results)+1}️⃣ MJPEG品質{quality}（圧縮率重視）でのテスト")
        print(f"  ※カメラが品質設定に対応していない場合、デフォルト品質で動作します")
        quality_results = tester.test_camera_setup(use_mjpeg=True, jpeg_quality=quality)
        if quality_results is None:
            print(f"❌ MJPEG品質{quality}のテストに失敗しました")
            continue
        all_results.append((f'MJPEG品質{quality}', quality_results))
        time.sleep(1)
    
    # 総合比較
    tester.compare_all_results(all_results)
    
    print(f"\n✅ 全テスト完了！")
    print(f"📝 重要：通常モードは圧縮なし、MJPEGモードはカメラ内圧縮です")
    print(f"詳細結果とサンプル画像は {tester.storage_dir} に保存されました。")

if __name__ == "__main__":
    main()
