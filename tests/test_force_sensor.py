#!/usr/bin/env python3
"""
フォースセンサー稼働確認用テストスクリプト
SPIKE Primeのフォースセンサーが正しく動作するかを確認します。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nnspike.unit import ETRobot
import time

def main():
    et = ETRobot()
    print("フォースセンサー稼働テスト開始。センサーを押すと値が変化します。Ctrl+Cで終了。")
    try:
        while True:
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            if force_val == 1:
                print("[押下] Force Sensor: 1 (押されています)   ", end='\r')
            elif force_val == 0:
                print("[未押下] Force Sensor: 0 (押されていません) ", end='\r')
            else:
                print(f"[不明] Force Sensor: {force_val}                ", end='\r')
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nテスト終了。")
        et.stop()

if __name__ == "__main__":
    main()
