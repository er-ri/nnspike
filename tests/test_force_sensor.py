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
    # --- run_manual.pyと同一のフォースセンサー起動時チェック ---
    try:
        status_init = et.get_spike_status()
        force_val_init = getattr(status_init.sensors, "force", None)
        if force_val_init is None:
            time.sleep(1)
            status_init = et.get_spike_status()
            force_val_init = getattr(status_init.sensors, "force", None)
        if force_val_init is not None:
            print("Force sensor is active. You can press it anytime to switch edge-following mode.")
        else:
            print("Force sensor is NOT detected. Please check connection.")
    except Exception:
        print("Force sensor check failed. Please check hardware.")

    print("フォースセンサー稼働テスト開始。センサーを押すと値が変化します。Ctrl+Cで終了。")
    state_switched = False
    try:
        while True:
            status = et.get_spike_status()
            force_val = getattr(status.sensors, "force", None)
            if not state_switched and force_val is not None and force_val > 0:
                print("\nForce sensor pressed: Switched mode (edge-following or similar)")
                state_switched = True
            if force_val == 1:
                print("Force Sensor: PRESSED   ", end='\r')
            elif force_val == 0:
                print("Force Sensor: RELEASED  ", end='\r')
            else:
                print(f"Force Sensor: UNKNOWN ({force_val})", end='\r')
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nテスト終了。")
        et.stop()

if __name__ == "__main__":
    main()
