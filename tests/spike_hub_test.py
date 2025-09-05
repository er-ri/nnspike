#!/usr/bin/env python3
"""
SPIKE Hub Minimal Test Program

このプログラムはSPIKEハブ上で動作する最小限のテストプログラムです。
バッテリー情報とセンサー情報を定期的に送信します。
"""

from pybricks.hubs import PrimeHub
from pybricks.parameters import Button, Color
from pybricks.tools import wait
import time

def main():
    """メイン関数"""
    hub = PrimeHub()
    
    print("SPIKE Hub Test - Battery & Sensors")
    hub.light.on(Color.GREEN)
    
    count = 0
    
    try:
        while True:
            count += 1
            
            # バッテリー情報表示
            battery_voltage = hub.battery.voltage()
            battery_current = hub.battery.current()
            
            print(f"[{count}] Battery: {battery_voltage}mV, Current: {battery_current}mA")
            
            # センターボタンが押されたら終了
            if Button.CENTER in hub.buttons.pressed():
                break
            
            # 1秒待機
            wait(1000)
            
    except KeyboardInterrupt:
        pass
    
    hub.light.on(Color.RED)
    print("Test completed")
    wait(1000)
    hub.light.off()

if __name__ == "__main__":
    main()
