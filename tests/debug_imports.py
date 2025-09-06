#!/usr/bin/env python3
"""
簡単なPIDテスト - デバッグ用
"""

import sys
import os
import time

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from nnspike.unit import ETRobot
    print("✅ ETRobot インポート成功")
except ImportError as e:
    print(f"❌ ETRobot インポートエラー: {e}")

try:
    from nnspike.utils import PIDController
    print("✅ PIDController インポート成功")
except ImportError as e:
    print(f"❌ PIDController インポートエラー: {e}")

def test_imports():
    """インポートテスト"""
    print("🔧 段階的PIDテスト - importテスト")
    
    try:
        et = ETRobot()
        print("✅ ETRobot 初期化成功")
        
        pid = PIDController(
            Kp=1.0,
            Ki=0,
            Kd=0.3,
            setpoint=0,
            output_limits=(-4, 4)
        )
        print("✅ PIDController 初期化成功")
        
        # 簡単なテスト
        correction = pid.update(0.1)
        print(f"✅ PID update成功: {correction}")
        
        et.stop()
        print("✅ 全てのテスト成功!")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
