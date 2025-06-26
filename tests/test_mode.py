"""
ModeManagerのロジック単体テスト
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from run_opencv import ModeManager, Mode


def test_mode_manager():
    mode_manager = ModeManager()
    print("=== 初期状態 ===")
    print(f"mode={mode_manager.mode}")

    print("=== LINE_TRACE→DIST_STOP遷移テスト (distance=40) ===")
    mode_manager.reset()
    print("update(distance=40)")
    mode_manager.update(40)  # 40 < 50
    print("  判定: distance=40 < OBSTACLE_DETECT_DISTANCE=50 → DIST_STOP")
    print(f"mode={mode_manager.mode}")

    print("=== DIST_STOP→OBSTACLE_AVOID遷移テスト (2秒経過) ===")
    mode_manager.reset()
    mode_manager.update(40)  # DIST_STOPへ
    print(f"start: mode={mode_manager.mode}")
    start_ms = int(time.time() * 1000)
    prev_mode = mode_manager.mode
    i = 0
    last_display_sec = -1
    while True:
        now_ms = int(time.time() * 1000)
        elapsed_ms = now_ms - start_ms
        elapsed_sec = elapsed_ms // 1000
        mode_manager.update(40)
        mode_changed = (mode_manager.mode != prev_mode)
        # modeまたは経過秒数が変化したときのみ表示
        if mode_changed and i > 0:
            print(f"step={i}, elapsed={elapsed_ms}ms, mode={prev_mode} (切替前)")
            print(f"step={i}, elapsed={elapsed_ms}ms, mode={mode_manager.mode}  判定: {elapsed_ms/1000:.2f}s経過 >= DIST_STOP_DURATION=2.0 → OBSTACLE_AVOID")
            break
        if elapsed_sec != last_display_sec or i == 0:
            print(f"step={i}, elapsed={elapsed_ms}ms, mode={mode_manager.mode}")
            last_display_sec = elapsed_sec
        prev_mode = mode_manager.mode
        time.sleep(0.03)
        i += 1
    print(f"end: mode={mode_manager.mode}")

    print("=== DIST_STOP→LINE_TRACE遷移テスト (distance=100) ===")
    mode_manager.reset()
    print("update(distance=40)")
    mode_manager.update(40)  # DIST_STOPへ
    print(f"mode={mode_manager.mode}")
    print("update(distance=100)")
    mode_manager.update(100)  # 100 > 50*(50/30)
    print(f"  判定: distance=100 >= OBSTACLE_DETECT_DISTANCE*50/30={50*50/30:.2f} → LINE_TRACE")
    print(f"mode={mode_manager.mode}")

    print("=== OBSTACLE_AVOID→(変化なし)テスト ===")
    mode_manager.mode = Mode.OBSTACLE_AVOID
    mode_manager.update(40)
    print(f"mode={mode_manager.mode}")

if __name__ == "__main__":
    test_mode_manager()
