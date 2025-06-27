import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
ModeManagerとActionManagerの結合テスト（最新run_opencv.py設計に同期）
"""
import time
from run_opencv import ModeManager, ActionManager, Mode, ControlCalculator, BASE_POWER, CURVE_POWER, STRAIGHT_POWER, SENSITIVITY, CURVE_THRESHOLD_DEG, STRAIGHT_THRESHOLD_DEG, ROI_OPENCV, IMAGE_WIDTH
import math

class DummyET:
    def __init__(self):
        self.prev_left_power = None
        self.prev_right_power = None
        self.elapsed_ms = 0
    def update_elapsed_ms(self, elapsed_ms):
        self.elapsed_ms = elapsed_ms
    def brake(self):
        print("brake called")
    def set_motor_relative_position(self, left_position, right_position):
        print(f"set_motor_relative_position called: left_position={left_position}, right_position={right_position}")
    def set_motor_forward_power(self, left_power, right_power):
        if (left_power != self.prev_left_power) or (right_power != self.prev_right_power):
            print(f"set_motor_forward_power called: left_power={left_power}, right_power={right_power}, elapsed={self.elapsed_ms}ms")
        self.prev_left_power = left_power
        self.prev_right_power = right_power

def test_mode_action_integration():
    et = DummyET()
    calc = ControlCalculator(
        ROI_OPENCV,
        IMAGE_WIDTH,
        SENSITIVITY,
        BASE_POWER,
        CURVE_POWER,
        STRAIGHT_POWER,
        math.radians(CURVE_THRESHOLD_DEG),
        math.radians(STRAIGHT_THRESHOLD_DEG)
    )
    mode_manager = ModeManager()
    action = ActionManager(et)
    # テスト用PIDパラメータ再設定
    action.pid.Kp = 2.0
    action.pid.Ki = 0
    action.pid.Kd = 0.4
    action.pid.setpoint = 0
    action.pid.output_limits = (-0.25, 0.25)

    print("=== LINE_TRACE→DIST_STOP→OBSTACLE_AVOID→LINE_TRACEの一連遷移・action出力テスト（最新設計同期） ===")
    offset_pixels = 20  # テスト用固定値
    start_ms = int(time.time() * 1000)
    prev_mode = mode_manager.mode
    prev_left = None
    prev_right = None
    i = 0
    last_display_sec = -1
    avoid2trace_printed = False  # OBSTACLE_AVOID→LINE_TRACE遷移print済みフラグ
    for _ in range(500):  # 最大500サイクルで強制終了
        now_ms = int(time.time() * 1000)
        elapsed_ms = now_ms - start_ms
        # 時間でdistanceを切り替えて全モード遷移を必ず再現
        if elapsed_ms < 3000:
            action.distance = 100  # LINE_TRACE
        elif elapsed_ms < 6000:
            action.distance = 40   # DIST_STOP
        elif elapsed_ms < 10000:
            action.distance = 40   # OBSTACLE_AVOID
        else:
            action.distance = 100  # LINE_TRACE復帰
        # 最新設計に合わせてupdate_and_actで全て管理
        if mode_manager.mode == Mode.LINE_TRACE:
            mode_manager.update_and_act(
                action_manager=action,
                offset_pixels=offset_pixels,
                calc=calc
            )
        elif mode_manager.mode == Mode.DIST_STOP:
            mode_manager.update_and_act(
                action_manager=action,
                offset_pixels=offset_pixels,
                calc=calc
            )
        elif mode_manager.mode == Mode.OBSTACLE_AVOID:
            mode_manager.update_and_act(
                action_manager=action
            )
        else:
            mode_manager.update_and_act(
                action_manager=action
            )
        left = action.left_power
        right = action.right_power
        mode_changed = (mode_manager.mode != prev_mode)
        power_changed = (left != prev_left or right != prev_right)
        elapsed_sec = elapsed_ms // 1000
        # OBSTACLE_AVOID→LINE_TRACE遷移をmode変化で必ず検出
        if (prev_mode == Mode.OBSTACLE_AVOID and mode_manager.mode == Mode.LINE_TRACE and not avoid2trace_printed):
            print(f"[MODE遷移] OBSTACLE_AVOID → LINE_TRACE (step={i}, elapsed={elapsed_ms}ms, state={action.state}, finished={action.finished}, left={left}, right={right})")
            avoid2trace_printed = True
        # 1秒ごと、またはmode/power変化時のみ出力
        if mode_changed or power_changed:
            print(f"step={i}, elapsed={elapsed_ms}ms, [BEFORE] mode={prev_mode}, left_power={prev_left}, right_power={prev_right}")
            print(f"step={i}, elapsed={elapsed_ms}ms, [AFTER]  mode={mode_manager.mode}, left_power={left}, right_power={right}")
        elif elapsed_sec != last_display_sec or i == 0:
            if mode_manager.mode == Mode.OBSTACLE_AVOID:
                print(f"step={i}, elapsed={elapsed_ms}ms, mode={mode_manager.mode}, state={action.state}, left_power={left}, right_power={right}, finished={action.finished}")
            else:
                print(f"step={i}, elapsed={elapsed_ms}ms, mode={mode_manager.mode}, left_power={left}, right_power={right}")
            last_display_sec = elapsed_sec
        if elapsed_ms > 12000:
            print(f"step={i}, elapsed={elapsed_ms}ms, [TIMEOUT] mode={mode_manager.mode}, left_power={left}, right_power={right}")
            break
        prev_mode = mode_manager.mode
        prev_left = left
        prev_right = right
        time.sleep(0.03)
        i += 1
    if not avoid2trace_printed:
        print("[WARNING] OBSTACLE_AVOID→LINE_TRACE遷移がテスト中に一度も発生しませんでした！")

if __name__ == "__main__":
    test_mode_action_integration()
