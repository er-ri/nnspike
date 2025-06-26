import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
ModeManagerとActionManagerの結合テスト
"""
import time
from run_opencv import ModeManager, ActionManager, Mode, PIDController, ControlCalculator, BASE_POWER, CURVE_POWER, STRAIGHT_POWER, SENSITIVITY, CURVE_THRESHOLD_DEG, STRAIGHT_THRESHOLD_DEG, ROI_OPENCV, IMAGE_WIDTH
import math

class DummyET:
    def brake(self):
        print("brake called")

def test_mode_action_integration():
    # --- run_opencv.pyのmainと同じ初期化 ---
    et = DummyET()
    pid = PIDController(
        Kp=2.0,
        Ki=0,
        Kd=0.4,
        setpoint=0,
        output_limits=(-0.25, 0.25),
    )
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

    print("=== LINE_TRACE→DIST_STOP→OBSTACLE_AVOID→LINE_TRACEの一連遷移・action出力テスト ===")
    offset_pixels = 20  # テスト用固定値
    start_ms = int(time.time() * 1000)
    prev_mode = mode_manager.mode
    prev_left = None
    prev_right = None
    i = 0
    last_display_sec = -1
    entered_obstacle_avoid = False  # OBSTACLE_AVOID突入フラグ
    stuck_counter = 0  # OBSTACLE_AVOIDで進まない場合のカウンタ
    while True:
        now_ms = int(time.time() * 1000)
        elapsed_ms = now_ms - start_ms
        # 時間でdistanceを切り替えて全モード遷移を必ず再現
        if elapsed_ms < 3000:
            distance = 100  # LINE_TRACE
        elif elapsed_ms < 6000:
            distance = 40   # DIST_STOP
        elif elapsed_ms < 10000:
            distance = 40   # OBSTACLE_AVOID
        else:
            distance = 100  # LINE_TRACE復帰
        mode_manager.update(distance)

        # OBSTACLE_AVOID突入時はactionの状態をリセットし2回呼ぶ（デバッグ用）
        if mode_manager.mode == Mode.OBSTACLE_AVOID and not entered_obstacle_avoid:
            action.reset()
            for _ in range(2):
                theta, pid_theta, power = action.do_obstacle_avoid()
            entered_obstacle_avoid = True
        else:
            if mode_manager.mode == Mode.LINE_TRACE:
                theta, pid_theta, power = action.do_line_trace(
                    offset_pixels=offset_pixels,
                    calc=calc,
                    pid=pid
                )
            elif mode_manager.mode == Mode.DIST_STOP:
                theta, pid_theta, power = action.do_dist_stop()
            elif mode_manager.mode == Mode.OBSTACLE_AVOID:
                theta, pid_theta, power = action.do_obstacle_avoid()
            else:
                theta, pid_theta, power = 0, 0, 0  # 未定義モードは何もしない

        left = action.left_power
        right = action.right_power
        mode_changed = (mode_manager.mode != prev_mode)
        power_changed = (left != prev_left or right != prev_right)
        elapsed_sec = elapsed_ms // 1000
        # 1秒ごと、またはmode/power変化時のみ出力
        if mode_changed or power_changed:
            print(f"step={i}, elapsed={elapsed_ms}ms, [BEFORE] mode={prev_mode}, left_power={prev_left}, right_power={prev_right}")
            print(f"step={i}, elapsed={elapsed_ms}ms, [AFTER]  mode={mode_manager.mode}, left_power={left}, right_power={right}")
        elif elapsed_sec != last_display_sec or i == 0:
            if mode_manager.mode == Mode.OBSTACLE_AVOID:
                print(f"step={i}, elapsed={elapsed_ms}ms, mode={mode_manager.mode}, state={action.state}, left_power={left}, right_power={right}, finished={action.finished}, action_sent={action.action_sent}, start_time={action.start_time}")
            else:
                print(f"step={i}, elapsed={elapsed_ms}ms, mode={mode_manager.mode}, left_power={left}, right_power={right}")
            last_display_sec = elapsed_sec
        # OBSTACLE_AVOIDで進まない場合の即時デバッグ出力
        if mode_manager.mode == Mode.OBSTACLE_AVOID and action.state == 0 and not action.action_sent and left == 0 and right == 0:
            stuck_counter += 1
            if stuck_counter == 1 or stuck_counter % 10 == 0:
                print(f"[DEBUG] OBSTACLE_AVOID stuck: state=0, action_sent=False, left=0, right=0, finished={action.finished}, start_time={action.start_time}")
        else:
            stuck_counter = 0
        # OBSTACLE_AVOID完了時はmode_manager.reset()のみ
        if mode_manager.mode == Mode.OBSTACLE_AVOID and action.is_finished():
            print(f"step={i}, elapsed={elapsed_ms}ms, [EXIT] OBSTACLE_AVOID完了→LINE_TRACEリセット")
            mode_manager.reset()
            action.reset()
            entered_obstacle_avoid = False
        if elapsed_ms > 12000:
            print(f"step={i}, elapsed={elapsed_ms}ms, [TIMEOUT] mode={mode_manager.mode}, left_power={left}, right_power={right}")
            break
        prev_mode = mode_manager.mode
        prev_left = left
        prev_right = right
        time.sleep(0.03)
        i += 1

if __name__ == "__main__":
    test_mode_action_integration()
