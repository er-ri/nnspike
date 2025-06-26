"""
ActionManagerのロジック単体テスト
"""
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_opencv import ActionManager, PIDController, ControlCalculator, BASE_POWER, CURVE_POWER, STRAIGHT_POWER, SENSITIVITY, CURVE_THRESHOLD_DEG, STRAIGHT_THRESHOLD_DEG, ROI_OPENCV, IMAGE_WIDTH
import math

class DummyET:
    def brake(self):
        print("brake called")

def test_action_manager():
    et = DummyET()
    # run_opencv.pyと同じ初期化
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
    action = ActionManager(et)

    print("=== LINE_TRACE ===")
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=20,
        calc=calc,
        pid=pid
    )
    print(f"[offset_pixels=20] left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")
    print(f"  theta=offset_pixels/max_offset*sensitivity=20/320*1.0={20/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((pid_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=0) ===")
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=0,
        calc=calc,
        pid=pid
    )
    print(f"[offset_pixels=0] left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")
    print(f"  theta=0/320*1.0=0.0")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((pid_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=10) ===")
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=10,
        calc=calc,
        pid=pid
    )
    print(f"[offset_pixels=10] left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")
    print(f"  theta=10/320*1.0={10/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((pid_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=-10) ===")
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=-10,
        calc=calc,
        pid=pid
    )
    print(f"[offset_pixels=-10] left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")
    print(f"  theta=-10/320*1.0={-10/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((pid_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=100) ===")
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=100,
        calc=calc,
        pid=pid
    )
    print(f"[offset_pixels=100] left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")
    print(f"  theta=100/320*1.0={100/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((pid_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=-100) ===")
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=-100,
        calc=calc,
        pid=pid
    )
    print(f"[offset_pixels=-100] left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")
    print(f"  theta=-100/320*1.0={-100/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((pid_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=±100で左右パワーが正しく反転するかの検証: offset_pixels=100で左大/右小, offset_pixels=-100で左小/右大になること) ===")
    # 例2: offset_pixels=100 のときのパワー値を検証
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=100,
        calc=calc,
        pid=pid
    )
    print(f"offset_pixels=100: left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")

    # 例3: offset_pixels=-100 のときのパワー値を検証
    theta, pid_theta, power = action.do_line_trace(
        offset_pixels=-100,
        calc=calc,
        pid=pid
    )
    print(f"offset_pixels=-100: left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, current_power={power}")

    print("=== DIST_STOP ===")
    theta, pid_theta, power = action.do_dist_stop()
    print(f"left_power={action.left_power}, right_power={action.right_power}")

    print("=== OBSTACLE_AVOID ===")
    # 3ステップ分呼び出して状態遷移を確認
    for i in range(6):
        theta, pid_theta, power = action.do_obstacle_avoid()
        print(f"step={i}, state={action.state}, left_power={action.left_power}, right_power={action.right_power}, finished={action.finished}")
        time.sleep(0.1)

    print("=== SMART_CARRY_1 ===")
    if hasattr(action, 'do_smart_carry_1'):
        theta, pid_theta, power = action.do_smart_carry_1()
        print(f"left_power={action.left_power}, right_power={action.right_power}")
    else:
        print("do_smart_carry_1()未実装")

    print("=== 未定義モード ===")
    try:
        theta, pid_theta, power = action.do_unknown()
        print(f"left_power={action.left_power}, right_power={action.right_power}")
    except AttributeError:
        print("未定義モードはAttributeErrorとなる (OK)")

    # === LINE_TRACE (power>0になるoffset_pixelsを逆算テスト) ===
    # BASE_POWER, pid補正、パワー差分を考慮し、power>0となるoffset_pixelsを探索
    # for offset in range(-100, 101, 10):
    #     theta, pid_theta, power = action.do_line_trace(
    #         offset_pixels=offset,
    #         calc=calc,
    #         pid=pid
    #     )
    #     if action.left_power > 0 or action.right_power > 0:
    #         print(f"offset_pixels={offset}: left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, power={power}")
    #         break
    # else:
    #     print("power>0となるoffset_pixelsが見つかりませんでした")

    print("=== OBSTACLE_AVOID (state=3になるまでの経過時間・状態遷移テスト: 各state分岐の動作・パワー・ブレーキ等を網羅的に確認) ===")
    action = ActionManager(et)  # 状態リセット
    start_ms = int(time.time() * 1000)
    i = 0
    last_display_ms = 0
    prev_state = None
    while True:
        now = time.time()
        now_ms = int(now * 1000)
        elapsed_ms = now_ms - start_ms
        theta, pid_theta, current_power = action.do_obstacle_avoid()
        state_changed = (prev_state != action.state)
        # 各stateの分岐内容・アクションを明示的に表示
        def get_action_detail(state, left_power, right_power):
            if state == 0:
                return f"左回転1: left_power={left_power}, right_power={right_power}"
            elif state == 1:
                return f"右弧旋回: left_power={left_power}, right_power={right_power}"
            elif state == 2:
                return f"左回転2: left_power={left_power}, right_power={right_power}"
            elif state == 3:
                return f"停止: left_power={left_power}, right_power={right_power} (brake呼び出し済みのはず)"
            else:
                return f"未知: left_power={left_power}, right_power={right_power}"
        # state切り替わり時は前フレームも必ず表示
        printed_prev = False
        if state_changed and prev_state is not None:
            print(f"step={i}, elapsed={elapsed_ms}ms, state={prev_state}, {get_action_detail(prev_state, prev_left_power, prev_right_power)}, finished={action.finished}")
            printed_prev = True
        # action_sent==False（=アクション初期化直後）も必ずprint
        # 直前と全く同じ内容なら2回目のprintは省略
        current_detail = get_action_detail(action.state, action.left_power, action.right_power)
        prev_detail = get_action_detail(prev_state, prev_left_power, prev_right_power) if prev_state is not None else None
        is_duplicate = (
            state_changed and printed_prev and
            action.state == prev_state and
            current_detail == prev_detail and
            action.finished == action.finished
        )
        if (elapsed_ms - last_display_ms >= 1000 or i == 0 or (action.finished and action.state == 3) or state_changed or (hasattr(action, 'action_sent') and action.action_sent is False)) and not is_duplicate:
            print(f"step={i}, elapsed={elapsed_ms}ms, state={action.state}, {current_detail}, finished={action.finished}")
            last_display_ms = elapsed_ms
        prev_state = action.state
        prev_left_power = action.left_power
        prev_right_power = action.right_power
        if action.finished and action.state == 3:
            break
        time.sleep(0.03)  # 30msサイクル
        i += 1

    print("=== ModeManager.update_and_act のテスト ===")
    from run_opencv import ModeManager, Mode
    mode_manager = ModeManager()
    action = ActionManager(et)
    # 1. LINE_TRACE: distance=100, offset_pixels=20
    theta, pid_theta, power = mode_manager.update_and_act(
        distance=100,  # 障害物なし
        action_manager=action,
        offset_pixels=20,
        calc=calc,
        pid=pid
    )
    print(f"[LINE_TRACE] mode={mode_manager.mode}, left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, power={power}")
    # 2. DIST_STOP: distance=10（障害物検知）
    theta, pid_theta, power = mode_manager.update_and_act(
        distance=10,  # 障害物あり
        action_manager=action,
        offset_pixels=20,
        calc=calc,
        pid=pid
    )
    print(f"[DIST_STOP] mode={mode_manager.mode}, left_power={action.left_power}, right_power={action.right_power}, theta={theta}, pid_theta={pid_theta}, power={power}")
    # 3. OBSTACLE_AVOID: DIST_STOP経過後に強制遷移
    mode_manager.mode = Mode.OBSTACLE_AVOID
    action.reset()
    for i in range(5):
        theta, pid_theta, power = mode_manager.update_and_act(
            distance=10,
            action_manager=action
        )
        print(f"[OBSTACLE_AVOID] step={i}, mode={mode_manager.mode}, state={action.state}, left_power={action.left_power}, right_power={action.right_power}, finished={action.finished}")
        time.sleep(0.05)
    # 4. OBSTACLE_AVOID完了後のreset挙動
    if action.is_finished():
        mode_manager.reset()
        print(f"[RESET] mode={mode_manager.mode}, state={action.state}, finished={action.finished}")

# ユーザー調整パラメータはrun_opencv.pyからimportしたものをそのまま使っていて、
# テスト時も本番と同じパラメータで動作します。
if __name__ == "__main__":
    test_action_manager()
