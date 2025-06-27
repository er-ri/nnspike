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

def test_action_manager():
    et = DummyET()
    # ActionManager生成後にPIDパラメータをテスト用に上書き
    action = ActionManager(et)
    action.pid.Kp = 2.0
    action.pid.Ki = 0
    action.pid.Kd = 0.4
    action.pid.setpoint = 0
    action.pid.output_limits = (-0.25, 0.25)
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

    print("=== LINE_TRACE ===")
    action.do_line_trace(
        offset_pixels=20,
        calc=calc
    )
    print(f"[offset_pixels=20] left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")
    print(f"  theta=offset_pixels/max_offset*sensitivity=20/320*1.0={20/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={action.current_power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((action.pid_corrected_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=0) ===")
    action.do_line_trace(
        offset_pixels=0,
        calc=calc
    )
    print(f"[offset_pixels=0] left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")
    print(f"  theta=0/320*1.0=0.0")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={action.current_power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((action.pid_corrected_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=10) ===")
    action.do_line_trace(
        offset_pixels=10,
        calc=calc
    )
    print(f"[offset_pixels=10] left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")
    print(f"  theta=10/320*1.0={10/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={action.current_power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((action.pid_corrected_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=-10) ===")
    action.do_line_trace(
        offset_pixels=-10,
        calc=calc
    )
    print(f"[offset_pixels=-10] left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")
    print(f"  theta=-10/320*1.0={-10/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={action.current_power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((action.pid_corrected_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=100) ===")
    action.do_line_trace(
        offset_pixels=100,
        calc=calc
    )
    print(f"[offset_pixels=100] left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")
    print(f"  theta=100/320*1.0={100/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={action.current_power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((action.pid_corrected_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=-100) ===")
    action.do_line_trace(
        offset_pixels=-100,
        calc=calc
    )
    print(f"[offset_pixels=-100] left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")
    print(f"  theta=-100/320*1.0={-100/320*1.0}")
    print(f"  current_power=calc_adaptive_speed(abs(theta))={action.current_power}")
    print(f"  power_adjustment=int((pid_theta/max_theta)*MAX_POWER_DIFF)={int((action.pid_corrected_theta/math.radians(50))*40)}")

    print("=== LINE_TRACE (offset_pixels=±100で左右パワーが正しく反転するかの検証: offset_pixels=100で左大/右小, offset_pixels=-100で左小/右大になること) ===")
    # 例2: offset_pixels=100 のときのパワー値を検証
    action.do_line_trace(
        offset_pixels=100,
        calc=calc
    )
    print(f"offset_pixels=100: left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")

    # 例3: offset_pixels=-100 のときのパワー値を検証
    action.do_line_trace(
        offset_pixels=-100,
        calc=calc
    )
    print(f"offset_pixels=-100: left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, current_power={action.current_power}")

    print("=== DIST_STOP ===")
    action.do_dist_stop()
    print(f"left_power={action.left_power}, right_power={action.right_power}")

    print("=== OBSTACLE_AVOID ===")
    # 3ステップ分呼び出して状態遷移を確認
    for i in range(6):
        action.do_obstacle_avoid()
        print(f"step={i}, state={action.state}, left_power={action.left_power}, right_power={action.right_power}, finished={action.finished}")
        time.sleep(0.1)

    print("=== SMART_CARRY_1 ===")
    if hasattr(action, 'do_smart_carry_1'):
        action.do_smart_carry_1()
        print(f"left_power={action.left_power}, right_power={action.right_power}")
    else:
        print("do_smart_carry_1()未実装")

    print("=== 未定義モード ===")
    try:
        action.do_unknown()
        print(f"left_power={action.left_power}, right_power={action.right_power}")
    except AttributeError:
        print("未定義モードはAttributeErrorとなる (OK)")

    print("=== OBSTACLE_AVOID (state=3になるまでの経過時間・状態遷移テスト: 各state分岐の動作・パワー・ブレーキ等を網羅的に確認) ===")
    action = ActionManager(et)  # 状態リセット
    start_ms = int(time.time() * 1000)
    i = 0
    last_display_ms = 0
    prev_state = None
    prev_left_power = 0
    prev_right_power = 0
    while True:
        now = time.time()
        now_ms = int(now * 1000)
        elapsed_ms = now_ms - start_ms
        et.update_elapsed_ms(elapsed_ms)  # DummyETに経過msを渡す
        action.do_obstacle_avoid()
        state_changed = (prev_state != action.state)
        power_changed = (action.left_power != prev_left_power or action.right_power != prev_right_power)
        # OBSTACLE_AVOIDテスト（action単体）ではmode printを省略
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
        # 1秒ごと、state変化、power変化、初回、終了時は必ず表示
        if (
            elapsed_ms - last_display_ms >= 1000 or
            i == 0 or
            (action.finished and action.state == 3) or
            state_changed or
            power_changed
        ):
            print(f"step={i}, elapsed={elapsed_ms}ms, state={action.state}, {get_action_detail(action.state, action.left_power, action.right_power)}, finished={action.finished}")
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
    # テスト用PIDパラメータ再設定
    action.pid.Kp = 2.0
    action.pid.Ki = 0
    action.pid.Kd = 0.4
    action.pid.setpoint = 0
    action.pid.output_limits = (-0.25, 0.25)
    # 1. LINE_TRACE: distance=100, offset_pixels=20
    action.distance = 100  # 障害物なし
    mode_manager.update_and_act(
        action_manager=action,
        offset_pixels=20,
        calc=calc
    )
    print(f"[LINE_TRACE] mode={mode_manager.mode}, left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, power={action.current_power}")
    # 2. DIST_STOP: distance=10（障害物検知）
    action.distance = 10  # 障害物あり
    mode_manager.update_and_act(
        action_manager=action,
        offset_pixels=20,
        calc=calc
    )
    print(f"[DIST_STOP] mode={mode_manager.mode}, left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, power={action.current_power}")
    # 3. OBSTACLE_AVOID: DIST_STOP経過後に強制遷移
    mode_manager.mode = Mode.OBSTACLE_AVOID
    action.reset()
    prev_mode = mode_manager.mode
    for i in range(20):  # ループ回数を増やしてLINE_TRACE復帰まで観察
        action.distance = 10
        mode_manager.update_and_act(
            action_manager=action
        )
        print(f"[OBSTACLE_AVOID] step={i}, mode={mode_manager.mode}, state={action.state}, left_power={action.left_power}, right_power={action.right_power}, finished={action.finished}")
        if mode_manager.mode != prev_mode:
            print(f"[MODE遷移] {prev_mode} → {mode_manager.mode} (step={i})")
            prev_mode = mode_manager.mode
        # OBSTACLE_AVOIDが終了したらmodeをLINE_TRACEに切り替える
        if mode_manager.mode == Mode.OBSTACLE_AVOID and action.is_finished():
            mode_manager.mode = Mode.LINE_TRACE
            print(f"[MODE遷移] OBSTACLE_AVOID → LINE_TRACE (step={i})")
        time.sleep(0.05)
    # 4. OBSTACLE_AVOID完了後のreset挙動
    if action.is_finished():
        mode_manager.reset()
        print(f"[RESET] mode={mode_manager.mode}, state={action.state}, finished={action.finished}")

    # OBSTACLE_AVOID完了後のLINE_TRACE復帰テスト
    if action.finished and action.state == 3:
        print("=== OBSTACLE_AVOID完了後、LINE_TRACEへ1秒間復帰テスト ===")
        mode_manager.reset()
        action.reset()
        mode_manager.mode = Mode.LINE_TRACE
        for j in range(33):  # 約1秒間（30ms*33≒1s）
            action.distance = 100
            mode_manager.update_and_act(
                action_manager=action,
                offset_pixels=10,
                calc=calc
            )
            print(f"mode={mode_manager.mode}")
            print(f"[LINE_TRACE復帰] step={j}, left_power={action.left_power}, right_power={action.right_power}, theta={action.theta}, pid_theta={action.pid_corrected_theta}, power={action.current_power}")
            time.sleep(0.03)

# ユーザー調整パラメータはrun_opencv.pyからimportしたものをそのまま使っていて、
# テスト時も本番と同じパラメータで動作します。
if __name__ == "__main__":
    test_action_manager()
