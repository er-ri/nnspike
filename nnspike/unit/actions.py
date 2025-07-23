import time

from nnspike.unit.etrobot import ETRobot



def _perform_action_chain(state, et: ETRobot, frame, action_chain):
    """
    汎用アクションチェーンを1フレームごとに進めるステートマシン。
    action_chain: [(left_speed, right_speed, duration_sec), ...]
    state: Noneまたはdict（index, 残り時間）
    戻り値: (新state, left_speed, right_speed, 完了フラグ)
    """
    if state is None:
        if not action_chain:
            return None, 0, 0, True
        idx = 0
        left, right, duration = action_chain[0]
        return {"idx": 0, "remain": duration}, left, right, False
    idx = state["idx"]
    remain = state["remain"]
    if idx >= len(action_chain):
        return None, 0, 0, True
    left, right, duration = action_chain[idx]
    # 1フレーム分進める（仮に1フレーム=0.05sとする）
    dt = 0.05
    remain -= dt
    if remain > 0:
        return {"idx": idx, "remain": remain}, left, right, False
    # 次のアクションへ
    idx += 1
    if idx >= len(action_chain):
        return None, 0, 0, True
    left, right, duration = action_chain[idx]
    return {"idx": idx, "remain": duration}, left, right, False



def avoid_obstacle(state, et: ETRobot, frame):
    """
    障害物回避アクションを1フレーム進める汎用step関数。
    """
    action_chain = [
        (40, 70, 0.8),  # 左旋回
        (80, 50, 1.3),  # 右旋回
    ]
    return _perform_action_chain(state, et, frame, action_chain)


def turn_right(state, et: ETRobot, frame):
    """
    右旋回アクションを1フレーム進める汎用step関数。
    """
    action_chain = [
        (60, 0, 1.0),  # 右旋回（左モーター前進、右モーター停止）
    ]
    return _perform_action_chain(state, et, frame, action_chain)


def turn_left(state, et: ETRobot, frame):
    """
    左旋回アクションを1フレーム進める汎用step関数。
    """
    action_chain = [
        (0, 60, 1.0),  # 左旋回（左モーター停止、右モーター前進）
    ]
    return _perform_action_chain(state, et, frame, action_chain)
