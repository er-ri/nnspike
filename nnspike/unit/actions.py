import time
from nnspike.unit.etrobot import ETRobot


def _perform_action_chain(action_chain: tuple[ETRobot, int, int, float]) -> None:
    """
    指定された速度と時間でETRobotのモーター制御アクションのシーケンスを実行します。

    チェーン内の各アクションは、左・右モーターの速度を指定時間維持する処理です。
    この関数はロボットの接続状態を常に監視し、切断された場合は即座に実行を停止します。

    引数:
        action_chain (tuple): 各要素が以下を含むタプル:
            - ETRobot: 制御対象のロボットインスタンス
            - int: 左モーター速度（-100〜100）
            - int: 右モーター速度（-100〜100）
            - float: この速度を維持する秒数

    戻り値:
        None: 実行中にロボットが切断された場合は即座に終了します
    """
    for action in action_chain:
        et, left_speed, right_speed, duration = action

        start_time = time.time()
        while time.time() - start_time < duration:
            # Check if the robot is still running
            if not et.is_running:
                print("ETRobot stopped, stopping action chain.")
                return

            # Set motor speeds
            if left_speed >= 0 and right_speed >= 0:
                et.set_motor_forward_speed(left_speed, right_speed)
            elif left_speed <= 0 and right_speed <= 0:
                et.set_motor_backward_speed(abs(left_speed), abs(right_speed))
            else:
                raise ValueError("Both speeds must be either positive or negative.")

            time.sleep(0.05)  # ロボットへの過負荷を避けるための短い遅延


def avoid_obstacle(et: ETRobot) -> None:
    """
    障害物を回避するための一連のアクションを実行します。

    引数:
        et (ETRobot): 制御対象のETRobotインスタンス。
    """
    action_chain = (
        (et, 40, 60, 1.3),  # 左モーター30、右モーター50で1.5秒間（左向き）
        (et, 80, 50, 1.3),  # 左モーター70、右モーター40で2.0秒間（左迂回）
        #(et, 0, 60, 0.6),   # 左モーター0、右モーター60で0.5秒間（右向き）
    )

    _perform_action_chain(action_chain)


def catch_bottle_blue(et: ETRobot) -> None:
    """
    青いボトルをキャッチするための一連のアクション（1回目）を実行します。

    引数:
        et (ETRobot): 制御対象のETRobotインスタンス。
    """
    action_chain = (
        (et, 30, 30, 2.0),  # 2秒間ボトルにまっすぐ接近（左30、右30）
        (et, 0, 60, 0.5),   # 60度左に回転（左0、右60）
    )

    _perform_action_chain(action_chain)


def release_bottle_blue(et: ETRobot) -> None:
    """
    青いボトルをリリースするための一連のアクションを実行します。

    引数:
        et (ETRobot): 制御対象のETRobotインスタンス。
    """
    action_chain = (
        (et, -30, -30, 2.0),  # 2秒間後退してボトルから離れる（左-30、右-30）
        (et, 60, 0, 1.5),     # 180度右に回転（左60、右0）
    )

    _perform_action_chain(action_chain)


