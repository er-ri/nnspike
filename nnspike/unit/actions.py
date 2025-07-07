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
            # ロボットがまだ接続されているか確認
            if not et.is_connected():
                print("ETRobotが切断されました。アクションチェーンを停止します。")
                return

            # モーター速度を設定
            if left_speed >= 0 and right_speed >= 0:
                et.set_motor_forward_speed(left_speed, right_speed)
            elif left_speed < 0 and right_speed < 0:
                et.set_motor_backward_speed(abs(left_speed), abs(right_speed))
            else:
                # 異なる方向のモーター（左右非対称）の場合
                if left_speed >= 0:
                    et.set_motor_forward_speed(left_speed, 0)
                else:
                    et.set_motor_backward_speed(abs(left_speed), 0)
                    
                if right_speed >= 0:
                    et.set_motor_forward_speed(0, right_speed)
                else:
                    et.set_motor_backward_speed(0, abs(right_speed))

            time.sleep(0.05)  # ロボットへの過負荷を避けるための短い遅延


def avoid_obstacle(et: ETRobot) -> None:
    """
    障害物を回避するための一連のアクションを実行します。

    引数:
        et (ETRobot): 制御対象のETRobotインスタンス。
    """
    action_chain = (
        (et, 30, 50, 1.5),  # 左モーター30、右モーター50で1.5秒間（左向き）
        (et, 70, 40, 2.0),  # 左モーター70、右モーター40で2.0秒間（左迂回）
        (et, 0, 60, 0.5),   # 左モーター0、右モーター60で0.5秒間（右向き）
    )

    _perform_action_chain(action_chain)
