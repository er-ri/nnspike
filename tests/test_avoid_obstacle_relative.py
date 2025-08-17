import unittest
import numpy as np
from nnspike.unit.action_chain import ActionChain
import sys
import importlib.util

# backup/20250815/action_chain.py から ActionChain クラスを動的import
backup_path = "c:/Users/MSAD/github/nnspike/backup/20250815/action_chain.py"
spec = importlib.util.spec_from_file_location("backup_action_chain", backup_path)
if spec and spec.loader:
    backup_module = importlib.util.module_from_spec(spec)
    sys.modules["backup_action_chain"] = backup_module
    spec.loader.exec_module(backup_module)
    BackupActionChain = getattr(backup_module, "ActionChain")
else:
    BackupActionChain = None

# ダミーETRobot（現行・バックアップ両方で使用）
from nnspike.unit.etrobot import ETRobot

class DummyETRobot(ETRobot):
    def __init__(self):
        self._pos = 0
    def get_motor_position(self):
        return self._pos
    class DummyStatus:
        def __init__(self, pos):
            self.motors = {"B": self.DummyMotor(pos)}
        class DummyMotor:
            def __init__(self, pos):
                self.relative_position = pos
    def get_spike_status(self):  # type: ignore
        return self.DummyStatus(self._pos)

class TestAvoidObstacleRelativeCompare(unittest.TestCase):
    def setUp(self):
        self.et = DummyETRobot()
        self.chain = ActionChain(self.et, course="right")
        if BackupActionChain:
            self.backup_chain = BackupActionChain(self.et, course="right")
        else:
            self.backup_chain = None

    def test_phase_boundary_values(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        phase_boundaries = [
            # phase0: 右モーター 499, 500, 501
            {"phase": 0, "motor_diff": [499, 500, 501]},
            # phase1: 右モーター 549, 550, 551
            {"phase": 1, "motor_diff": [549, 550, 551]},
            # phase2: 右モーター 249, 250, 251, 349, 350, 351
            {"phase": 2, "motor_diff": [249, 250, 251, 349, 350, 351]},
        ]
        all_match = True
        for boundary in phase_boundaries:
            phase = boundary["phase"]
            print(f"\n[PHASE {phase} boundary test]")
            for diff in boundary["motor_diff"]:
                # 初期値を明示
                position_start = 1000
                motor_pos = position_start + diff
                # ダミーロボットを毎回新規生成し、両方同じ値でセット
                et_current = DummyETRobot()
                et_backup = DummyETRobot()
                et_current._pos = motor_pos
                et_backup._pos = motor_pos
                # ActionChain/BackupActionChainを毎回新規生成
                chain = ActionChain(et_current, course="right")
                backup_chain = BackupActionChain(et_backup, course="right") if BackupActionChain else None
                # PhaseManagerの状態を完全一致させる
                chain.initialize_action()
                chain._phase._state["phase"] = phase
                chain._phase._state["position_start"] = position_start
                if backup_chain:
                    from nnspike.unit.action_chain import PhaseManager
                    backup_chain._phase = PhaseManager()
                    backup_chain._phase._state["phase"] = phase
                    backup_chain._phase._state["position_start"] = position_start
                print(f"motor_diff={diff}, right_position_start={position_start}, motor_pos={motor_pos}")
                _, speeds_current, mode_current = chain.avoid_obstacle_relative(image)
                if backup_chain:
                    _, speeds_backup, mode_backup = backup_chain.avoid_obstacle_relative(image)
                else:
                    speeds_backup, mode_backup = None, None
                if (speeds_current != speeds_backup) or (mode_current != mode_backup):
                    print(f"[DIFF] motor_diff={diff}: current=({speeds_current}, {mode_current}), backup=({speeds_backup}, {mode_backup})")
                    all_match = False
                else:
                    print(f"[MATCH] motor_diff={diff}: speeds={speeds_current}, mode={mode_current}")
        if all_match:
            print("\n[TEST RESULT] avoid_obstacle_relative: 全ての境界値で一致 → 正常終了")
        else:
            print("\n[TEST RESULT] avoid_obstacle_relative: 差分あり → 要確認")

if __name__ == "__main__":
    unittest.main()
