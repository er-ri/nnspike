
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
class DummyETRobot:
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
    def get_spike_status(self):
        return self.DummyStatus(self._pos)

class TestCarryBottle1RelativeCompare(unittest.TestCase):
    def setUp(self):
        self.et = DummyETRobot()
        self.chain = ActionChain(self.et, course="right")
        # バックアップ側もインスタンス生成
        try:
            backup_path = "c:/Users/MSAD/github/nnspike/backup/20250815/action_chain.py"
            spec = importlib.util.spec_from_file_location("backup_action_chain", backup_path)
            backup_module = importlib.util.module_from_spec(spec)
            sys.modules["backup_action_chain"] = backup_module
            spec.loader.exec_module(backup_module)
            self.BackupActionChain = getattr(backup_module, "ActionChain", None)
        except Exception:
            self.BackupActionChain = None

    def test_phase_boundary_values(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 各フェーズの分岐条件に合わせて境界値を設定
        phase_boundaries = [
            # phase0: 赤ピクセル数 2999, 3000, 3001
            {"phase": 0, "red_pixel_count": [2999, 3000, 3001]},
            # phase1: モーター差分 999, 1000, 1001, 赤ピクセル 499, 500, 501
            {"phase": 1, "motor_diff": [999, 1000, 1001], "red_pixel_count": [499, 500, 501]},
            # phase2: モーター差分 1199, 1200, 1201
            {"phase": 2, "motor_diff": [1199, 1200, 1201]},
            # phase3: モーター差分 389, 390, 391
            {"phase": 3, "motor_diff": [389, 390, 391]},
            # phase4: モーター差分 99, 100, 101
            {"phase": 4, "motor_diff": [99, 100, 101]},
            # phase5: モーター差分 999, 1000, 1001
            {"phase": 5, "motor_diff": [999, 1000, 1001]},
            # phase6: モーター差分 1899, 1900, 1901
            {"phase": 6, "motor_diff": [1899, 1900, 1901]},
            # phase7: モーター差分 299, 300, 301, 499, 500, 501, 青ターゲット検出 True/False
            {"phase": 7, "motor_diff": [299, 300, 301, 499, 500, 501], "blue_detected": [False, True]},
            # phase8: 青ピクセル数 999, 1000, 1001
            {"phase": 8, "blue_pixel_count": [999, 1000, 1001]},
            # phase9: 青ピクセル数 499, 500, 501
            {"phase": 9, "blue_pixel_count": [499, 500, 501]},
            # phase10: モーター差分 299, 300, 301
            {"phase": 10, "motor_diff": [299, 300, 301]},
        ]
        all_match = True
        for boundary in phase_boundaries:
            phase = boundary["phase"]
            print(f"\n[PHASE {phase} boundary test]")
            # バックアップ側も毎回初期化
            backup_chain = self.BackupActionChain(self.et, course="right") if self.BackupActionChain else None
            if "red_pixel_count" in boundary:
                for red_px in boundary["red_pixel_count"]:
                    self.et._pos = 0
                    self.chain.initialize_action()
                    self.chain._phase._state["phase"] = phase
                    if backup_chain:
                        backup_chain._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": None}
                    print(f"red_pixel_count={red_px}")
                    _, speeds_current, mode_current = self.chain.carry_bottle1_relative(image)
                    if backup_chain:
                        _, speeds_backup, mode_backup = backup_chain.carry_bottle1_relative(image)
                    else:
                        speeds_backup, mode_backup = None, None
                    if (speeds_current != speeds_backup) or (mode_current != mode_backup):
                        print(f"[DIFF] red_pixel_count={red_px}: current=({speeds_current}, {mode_current}), backup=({speeds_backup}, {mode_backup})")
                        all_match = False
                    else:
                        print(f"[MATCH] red_pixel_count={red_px}: speeds={speeds_current}, mode={mode_current}")
            if "motor_diff" in boundary:
                for diff in boundary["motor_diff"]:
                    self.chain.initialize_action()
                    self.chain._phase._state["phase"] = phase
                    self.chain._phase._state["position_start"] = 0
                    self.et._pos = diff
                    if backup_chain:
                        backup_chain._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": None}
                    print(f"motor_diff={diff}")
                    _, speeds_current, mode_current = self.chain.carry_bottle1_relative(image)
                    if backup_chain:
                        _, speeds_backup, mode_backup = backup_chain.carry_bottle1_relative(image)
                    else:
                        speeds_backup, mode_backup = None, None
                    if (speeds_current != speeds_backup) or (mode_current != mode_backup):
                        print(f"[DIFF] motor_diff={diff}: current=({speeds_current}, {mode_current}), backup=({speeds_backup}, {mode_backup})")
                        all_match = False
                    else:
                        print(f"[MATCH] motor_diff={diff}: speeds={speeds_current}, mode={mode_current}")
            if "blue_pixel_count" in boundary:
                for blue_px in boundary["blue_pixel_count"]:
                    self.chain.initialize_action()
                    self.chain._phase._state["phase"] = phase
                    if backup_chain:
                        backup_chain._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": None}
                    print(f"blue_pixel_count={blue_px}")
                    _, speeds_current, mode_current = self.chain.carry_bottle1_relative(image)
                    if backup_chain:
                        _, speeds_backup, mode_backup = backup_chain.carry_bottle1_relative(image)
                    else:
                        speeds_backup, mode_backup = None, None
                    if (speeds_current != speeds_backup) or (mode_current != mode_backup):
                        print(f"[DIFF] blue_pixel_count={blue_px}: current=({speeds_current}, {mode_current}), backup=({speeds_backup}, {mode_backup})")
                        all_match = False
                    else:
                        print(f"[MATCH] blue_pixel_count={blue_px}: speeds={speeds_current}, mode={mode_current}")
            if "blue_detected" in boundary:
                for detected in boundary["blue_detected"]:
                    self.chain.initialize_action()
                    self.chain._phase._state["phase"] = phase
                    if backup_chain:
                        backup_chain._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": None}
                    print(f"blue_detected={detected}")
                    _, speeds_current, mode_current = self.chain.carry_bottle1_relative(image)
                    if backup_chain:
                        _, speeds_backup, mode_backup = backup_chain.carry_bottle1_relative(image)
                    else:
                        speeds_backup, mode_backup = None, None
                    if (speeds_current != speeds_backup) or (mode_current != mode_backup):
                        print(f"[DIFF] blue_detected={detected}: current=({speeds_current}, {mode_current}), backup=({speeds_backup}, {mode_backup})")
                        all_match = False
                    else:
                        print(f"[MATCH] blue_detected={detected}: speeds={speeds_current}, mode={mode_current}")
        if all_match:
            print("\n[TEST RESULT] carry_bottle1_relative: 全ての境界値で一致 → 正常終了")
        else:
            print("\n[TEST RESULT] carry_bottle1_relative: 差分あり → 要確認")

if __name__ == "__main__":
    tester = TestCarryBottle1RelativeCompare()
    tester.setUp()
    tester.test_phase_boundary_values()
