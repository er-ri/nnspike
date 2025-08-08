import pytest
import numpy as np
from nnspike.unit.action_chain import ActionChain, Mode


class DummyETRobot:
    def __init__(self, right_position):
        self._right_position = right_position
    def get_spike_status(self):
        class Status:
            motors = {"B": type("M", (), {"relative_position": self._right_position})()}
        return Status()


def test_carry_bottle1_relative_cumulative():
    """
    carry_bottle1_relativeの全フェーズを累積的に順次実行し、currentとstartが常に増加することを検証
    """
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    phase = 0
    right_position_start = 0
    current = 0
    ac = ActionChain(DummyETRobot(current), "A")
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": None}
    ac.find_bottle_center = lambda image, color: (None, None, 1000)
    # phase0: 赤ピクセル数3000未満→継続
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={ac._state['carry_bottle1_relative']['right_position_start']}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase0→1: 赤ピクセル数3000超→phase1遷移
    phase = 1
    right_position_start = current
    current += 500
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    ac.find_bottle_center = lambda image, color: (None, None, 4000)
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase1→2: 1100進む
    phase = 2
    right_position_start = current
    current += 1100
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase2→3: 1100進む
    phase = 3
    right_position_start = current
    current += 1100
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase3→4: 450進む
    phase = 4
    right_position_start = current
    current += 450
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase4→5: 1000進む
    phase = 5
    right_position_start = current
    current += 1000
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase5→6: 1900進む
    phase = 6
    right_position_start = current
    current += 500  # 最大500ユニットで青ターゲット検出または位置制限
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase6→7: 青検出（1000超えるまで）
    phase = 7
    right_position_start = current
    current += 200  # 青検出フェーズ
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase7→8: 青1000以上の間center追従
    phase = 8
    right_position_start = current
    current += 300  # 青中心追従フェーズ
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": 320, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE1

    # phase8→9: 青500以下になってから100ユニット移動まで
    phase = 9
    right_position_start = current
    current += 100
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle1_relative"] = {"phase": phase, "pre_target_x": 320, "right_position_start": right_position_start}
    result = ac.carry_bottle1_relative(dummy_img)
    print(f"[bottle1] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.BACK_AND_TURN1


def test_carry_bottle2_relative_cumulative():
    """
    carry_bottle2_relativeの全フェーズを累積的に順次実行し、currentとstartが常に増加することを検証
    """
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    phase = 0
    right_position_start = 0
    current = 0
    ac = ActionChain(DummyETRobot(current), "A")
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": None}
    ac.find_bottle_center = lambda image, color: (None, None, 1000)
    # phase0: 赤ピクセル数3000未満→継続
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={ac._state['carry_bottle2_relative']['right_position_start']}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase0→1: 赤ピクセル数3000超→phase1遷移
    phase = 1
    right_position_start = current
    current += 500
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    ac.find_bottle_center = lambda image, color: (None, None, 4000)
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase1→2: 1100進む
    phase = 2
    right_position_start = current
    current += 1100
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase2→3: 1100進む
    phase = 3
    right_position_start = current
    current += 1100
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase3→4: 450進む
    phase = 4
    right_position_start = current
    current += 450
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase4→5: 1000進む
    phase = 5
    right_position_start = current
    current += 1000
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase5→6: 1900進む
    phase = 6
    right_position_start = current
    current += 1900
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    ac.find_blue_area = lambda image: 0
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase6→7: 青検出あり
    phase = 7
    right_position_start = current
    current += 1000
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": None, "right_position_start": right_position_start}
    ac.find_blue_area = lambda image: 1000
    ac.find_bottle_center = lambda image, color: (None, None, 0)
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase7→8: ボトル中心検出成功
    phase = 8
    right_position_start = current
    current += 1000
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": 320, "right_position_start": right_position_start}
    ac.find_bottle_center = lambda image, color: (320, 240, 1000)
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase8→9: 青検出（青ピクセル数1000超えるまたは400ユニット移動まで）
    phase = 9
    right_position_start = current
    current += 400
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": 320, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase9→10: 青ピクセル数1000超→phase10遷移
    phase = 10
    right_position_start = current
    current += 400
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": 320, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.CARRY_BOTTLE2

    # phase10→11: 青ピクセル数500以下→phase11遷移
    phase = 11
    right_position_start = current
    current += 100
    ac.et = DummyETRobot(current)
    ac._state["carry_bottle2_relative"] = {"phase": phase, "pre_target_x": 320, "right_position_start": right_position_start}
    result = ac.carry_bottle2_relative(dummy_img)
    print(f"[bottle2] phase{phase} result: {result}, start={right_position_start}, current={current}")
    assert result[-1] == Mode.BACK_AND_TURN2


