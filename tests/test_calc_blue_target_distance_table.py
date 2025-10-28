from nnspike.utils.control import calc_blue_target_distance

# 表のサンプル値（y, distance, practical_distance）
sample_table = [
    (30, 400, 1200),
    (100, 330, 994),
    (150, 280, 818),
    (200, 230, 642),
    (250, 180, 500),
    (260, 170, 482),
    (275, 155, 457),
    (290, 140, 441),
    (300, 130, 420),
    (310, 120, 412),
    (325, 105, 381),
    (340, 90, 361),
    (350, 80, 350),
    (400, 30, 216),
    (430, 0, 165),
]

# CAMERA_HEIGHT*0.9 = 430前提でテスト
def test_calc_blue_target_distance():
    CAMERA_HEIGHT = 430 / 0.9
    errors = []
    for y, distance_expected, pd_expected in sample_table:
        blue_center = (0, y)
        # monkeypatch: CAMERA_HEIGHTを430にする
        import nnspike.utils.control as control_mod
        control_mod.CAMERA_HEIGHT = 430
        result = calc_blue_target_distance(blue_center)
        if result != pd_expected:
            errors.append(f"y={y}: got {result}, expected {pd_expected}")
    if errors:
        print("NG:")
        for e in errors:
            print(e)
    else:
        print("OK: all values match table")

if __name__ == "__main__":
    test_calc_blue_target_distance()
