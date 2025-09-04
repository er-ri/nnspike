#!/usr/bin/env python3
"""
calculate_attitude_angle関数の簡素化前後で結果が同じになるかテスト
"""
import math

def calculate_attitude_angle_old(
    offset_pixels: float,
    roi_bottom_y: int,
    camera_height: float = 0.20,
    focal_length_pixels: float = 640,
) -> float:
    """
    元の複雑な計算式
    """
    image_height = 480  # 標準カメラ解像度
    ground_distance = camera_height * focal_length_pixels / (image_height - roi_bottom_y)
    lateral_offset_meters = offset_pixels * ground_distance / focal_length_pixels
    theta = math.atan2(lateral_offset_meters, ground_distance)
    return theta

def calculate_attitude_angle_new(
    offset_pixels: float,
    focal_length_pixels: float = 640,
) -> float:
    """
    簡素化した計算式
    """
    theta = math.atan2(offset_pixels, focal_length_pixels)
    return theta

# テストケース - 同じoffset_pixelsで異なるroi_bottom_yの組み合わせを多数検証
test_cases = [
    # offset_pixels = 0のケース
    {"offset_pixels": 0, "roi_bottom_y": 300},
    {"offset_pixels": 0, "roi_bottom_y": 450},
    {"offset_pixels": 0, "roi_bottom_y": 470},
    
    # offset_pixels = 25のケース
    {"offset_pixels": 25, "roi_bottom_y": 300},
    {"offset_pixels": 25, "roi_bottom_y": 450},
    {"offset_pixels": 25, "roi_bottom_y": 470},
    
    # offset_pixels = 50のケース
    {"offset_pixels": 50, "roi_bottom_y": 300},
    {"offset_pixels": 50, "roi_bottom_y": 450},
    {"offset_pixels": 50, "roi_bottom_y": 470},
    
    # offset_pixels = 100のケース
    {"offset_pixels": 100, "roi_bottom_y": 300},
    {"offset_pixels": 100, "roi_bottom_y": 450},
    {"offset_pixels": 100, "roi_bottom_y": 470},
    
    # offset_pixels = -25のケース（負の値）
    {"offset_pixels": -25, "roi_bottom_y": 300},
    {"offset_pixels": -25, "roi_bottom_y": 450},
    {"offset_pixels": -25, "roi_bottom_y": 470},
    
    # offset_pixels = -50のケース（負の値）
    {"offset_pixels": -50, "roi_bottom_y": 300},
    {"offset_pixels": -50, "roi_bottom_y": 450},
    {"offset_pixels": -50, "roi_bottom_y": 470},
    
    # offset_pixels = -100のケース（負の値）
    {"offset_pixels": -100, "roi_bottom_y": 300},
    {"offset_pixels": -100, "roi_bottom_y": 450},
    {"offset_pixels": -100, "roi_bottom_y": 470},
    
    # 極端なケース
    {"offset_pixels": 200, "roi_bottom_y": 300},
    {"offset_pixels": 200, "roi_bottom_y": 450},
    {"offset_pixels": 200, "roi_bottom_y": 470},
    
    {"offset_pixels": -200, "roi_bottom_y": 300},
    {"offset_pixels": -200, "roi_bottom_y": 450},
    {"offset_pixels": -200, "roi_bottom_y": 470},
]

print("計算結果比較テスト")
print("=" * 80)
print(f"{'offset_px':<10} {'roi_y':<8} {'旧式(rad)':<15} {'新式(rad)':<15} {'差分':<12} {'同じ?'}")
print("-" * 80)

# 同じoffset_pixelsでの結果をグループ化して検証
offset_groups = {}
for case in test_cases:
    offset_px = case["offset_pixels"]
    if offset_px not in offset_groups:
        offset_groups[offset_px] = []
    offset_groups[offset_px].append(case)

all_same = True
for case in test_cases:
    offset_px = case["offset_pixels"]
    roi_y = case["roi_bottom_y"]
    
    # 旧式計算
    theta_old = calculate_attitude_angle_old(offset_px, roi_y)
    
    # 新式計算
    theta_new = calculate_attitude_angle_new(offset_px)
    
    # 差分計算
    diff = abs(theta_old - theta_new)
    is_same = diff < 1e-10  # 浮動小数点誤差を考慮
    if not is_same:
        all_same = False
    
    print(f"{offset_px:<10} {roi_y:<8} {theta_old:<15.10f} {theta_new:<15.10f} {diff:<12.2e} {'OK' if is_same else 'NG'}")

print("-" * 80)
print(f"全体結果: {'OK 全て同じ' if all_same else 'NG 相違あり'}")

# 同じoffset_pixelsでのroi_bottom_y間の一致性を検証
print("\n" + "=" * 80)
print("同じoffset_pixelsでのroi_bottom_y間一致性検証")
print("=" * 80)

for offset_px, cases in offset_groups.items():
    if len(cases) <= 1:
        continue
    
    print(f"\noffset_pixels = {offset_px}の場合:")
    results = []
    for case in cases:
        roi_y = case["roi_bottom_y"]
        theta_old = calculate_attitude_angle_old(offset_px, roi_y)
        theta_new = calculate_attitude_angle_new(offset_px)
        results.append((roi_y, theta_old, theta_new))
    
    # 各roi_bottom_yでの結果を表示
    for roi_y, theta_old, theta_new in results:
        print(f"  roi_y={roi_y}: 旧式={theta_old:.10f}, 新式={theta_new:.10f}")
    
    # 同じoffset_pixelsでの旧式結果が全て同じかチェック
    old_values = [result[1] for result in results]
    new_values = [result[2] for result in results]
    
    old_all_same = all(abs(val - old_values[0]) < 1e-10 for val in old_values)
    new_all_same = all(abs(val - new_values[0]) < 1e-10 for val in new_values)
    
    print(f"  旧式結果の一致性: {'OK' if old_all_same else 'NG'}")
    print(f"  新式結果の一致性: {'OK' if new_all_same else 'NG'}")
    print(f"  旧式vs新式の一致性: {'OK' if abs(old_values[0] - new_values[0]) < 1e-10 else 'NG'}")

# 詳細検証：数式の展開
print("\n" + "=" * 60)
print("数式展開の検証")
print("=" * 60)

offset_pixels = 50
roi_bottom_y = 470
camera_height = 0.20
focal_length_pixels = 640
image_height = 480

print(f"テストパラメータ:")
print(f"  offset_pixels = {offset_pixels}")
print(f"  roi_bottom_y = {roi_bottom_y}")
print(f"  camera_height = {camera_height}")
print(f"  focal_length_pixels = {focal_length_pixels}")
print(f"  image_height = {image_height}")

print(f"\n旧式の計算過程:")
ground_distance = camera_height * focal_length_pixels / (image_height - roi_bottom_y)
print(f"  ground_distance = {camera_height} * {focal_length_pixels} / ({image_height} - {roi_bottom_y})")
print(f"                  = {ground_distance}")

lateral_offset_meters = offset_pixels * ground_distance / focal_length_pixels
print(f"  lateral_offset_meters = {offset_pixels} * {ground_distance} / {focal_length_pixels}")
print(f"                        = {lateral_offset_meters}")

theta_old_detailed = math.atan2(lateral_offset_meters, ground_distance)
print(f"  theta = atan2({lateral_offset_meters}, {ground_distance})")
print(f"        = {theta_old_detailed}")

print(f"\n新式の計算過程:")
theta_new_detailed = math.atan2(offset_pixels, focal_length_pixels)
print(f"  theta = atan2({offset_pixels}, {focal_length_pixels})")
print(f"        = {theta_new_detailed}")

print(f"\n数式的証明:")
print(f"  atan2(lateral_offset_meters, ground_distance)")
print(f"  = atan2(offset_pixels * ground_distance / focal_length_pixels, ground_distance)")
print(f"  = atan2(offset_pixels * ground_distance, ground_distance * focal_length_pixels)")
print(f"  = atan2(offset_pixels, focal_length_pixels)")
print(f"  ∴ roi_bottom_yの値に関係なく結果は同じ")
