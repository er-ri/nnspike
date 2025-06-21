#!/usr/bin/env python3
"""
Test script for calculate_differential_steering function
"""
import math
from nnspike.utils import calculate_differential_steering

def test_steering():
    base_speed = 30
    wheelbase = 0.15
    
    print("Testing calculate_differential_steering function:")
    print(f"Base speed: {base_speed}")
    print(f"Wheelbase: {wheelbase}")
    print("-" * 50)
    
    # Test cases with different angles
    test_angles = [
        0.0,      # Straight
        0.1,      # Slight right turn
        -0.1,     # Slight left turn
        0.3,      # Medium right turn
        -0.3,     # Medium left turn
        0.5,      # Sharp right turn
        -0.5,     # Sharp left turn
    ]
    
    for theta in test_angles:
        left_speed, right_speed = calculate_differential_steering(theta, base_speed, wheelbase)
        theta_deg = math.degrees(theta)
        speed_diff = abs(left_speed - right_speed)
        
        turn_direction = "STRAIGHT"
        if theta > 0.001:
            turn_direction = "RIGHT"
        elif theta < -0.001:
            turn_direction = "LEFT"
            
        print(f"Angle: {theta_deg:6.1f}° ({turn_direction:8s}) | "
              f"Left: {left_speed:5.1f} | Right: {right_speed:5.1f} | "
              f"Diff: {speed_diff:5.1f}")

if __name__ == "__main__":
    test_steering()
