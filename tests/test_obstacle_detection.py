"""
Test module for obstacle detection functionality.

This module tests the detect_two_obstacles_and_stop function to ensure
it correctly integrates with the robot framework and provides expected
obstacle detection behavior.
"""

import os
import sys
import time
from unittest.mock import Mock, MagicMock

# Add parent directory to path to import nnspike modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from nnspike.utils.control import detect_two_obstacles_and_stop, reset_obstacle_detection_state
from nnspike.unit.etrobot import ETRobot
from nnspike.unit.spike_status import SpikeStatus, SensorStatus


def create_mock_etrobot(distance_reading):
    """Create a mock ETRobot with specified distance sensor reading."""
    mock_robot = Mock(spec=ETRobot)
    mock_status = Mock(spec=SpikeStatus)
    mock_sensors = Mock(spec=SensorStatus)
    mock_sensors.distance = distance_reading
    mock_status.sensors = mock_sensors
    mock_robot.get_spike_status.return_value = mock_status
    mock_robot.brake = Mock()
    return mock_robot


def test_no_obstacle_detection():
    """Test that robot doesn't stop when no obstacles are detected."""
    print("Testing no obstacle detection...")
    reset_obstacle_detection_state()
    
    # Create mock robot with no obstacle (distance > threshold)
    mock_robot = create_mock_etrobot(distance_reading=200)
    
    # Call function multiple times - should not stop
    for _ in range(5):
        result = detect_two_obstacles_and_stop(mock_robot)
        assert result is False, "Robot should not stop when no obstacles detected"
        time.sleep(0.1)
    
    # Verify brake was never called
    mock_robot.brake.assert_not_called()
    print("✓ No obstacle detection test passed")


def test_single_obstacle_detection():
    """Test that robot doesn't stop with only one obstacle detection."""
    print("Testing single obstacle detection...")
    reset_obstacle_detection_state()
    
    # Create mock robot with obstacle (distance < threshold)
    mock_robot = create_mock_etrobot(distance_reading=50)
    
    # Call function once - should detect but not stop
    result = detect_two_obstacles_and_stop(mock_robot)
    assert result is False, "Robot should not stop with only one obstacle detection"
    
    # Verify brake was not called
    mock_robot.brake.assert_not_called()
    print("✓ Single obstacle detection test passed")


def test_two_obstacles_detection():
    """Test that robot stops when two obstacles are detected."""
    print("Testing two obstacles detection...")
    reset_obstacle_detection_state()
    
    # Create mock robot with obstacle (distance < threshold)
    mock_robot = create_mock_etrobot(distance_reading=50)
    
    # First detection
    result1 = detect_two_obstacles_and_stop(mock_robot)
    assert result1 is False, "Robot should not stop on first detection"
    
    # Wait a bit to ensure detections are separate
    time.sleep(0.2)
    
    # Second detection - should trigger stop
    result2 = detect_two_obstacles_and_stop(mock_robot)
    assert result2 is True, "Robot should stop on second obstacle detection"
    
    # Verify brake was called
    mock_robot.brake.assert_called_once()
    print("✓ Two obstacles detection test passed")


def test_obstacle_detection_with_none_reading():
    """Test that function handles None distance readings gracefully."""
    print("Testing None distance reading...")
    reset_obstacle_detection_state()
    
    # Create mock robot with None distance reading
    mock_robot = create_mock_etrobot(distance_reading=None)
    
    # Call function - should return False and not crash
    result = detect_two_obstacles_and_stop(mock_robot)
    assert result is False, "Function should return False for None readings"
    
    # Verify brake was not called
    mock_robot.brake.assert_not_called()
    print("✓ None reading test passed")


def test_invalid_etrobot_parameter():
    """Test that function raises error for invalid ETRobot parameter."""
    print("Testing invalid ETRobot parameter...")
    reset_obstacle_detection_state()
    
    # Try to call with invalid parameter
    try:
        detect_two_obstacles_and_stop("not_a_robot")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "etrobot must be an instance of ETRobot" in str(e)
        print("✓ Invalid parameter test passed")


def test_custom_parameters():
    """Test function with custom threshold and detection parameters."""
    print("Testing custom parameters...")
    reset_obstacle_detection_state()
    
    # Create mock robot with obstacle at 150mm
    mock_robot = create_mock_etrobot(distance_reading=150)
    
    # Test with higher threshold (200mm) - should detect obstacle
    result1 = detect_two_obstacles_and_stop(
        mock_robot, 
        obstacle_threshold=200, 
        required_detections=3  # Need 3 detections
    )
    assert result1 is False, "Should not stop with only 1 detection when 3 required"
    
    time.sleep(0.2)
    result2 = detect_two_obstacles_and_stop(
        mock_robot, 
        obstacle_threshold=200, 
        required_detections=3
    )
    assert result2 is False, "Should not stop with only 2 detections when 3 required"
    
    time.sleep(0.2)
    result3 = detect_two_obstacles_and_stop(
        mock_robot, 
        obstacle_threshold=200, 
        required_detections=3
    )
    assert result3 is True, "Should stop with 3 detections when 3 required"
    
    # Verify brake was called
    mock_robot.brake.assert_called_once()
    print("✓ Custom parameters test passed")


def run_all_tests():
    """Run all tests."""
    print("Running obstacle detection tests...\n")
    
    test_no_obstacle_detection()
    test_single_obstacle_detection()
    test_two_obstacles_detection()
    test_obstacle_detection_with_none_reading()
    test_invalid_etrobot_parameter()
    test_custom_parameters()
    
    print("\n✅ All obstacle detection tests passed!")


if __name__ == "__main__":
    run_all_tests()