from enum import Enum


# Region of Interest for CNN model
ROI_CNN = (0, 0, 640, 480)
OFFSET_Y = 470  # 0.20 meters to the ground

# Base speed for robot movement (used throughout action logic)
BASE_SPEED = 45


# Behavior Mode
class Mode(Enum):
    FOLLOW_LEFT_EDGE = 0
    FOLLOW_RIGHT_EDGE = 1
    AVOID_OBSTACLE = 2
    CARRY_BOTTLE1 = 3
    BACK_AND_TURN1 = 4
    CARRY_BOTTLE2 = 5
    BACK_AND_TURN2 = 6
    HEAD_GOAL = 7
    PAUSE = 8
    TURN_LEFT = 9
    TURN_RIGHT = 10
    FORWARD = 11
    BACKWARD = 12
    GATE_PASS = 13
    EYE_BLUE = 14
    SMALL_TURN_LEFT = 15
    SMALL_TURN_RIGHT = 16
    BLUE_BOTTLE_CATCH = 17
    TURN_AT_END = 18
    TURN_LEFT_RELATIVE = 19
    TURN_RIGHT_RELATIVE = 20

# One-Hot Encoding for Modes
NUM_MODES = len(Mode)
# Scale for relative position in motor control
RELATIVE_POSITION_SCALE = 80000
# Threshold for obstacle avoidance based on yellow pixel count
OBSTACLE_AVOIDANCE_THRESHOLD = 14000

# Camera and Robot Geometry Constants
CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # Approximate focal length in pixels
WHEELBASE = 0.11  # Distance between wheels in meters


# Data Types used for data balancing
class DataType(Enum):
    SINGLE_LINE = 0
    INTERSECTION = 1
    SPECIAL = 9
