from enum import Enum

# Region of Interest for CNN model
ROI_CNN = (0, 0, 640, 480)
OFFSET_Y = 470  # 0.20 meters to the ground


# Behavior Mode
class Mode(Enum):
<<<<<<< HEAD
    LEFT_EDGE_FOLLOWING = 0
    RIGHT_EDGE_FOLLOWING = 1
    OBSTACLE_AVOIDANCE = 2
    BOTTLE_CARRYING = 3
    BOTTLE_CATCH_BLUE = 4
    BLUE_BOTTLE_TO_GATE = 5
    RED_BOTTLE_TO_GATE = 6
    HEADING_GATE = 7
    BACKWARD_AND_TURN_AROUND = 8
    TURN_LEFT = 9
    TURN_RIGHT = 10
    BACKWARD_AND_HEADING_GOAL = 11
    PAUSE = 12
    FORWARD = 13        # 前進専用
    BACKWARD = 14       # 後退専用
    BLUE_BOTTLE = 15   # 新しいモードを追加
    SMALL_TURN_LEFT = 16  # 小さいターンレフト
=======
    FOLLOW_LEFT_EDGE = 0
    FOLLOW_RIGHT_EDGE = 1
    AVOID_OBSTACLE = 2
    HEAD_BOTTLE1 = 3
    CARRY_BOTTLE1 = 4
    HEAD_BOTTLE2 = 5
    CARRY_BOTTLE2 = 6
    HEAD_GOAL = 7
    PAUSE = 8
>>>>>>> wip/teamwork-li


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
