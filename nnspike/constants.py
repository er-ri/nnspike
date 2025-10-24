
from enum import Enum


# Region of Interest for CNN model
ROI_CNN = (0, 0, 640, 480)
ROI_LINE_TRACING = (0, 250, 640, 480)
ROI_LINE_STRAIGHT = (100, 250, 540, 480)
ROI_LINE_STRAIGHT_FAST = (150, 0, 490, 480)
ROI_VIRTUAL = (100, 100, 540, 330)
ROI_LOOP = (100, 200, 540, 480)
ROI_LINE_LEFT = (0, 50, 240, 480)
ROI_LINE_HORIZON1 = (200, 50, 440, 480)
ROI_LINE_HORIZON2 = (100, 300, 540, 480)
ROI_LINE_HORIZON3 = (200, 300, 440, 480)
ROI_LINE_VERTICAL1 = (200, 200, 440, 480)
ROI_LINE_CORNER = (0, 250, 400, 480)
ROI_COLOR = (100, 0, 540, 480)
ROI_COLOR2 = (200, 0, 440, 480)

OFFSET_Y = 470  # 0.20 meters to the ground

# Base speed for robot movement (used throughout action logic)
BASE_SPEED = 45
HIGH_SPEED_BASE = 10000

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
    HIGH_SPEED = 10  # 旧TURN_RIGHT, 高速走行モード
    FORWARD = 11
    BACKWARD = 12
    GATE_PASS = 13
    EYE_BLUE = 14
    TURN_LEFT_YAW = 15
    TURN_RIGHT_YAW = 16
    BLUE_BOTTLE_CATCH = 17
    TURN_LEFT = 19
    TURN_RIGHT = 20
    NVIDIA_FOLLOW = 21
    DOUBLE_LOOP = 22  # key1ダブルループモード
    FAST_LAP = 23  # 高速周回モード（yキー用）
    TEST = 99  # テスト用モード

# Camera and Robot Geometry Constants
CAMERA_WIDTH = 640  # Camera frame width in pixels
CAMERA_HEIGHT = 480  # Camera frame height in pixels  


