from enum import Enum

# Region of Interest for CNN model
ROI_CNN = (0, 0, 640, 480)
OFFSET_Y = 470  # 0.20 meters to the ground


# Behavior Mode
class Mode(Enum):
    # Double Loop
    FOLLOW_LEFT_EDGE = 0
    FOLLOW_RIGHT_EDGE = 1
    AVOID_OBSTACLE = 2

    # Carry Bottle Phases
    CARRY_BOTTLE_PHASE1 = 3
    CARRY_BOTTLE_PHASE2 = 4
    CARRY_BOTTLE_PHASE3 = 5
    CARRY_BOTTLE_PHASE4 = 6
    CARRY_BOTTLE_PHASE5 = 7
    CARRY_BOTTLE_PHASE6 = 8
    CARRY_BOTTLE_PHASE7 = 9
    CARRY_BOTTLE_PHASE8 = 10
    CARRY_BOTTLE_PHASE9 = 11
    CARRY_BOTTLE_PHASE10 = 12
    CARRY_BOTTLE_PHASE11 = 13
    CARRY_BOTTLE_PHASE12 = 14
    CARRY_BOTTLE_PHASE13 = 15
    CARRY_BOTTLE_PHASE14 = 16
    CARRY_BOTTLE_PHASE15 = 17
    CARRY_BOTTLE_PHASE16 = 18
    CARRY_BOTTLE_PHASE17 = 19
    CARRY_BOTTLE_PHASE18 = 20
    CARRY_BOTTLE_PHASE19 = 21
    GOAL = 22

    PHASE_COMPLETED = 99

    # Manual Control Modes
    MOVE_FORWARD = 100
    MOVE_FORWARD_LEFT = 110
    MOVE_FORWARD_RIGHT = 120
    TURN_LEFT = 130
    TURN_RIGHT = 140
    MOVE_BACKWARD = 150
    PAUSE = 160
    REMOTE_CONTROL = 999


# Scale for relative position in motor control
RELATIVE_POSITION_SCALE = 80000
# Threshold for obstacle avoidance based on yellow pixel count
OBSTACLE_AVOIDANCE_THRESHOLD = 14000

# Camera and Robot Geometry Constants
CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # Approximate focal length in pixels
WHEELBASE = 0.11  # Distance between wheels in meters
