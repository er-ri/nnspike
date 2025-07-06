from enum import Enum

# Region of Interest for CNN model
ROI_CNN = (0, 0, 640, 480)
OFFSET_Y = 470  # 0.20 meters to the ground


# Behavior Mode
class Mode(Enum):
    LEFT_EDGE_FOLLOWING = 0
    RIGHT_EDGE_FOLLOWING = 1
    OBSTACLE_AVOIDANCE = 2
    BOTTLE_CARRYING = 3
    HEADING_GATE = 4
    BACKWARD_AND_TURN_AROUND = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    BACKWARD_AND_HEADING_GOAL = 8


# Camera and Robot Geometry Constants
CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # Approximate focal length in pixels
WHEELBASE = 0.11  # Distance between wheels in meters
