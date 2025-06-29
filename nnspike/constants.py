# Region of Interest for CNN model
# ROI_CNN = (50, 50, 590, 350)

# Region of Interest for opencv
# ROI_OPENCV = (180, 300, 460, 400)  # 左右をそれぞれ30pxずつ内側に狭めた例

# Camera and robot geometry constants
CAMERA_HEIGHT = 0.20  # Camera height above ground in meters
CAMERA_FOCAL_LENGTH_PIXELS = 640  # Approximate focal length in pixels
WHEELBASE = 0.10  # Distance between wheels in meters

# OpenCV用のROIと画像サイズ（run_opencv.pyと同じ値）
# IMAGE_WIDTH = 640                  # カメラ画像の幅
# IMAGE_HEIGHT = 480                 # カメラ画像の高さ
