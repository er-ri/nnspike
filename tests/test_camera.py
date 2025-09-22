import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import time
from nnspike.constants import CAMERA_WIDTH, CAMERA_HEIGHT

def test_camera(camera_id=0):
    """
    Test camera connection and display the image to the user
    Args:
        camera_id (int): Camera device ID (default: 0 for primary camera)
    """
    try:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {camera_id}")
            return False

        print(f"Camera {camera_id} connected successfully")
        print("Press 'q' to quit")

        frame_count = 0
        start_time = time.time()
        last_time = start_time
        while True:
            ret, frame = cap.read()
            now = time.time()
            dt = now - last_time
            last_time = now
            if not ret:
                print("Error: Failed to grab frame")
                break
            frame_count += 1
            print(f"Frame {frame_count}: dt={dt*1000:.2f}ms")
            if frame_count >= 100:
                break
        elapsed = time.time() - start_time
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames: {frame_count}, elapsed: {elapsed:.2f}s, fps: {frame_count/elapsed:.2f}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_camera()
