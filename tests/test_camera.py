import cv2


def test_camera(camera_id: int = 0) -> bool:
    """Test camera connection and display the image to the user.

    Args:
        camera_id (int): Camera device ID (default: 0 for primary camera)
    """
    try:
        # Open the camera
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {camera_id}")
            return False

        print(f"Camera {camera_id} connected successfully")
        print("Press 'q' to quit")

        # Display frames from the camera
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to grab frame")
                break

            # Display the current frame
            cv2.imshow("Camera Test", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    test_camera()
