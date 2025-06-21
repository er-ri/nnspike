import cv2
import sys
import os
import time
import random

# Add parent directory to path to import nnspike modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from nnspike.utils import SensorRecorder
from nnspike.unit import ETRobot


def test_recorder_with_camera_real_sensors(
    camera_id=0, duration_seconds=10, port="COM4"
):
    """
    Test the SensorRecorder with camera feed and REAL ETRobot sensor data

    Args:
        camera_id (int): Camera device ID (default: 0 for primary camera)
        duration_seconds (int): Test duration in seconds
        port (str): Serial port for ETRobot connection
    """
    print("=== Testing SensorRecorder with Camera and REAL Sensors ===")

    # Initialize sensor recorder
    sensor_recorder = SensorRecorder(output_dir="tests/output")
    sensor_recorder.start_recording()

    # Video recording setup
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    video_filename = f"tests/output/{timestamp}_test_recording.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = None

    try:
        # Initialize camera
        cap = cv2.VideoCapture(
            camera_id, cv2.CAP_DSHOW
        )  # Use DSHOW backend for better compatibility on Windows

        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {camera_id}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Initialize video writer
        video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

        print(f"Camera {camera_id} connected successfully")
        print(f"Video will be saved to: {video_filename}")

        # Initialize REAL ETRobot
        print(f"Connecting to ETRobot on port {port}...")
        try:
            et_robot = ETRobot(port=port)
            print("ETRobot connected successfully!")
        except Exception as e:
            print(f"Failed to connect to ETRobot: {e}")
            print("Make sure the robot is connected and the port is correct.")
            cap.release()
            return False

        print(f"Recording for {duration_seconds} seconds...")
        print("Press 'q' to quit early")

        start_time = time.time()
        frame_count = 0

        et_robot.set_motor_forward_power(30, 30)

        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to grab frame")
                    break

                frame_count += 1

                # Log sensor data using the recorder
                sensor_recorder.log_frame_data(et_robot.get_spike_status())

                # Display frame with REAL sensor info
                info_text = f"Frame: {frame_count} | Recording: {sensor_recorder.get_frame_count()}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Display real sensor readings
                sensors = et_robot.spike_status.sensors
                distance_text = (
                    f"Distance: {sensors.distance if sensors.distance else 'N/A'}mm"
                )
                cv2.putText(
                    frame,
                    distance_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                force_text = f"Force: {sensors.force if sensors.force else 'N/A'}"
                cv2.putText(
                    frame,
                    force_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                if sensors.color:
                    color_text = f"Color Reflected: {sensors.color.reflected if sensors.color.reflected else 'N/A'}"
                    cv2.putText(
                        frame,
                        color_text,
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                remaining_time = duration_seconds - (time.time() - start_time)
                time_text = f"Time remaining: {remaining_time:.1f}s"
                cv2.putText(
                    frame,
                    time_text,
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                cv2.imshow("Recorder Test - Real Sensors + Camera", frame)

                # Write frame to video file
                if video_writer is not None:
                    video_writer.write(frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Test interrupted by user")
                    break

                # Small delay to simulate real processing time
                time.sleep(0.01)

            print(f"\nTest completed!")
            print(f"Total frames processed: {frame_count}")
            print(f"Total frames recorded: {sensor_recorder.get_frame_count()}")

        finally:
            # Stop ETRobot properly
            et_robot.stop()

        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"Error during recorder test: {e}")
        return False

    finally:
        sensor_recorder.stop_recording()
        print(f"CSV file saved to: {sensor_recorder.get_filename()}")


if __name__ == "__main__":
    print("ETRobot SensorRecorder Test - Camera + Real Sensors")
    print("==================================================")

    # Create test output directory
    os.makedirs("tests/output", exist_ok=True)

    # Test configuration
    duration = input("Enter test duration in seconds (default 10): ")
    try:
        duration = int(duration) if duration else 10
    except ValueError:
        duration = 10

    port = "COM4"  # for Windows

    success = test_recorder_with_camera_real_sensors(
        duration_seconds=duration, port=port
    )
    if success:
        print("\n✅ Camera + Real Sensors test completed successfully!")
    else:
        print("\n❌ Camera + Real Sensors test failed!")
