from __future__ import annotations

from threading import Thread

import cv2
import numpy as np


class WebcamVideoStream:
    """A threaded video stream reader for webcams with optional video recording.

    This class provides a non-blocking way to read frames from a video source
    by running the capture loop in a separate thread. It supports configurable
    resolution, frame rate, and optional video recording to file.

    The threaded approach helps prevent frame drops and provides smoother
    video processing by maintaining a minimal buffer size and continuous
    frame updates in the background.

    Args:
        src: Video source - can be camera index (int) or video file path (str)
        save_video: Whether to save captured video to file
        save_path: Path for saving video file (required if save_video is True)
        resolution: Video resolution as (width, height) tuple. Default: (640, 320)
        fps: Frames per second for capture and recording. Default: 30

    Example:
        >>> stream = WebcamVideoStream(src=0, save_video=False)
        >>> stream.start()
        >>> grabbed, frame = stream.read()
        >>> stream.stop()
    """

    def __init__(
        self,
        src: int | str,
        save_video: bool,
        save_path: str = "",
        resolution: tuple = (640, 320),
        fps: int = 30,
    ):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        (self.grabbed, self.frame) = self.stream.read()

        self.video_writer = None
        if save_video is True:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
            video_filename = save_path
            self.video_writer = cv2.VideoWriter(
                filename=video_filename,
                fourcc=fourcc,
                fps=fps,
                frameSize=(resolution[0], resolution[1]),
            )

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self) -> WebcamVideoStream:
        """Start the background thread for reading video frames.

        Returns:
            Self reference for method chaining.
        """
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self) -> None:
        """Continuously update frames from the video stream in a background thread.

        This method runs in a loop until stopped, constantly reading new frames
        from the video source to keep the frame buffer current.
        """
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the most recently captured frame.

        If video recording is enabled, also writes the frame to the output file.

        Returns:
            Tuple of (success_flag, frame) where success_flag indicates if the
            frame was successfully captured and frame is the image data as a
            numpy array, or None if capture failed.
        """
        if self.video_writer is not None:
            self.video_writer.write(self.frame)

        # return the frame most recently read
        return (self.grabbed, self.frame)

    def stop(self) -> None:
        """Stop the video stream and clean up resources.

        This method stops the background thread, releases the video writer
        (if recording), and releases the video capture stream.
        """
        # indicate that the thread should be stopped
        self.stopped = True

        if self.video_writer is not None:
            self.video_writer.release()

        self.stream.release()
