from threading import Thread

import cv2
import numpy as np


class WebcamVideoStream:
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

    def start(self) -> "WebcamVideoStream":
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self) -> None:
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.video_writer is not None:
            self.video_writer.write(self.frame)

        # return the frame most recently read
        return (self.grabbed, self.frame)

    def stop(self) -> None:
        # indicate that the thread should be stopped
        self.stopped = True

        if self.video_writer is not None:
            self.video_writer.release()

        self.stream.release()
