from threading import Thread

import cv2


class WebcamVideoStream:
    def __init__(self, src: int | str, save_video: bool, timestamp: str, resolution: tuple = (640, 320)):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        (self.grabbed, self.frame) = self.stream.read()

        self.video_writer = None
        if save_video is True:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
            video_filename = f"storage/videos/{timestamp}_picamera.avi"
            self.video_writer = cv2.VideoWriter(
                filename=video_filename,
                fourcc=fourcc,
                fps=30,
                frameSize=(640, 480),
            )

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        if self.video_writer is not None:
            self.video_writer.write(self.frame)

        # return the frame most recently read
        return (self.grabbed, self.frame)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        if self.video_writer is not None:
            self.video_writer.release()

        self.stream.release()
