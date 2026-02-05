# Copyright (C) 2022 twyleg
import os
import cv2


class Camera:
    """
    Simple camera wrapper using OpenCV VideoCapture.
    Sets device rotation via v4l2-ctl when available.
    """

    def __init__(self, device_path: str = '/dev/video4', width: int = 640, height: int = 480, rotation: int = 0):
        self.video_capture = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        try:
            os.system(f'v4l2-ctl -d {device_path} --set-ctrl=rotate={rotation}')

            fourcc_yuyv = cv2.VideoWriter_fourcc(*'YUYV')
            fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_capture.set(cv2.CAP_PROP_FOURCC, fourcc_yuyv)

            print(f"Requesting resolution: {width}x{height}")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Request higher FPS (60fps if supported)
            self.video_capture.set(cv2.CAP_PROP_FPS, 60)

            # Set buffer size to 1 to minimize latency
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Get actual resolution and FPS (camera may not support requested)
            self.actual_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            print(f"Actual resolution: {self.actual_width}x{self.actual_height} @ {actual_fps:.0f}fps")

            print(f"Warming up camera at {device_path}...")
            for _ in range(10):
                self.video_capture.read()
            print("Camera ready!")

        except Exception:
            pass

    def __del__(self):
        try:
            self.video_capture.release()
        except Exception:
            pass

    def read_image(self):
        ret, frame = self.video_capture.read()
        return frame


class MonochromeCamera(Camera):
    """
    Camera that requests monochrome color effects via v4l2.
    """

    def __init__(self, device_path: str = '/dev/video3', rotation: int = 0):
        super().__init__(device_path, rotation)
        try:
            os.system(f'v4l2-ctl -d {device_path} --set-ctrl=color_effects=1')
        except Exception:
            pass