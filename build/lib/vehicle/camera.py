# Copyright (C) 2022 twyleg
import os
import cv2


class Camera:
    """
    Simple camera wrapper using OpenCV VideoCapture.
    Sets device rotation via v4l2-ctl when available.
    """

    def __init__(self, device_path: str = '/dev/video4', rotation: int = 0):
        self.video_capture = cv2.VideoCapture(device_path)
        try:
            os.system(f'v4l2-ctl -d {device_path} --set-ctrl=rotate={rotation}')
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