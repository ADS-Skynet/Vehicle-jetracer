# Copyright (C) 2022 twyleg
import cv2
import numpy as np
import pyrealsense2 as rs


class Camera:
    """
    Camera wrapper using Intel RealSense SDK.
    Captures color frames via pyrealsense2 pipeline.
    Optionally captures aligned depth frames.

    Depth data is returned alongside color from read_frames() and travels
    through the unified image shared memory via vehicle.py → lkas.send_image().
    """

    # Native color resolutions supported by D455 (width, height)
    _SUPPORTED_RESOLUTIONS = [
        (424, 240),
        (480, 270),
        (640, 360),
        (640, 480),
        (848, 480),
        (1280, 720),
        (1280, 800),
    ]

    def __init__(self, device_path: str = '', width: int = 640, height: int = 480,
                 fps: int = 30, rotation: int = 0, enable_depth: bool = False):
        self.width = width
        self.height = height
        self._enable_depth = enable_depth
        self._depth_scale = 0.0
        self.depth_scale = 0.0
        self._align = None

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()

        # If a serial number is provided, bind to that specific device
        if device_path and not device_path.startswith('/dev/'):
            self.rs_config.enable_device(device_path)

        # Pick the closest native resolution that is >= the requested size
        native_w, native_h = self._pick_native_resolution(width, height)
        self._needs_resize = (native_w != width or native_h != height)

        self.rs_config.enable_stream(rs.stream.color, native_w, native_h, rs.format.bgr8, fps)

        if enable_depth:
            # Depth at 640x480 is universally supported on D455
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
            self._align = rs.align(rs.stream.color)

        depth_label = " + depth" if enable_depth else ""
        print(f"Starting RealSense pipeline (native {native_w}x{native_h} @ {fps}fps{depth_label}"
              f"{f', resize to {width}x{height}' if self._needs_resize else ''})...")
        try:
            self.profile = self.pipeline.start(self.rs_config)

            # Get actual stream parameters
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color_profile.get_intrinsics()
            self.actual_width = intrinsics.width
            self.actual_height = intrinsics.height
            print(f"Actual native resolution: {self.actual_width}x{self.actual_height}")

            if enable_depth:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self._depth_scale = depth_sensor.get_depth_scale()
                self.depth_scale = self._depth_scale
                print(f"Depth scale: {self._depth_scale}")

            # Warm up — let auto-exposure settle
            print("Warming up RealSense camera...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            print("Camera ready!")

        except Exception as e:
            print(f"Failed to start RealSense pipeline: {e}")
            raise

    @classmethod
    def _pick_native_resolution(cls, target_w: int, target_h: int):
        """Pick the smallest native resolution that covers the target size."""
        best = None
        best_pixels = float('inf')
        for w, h in cls._SUPPORTED_RESOLUTIONS:
            if w >= target_w and h >= target_h:
                px = w * h
                if px < best_pixels:
                    best = (w, h)
                    best_pixels = px
        if best:
            return best
        # Fallback: pick the largest available resolution
        return cls._SUPPORTED_RESOLUTIONS[-1]

    def __del__(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def read_image(self):
        """Read a color frame only."""
        frames = self.pipeline.wait_for_frames()

        if self._enable_depth and self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        frame = np.asanyarray(color_frame.get_data())
        if self._needs_resize:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def read_frames(self, timestamp: float = 0.0, frame_id: int = 0):
        """
        Read color and depth frames together.

        Depth is aligned to color and resized to match color dimensions.

        Returns:
            (color_image, depth_array) — depth_array is None when depth is disabled.
        """
        frames = self.pipeline.wait_for_frames()
        depth_array = None

        if self._enable_depth and self._align is not None:
            frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_array = np.asanyarray(depth_frame.get_data())
                # Resize depth to match target color dimensions if needed
                if self._needs_resize:
                    depth_array = cv2.resize(
                        depth_array, (self.width, self.height),
                        interpolation=cv2.INTER_NEAREST
                    )

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        if self._needs_resize:
            color_image = cv2.resize(color_image, (self.width, self.height))
        return color_image, depth_array
