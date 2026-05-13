# Copyright (C) 2022 twyleg
import time
import json
import zmq
import cv2
from lkas import LKASClient as LKAS

# Try to reuse common VehicleStatusPublisher if available
try:
    from common.communication.zmq_broadcast import (
        VehicleStatusPublisher,
        ActionSubscriber,
        ParameterSubscriber
    )
    _HAS_COMMON_PUB = True
except Exception:
    _HAS_COMMON_PUB = False

# Import camera from local module
from .camera import Camera
from .rt_control_shm_writer import RtControlShmWriter


class Vehicle:
    """
    Vehicle loop:
      - read images from Camera
      - send images to LKAS via shared memory
      - receive control commands from LKAS via shared memory
      - publish vehicle state to LKAS broker (optional)
      - apply steering (throttle is fixed by config)
    """

    def __init__(
        self,
        device: str = '/dev/video4',
        status_pub_port: int = 5562,
        action_sub_port: int = 5561,
        param_sub_port: int = 5560,
        jpeg_quality: int = 80,
        publish_state_hz: int = 10,
        throttle_base: float = 0.15,
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        control_shm_name: str = "control_commands",
        image_width: int = 640,
        image_height: int = 480,
        keepalive_camera: bool = False,
        use_cpp_actuator: bool = False,
    ):
        self.camera = Camera(device_path=device, width=image_width, height=image_height)
        self.status_pub_url = f"tcp://localhost:{status_pub_port}"
        self.action_sub_url = f"tcp://localhost:{action_sub_port}"
        self.param_sub_url = f"tcp://localhost:{param_sub_port}"
        self.steering = 0.0
        self.throttle = float(throttle_base)
        self.throttle_paused = float(throttle_base)
        self.throttle_base = float(throttle_base)
        self.brake = 0.0
        self.image_width = image_width
        self.image_height = image_height
        self.keepalive_camera = keepalive_camera
        self.use_cpp_actuator = use_cpp_actuator
        self.status = "READY"
        self.rt_control_writer = None

        # Initialize NvidiaRacecar for real hardware actuation (optional)
        self.car = None
        if not self.use_cpp_actuator:
            from jetracer.nvidia_racecar import NvidiaRacecar

            print("Initializing NvidiaRacecar for real actuation...")
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            print("✓ NvidiaRacecar initialized")
        else:
            print("✓ C++ actuator mode enabled (Python actuation disabled)")
            self.rt_control_writer = RtControlShmWriter()
            self.rt_control_writer.open()
            self.rt_control_writer.write_lkas(0.0, 0.0)
            print("✓ rt_control_shm LKAS writer initialized")

        # Initialize LKAS with shared memory
        print(f"Initializing LKAS with shared memory...")
        print(f"  Image SHM: {image_shm_name}")
        print(f"  Detection SHM: {detection_shm_name}")
        print(f"  Control SHM: {control_shm_name}")
        print(f"  Image size: {image_width}x{image_height}")

        self.lkas = LKAS(
            image_shm_name=image_shm_name,
            detection_shm_name=detection_shm_name,
            control_shm_name=control_shm_name,
        )
        print("✓ LKAS initialized")

        self.context = zmq.Context()

        # Vehicle status publisher (CONNECT to LKAS broker, no BIND)
        if _HAS_COMMON_PUB:
            self.state_pub = VehicleStatusPublisher(lkas_broker_url=self.status_pub_url)
        else:
            self.state_pub = self.context.socket(zmq.PUB)
            self.state_pub.connect(self.status_pub_url)
            self.state_pub.setsockopt(zmq.SNDHWM, 5)
        print(f"✓ Vehicle status publisher connected to {self.status_pub_url}")

        self.jpeg_quality = int(jpeg_quality)
        self.running = False
        self.publish_state_hz = float(publish_state_hz)
        self.paused = False if not self.keepalive_camera else True  # Stop/resume state

        # Action subscriber (receive commands from viewer via LKAS broker)
        if _HAS_COMMON_PUB:
            self.action_sub = ActionSubscriber(bind_url=self.action_sub_url, connect_mode=True)
            self.action_sub.register_action('stop', self._on_pause)
            self.action_sub.register_action('resume', self._on_resume)
            self.action_sub.register_action('pause', self._on_pause)  # Alias
            self.action_sub.register_action('reset', self._on_reset)
            print(f"✓ Action subscriber connected to {self.action_sub_url}")
        else:
            self.action_sub = None
            print("⚠ ActionSubscriber not available, stop/resume disabled")

        # Parameter subscriber (receive throttle updates from viewer via LKAS broker)
        if _HAS_COMMON_PUB:
            self.param_sub = ParameterSubscriber(category='vehicle', broker_url=self.param_sub_url)
            self.param_sub.register_callback(self._on_parameter_update)
        else:
            self.param_sub = None
            print("⚠ ParameterSubscriber not available, throttle updates disabled")

        # Give ZMQ time to establish connection (slow joiner problem)
        time.sleep(0.2)

    def _on_pause(self):
        if not self.paused:
            self.paused = True
            self.throttle_paused = self.throttle
            self._set_throttle(0.0)
            self._publish_lkas_raw_control()
            self._update_vehicle_state()
            self.status = "PAUSED"

            print("\n[vehicle] PAUSED - Control disabled")

    def _on_resume(self):
        if self.paused:
            self.paused = False
            self._set_throttle(self.throttle_paused)
            self._publish_lkas_raw_control()
            self._update_vehicle_state()

            print("\n[vehicle] RESUMED - Control enabled")

    def _on_reset(self):
        """Reset vehicle state: set steering to 0 and pause."""
        self._set_steering(0.0)
        self._publish_lkas_raw_control()
        if not self.paused:
            # self._set_throttle(self.throttle_base)
            self._on_pause()
        # else:
        #     self.throttle_paused = self.throttle_base

        print("\n[vehicle] RESET - Steering set to 0, vehicle paused")

    def _on_parameter_update(self, parameter: str, value: float):
        """
        Handle parameter updates from viewer (via LKAS broker).

        Args:
            parameter: Parameter name (e.g., 'throttle')
            value: New parameter value
        """
        if parameter == 'throttle':
            self._set_throttle(value)
            self._publish_lkas_raw_control()

    def _send_state(self, frame_id: int):
        state = {
            'steering': float(self.steering),
            'throttle': float(self.throttle),
            'brake': float(self.brake),
            'speed_kmh': 0.0,
            'position': None,
            'rotation': None,
            'paused': self.paused,
        }

        try:
            if _HAS_COMMON_PUB:
                try:
                    self.state_pub.send_state(state)
                except Exception:
                    try:
                        self.state_pub.pub_socket.send_multipart([b'vehicle_status', json.dumps(state).encode('utf-8')], flags=zmq.NOBLOCK)
                    except Exception:
                        pass
            else:
                self.state_pub.send_multipart([b'vehicle_status', json.dumps(state).encode('utf-8')], flags=zmq.NOBLOCK)
        except Exception:
            pass

    def _set_throttle(self, throttle: float):
        self.throttle = max(0.0, min(1.0, throttle))

    def _set_steering(self, steering: float):
        self.steering = max(-0.9, min(0.85, steering))


    def _update_vehicle_state(self):
        if self.use_cpp_actuator or self.car is None:
            return
        # print(f"\r[vehicle] Applying control - Throttle: {self.throttle:.2f}, Steering: {self.steering:.2f}", end="", flush=True)
        self.car.throttle = -self.throttle
        self.car.steering = -self.steering

    def _publish_lkas_raw_control(self):
        if self.rt_control_writer is None:
            return
        self.rt_control_writer.write_lkas(self.throttle, self.steering)

    def _apply_control_from_lkas(self):
        """
        Apply or publish raw control commands from LKAS.

        Legacy mode keeps Python hardware actuation behavior. C++ actuator mode
        publishes LKAS raw throttle/steering into rt_control_shm for DCAS.
        """
        control = self.lkas.get_control(timeout=0.1)
        if control is not None:
            self._set_steering(control.steering)
            if self.use_cpp_actuator:
                self._set_throttle(getattr(control, "throttle", self.throttle))
                self._publish_lkas_raw_control()

    def run(self):
        self.running = True

        frame_id = 0
        last_state_ts = 0.0
        last_fps_ts = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        print("\n[vehicle] Starting main loop...")
        print("[vehicle] Sending frames to LKAS via shared memory...")
        print("[vehicle] LKAS broker handles broadcasting to viewers")

        try:
            while self.running:
                # Publish & Subscribe ZMQ messages
                if self.action_sub:
                    self.action_sub.poll()

                # Poll for parameter updates (throttle from viewer)
                if self.param_sub:
                    self.param_sub.poll()

                now = time.time()
                if now - last_state_ts >= 1.0 / self.publish_state_hz:
                    self._send_state(frame_id)
                    last_state_ts = now

                # If paused, skip reading/applying control
                if self.paused and not self.keepalive_camera:
                    time.sleep(0.5)
                    continue

                # Read frame from camera
                frame = self.camera.read_image()
                if frame is None:
                    # print("[vehicle] Warning: Failed to read frame from camera")
                    continue

                # Send frame to LKAS via shared memory
                timestamp = time.time()
                self.lkas.send_image(frame, timestamp, frame_id)

                # If keepalive is enabled, the loop may be fallout to here, so we should check pause state again at here
                if not self.paused :
                    self._apply_control_from_lkas()
                    self._update_vehicle_state()

                # FPS calculation and status update
                fps_frame_count += 1
                if now - last_fps_ts >= 1.0:
                    current_fps = fps_frame_count / (now - last_fps_ts)
                    self.status = f"{current_fps:.1f} FPS"
                    print(f"\r[vehicle] {self.status} | Frame {frame_id}", end="", flush=True)
                    fps_frame_count = 0
                    last_fps_ts = now

                # Increment frame ID
                frame_id += 1

                time.sleep(0.016)  # 60 FPS (was 0.015 = 66 FPS)

        except KeyboardInterrupt:
            print("[vehicle] Interrupted by user")
        finally:
            self.close()

    def close(self):
        self.running = False
        print("[vehicle] Closing...")

        try:
            if self.car is not None:
                self.car.throttle = 0.0
                self.car.steering = 0.0
                print("✓ Vehicle stopped (safety)")
        except Exception:
            pass

        try:
            if self.rt_control_writer is not None:
                self.rt_control_writer.write_lkas(0.0, 0.0)
                self.rt_control_writer.close()
                print("✓ rt_control_shm LKAS writer closed")
        except Exception:
            pass

        try:
            self.lkas.close()
            print("✓ LKAS closed")
        except Exception:
            pass

        try:
            if self.action_sub:
                self.action_sub.close()
                print("✓ Action subscriber closed")
        except Exception:
            pass

        try:
            if self.param_sub:
                self.param_sub.close()
                print("✓ Parameter subscriber closed")
        except Exception:
            pass

        try:
            if _HAS_COMMON_PUB:
                self.state_pub.close()
            self.context.term()
            print("✓ ZMQ state publisher closed")
        except Exception:
            pass
        print("✓ Vehicle closed")
