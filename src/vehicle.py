# Copyright (C) 2022 twyleg
import time
import json
import zmq
import cv2
from lkas import LKAS
from jetracer.nvidia_racecar import NvidiaRacecar

# Try to reuse common VehicleStatusPublisher if available
try:
    from skynet_common.communication.zmq_broadcast import VehicleStatusPublisher, ActionSubscriber
    _HAS_COMMON_PUB = True
except Exception:
    _HAS_COMMON_PUB = False

# Import camera from local module
from .camera import Camera


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
        broker_status_url: str = 'tcp://localhost:5562',
        action_url: str = 'tcp://localhost:5561',
        jpeg_quality: int = 80,
        publish_state_hz: int = 10,
        throttle: float = 0.15,
        image_shm_name: str = "camera_feed",
        detection_shm_name: str = "detection_results",
        control_shm_name: str = "control_commands",
        image_width: int = 640,
        image_height: int = 480,
    ):
        self.camera = Camera(device_path=device)
        self.steering = 0.0
        self.throttle = float(throttle)
        self.brake = 0.0
        self.image_width = image_width
        self.image_height = image_height

        # Initialize NvidiaRacecar for real hardware actuation
        print("Initializing NvidiaRacecar for real actuation...")
        self.car = NvidiaRacecar()
        self.car.steering = 0.0
        self.car.throttle = 0.0
        print("✓ NvidiaRacecar initialized")

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
            image_shape=(image_height, image_width, 3),
        )
        print("✓ LKAS initialized")

        self.context = zmq.Context()

        # Vehicle status publisher (CONNECT to LKAS broker, no BIND)
        if _HAS_COMMON_PUB:
            self.state_pub = VehicleStatusPublisher(lkas_broker_url=broker_status_url)
        else:
            self.state_pub = self.context.socket(zmq.PUB)
            self.state_pub.connect(broker_status_url)
            self.state_pub.setsockopt(zmq.SNDHWM, 5)
        print(f"✓ Vehicle status publisher connected to {broker_status_url}")

        self.jpeg_quality = int(jpeg_quality)
        self.running = False
        self.publish_state_hz = float(publish_state_hz)
        self.paused = False  # Stop/resume state

        # Action subscriber (receive commands from viewer via LKAS broker)
        if _HAS_COMMON_PUB:
            self.action_sub = ActionSubscriber(bind_url=action_url, connect_mode=True)
            self.action_sub.register_action('stop', self._on_stop)
            self.action_sub.register_action('resume', self._on_resume)
            self.action_sub.register_action('pause', self._on_stop)  # Alias
            print(f"✓ Action subscriber connected to {action_url}")
        else:
            self.action_sub = None
            print("⚠ ActionSubscriber not available, stop/resume disabled")

        # Give ZMQ time to establish connection (slow joiner problem)
        time.sleep(0.2)

    def _on_stop(self):
        if not self.paused:
            self.paused = True
            self.steering = 0.0
            self.throttle = 0.0
            self.car.steering = 0.0
            self.car.throttle = 0.0
            print("\n[vehicle] STOPPED - Control disabled")

    def _on_resume(self):
        if self.paused:
            self.paused = False
            print("\n[vehicle] RESUMED - Control enabled")

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

    def _apply_control_from_lkas(self):
        if self.paused:
            return

        control = self.lkas.get_control(timeout=0.1)
        if control is not None:
            self.steering = max(-1.0, min(1.0, control.steering))
            self.brake = control.brake

            # Apply to real hardware
            self.car.steering = -self.steering * 10
            if self.brake > 0.5:
                self.car.throttle = 0.0
            else:
                self.car.throttle = -self.throttle

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
                frame = self.camera.read_image()
                if frame is None:
                    print("[vehicle] Warning: Failed to read frame from camera")
                    continue

                if frame.shape[0] != self.image_height or frame.shape[1] != self.image_width:
                    frame = cv2.resize(frame, (self.image_width, self.image_height))

                timestamp = time.time()
                self.lkas.send_image(frame, timestamp, frame_id)

                self._apply_control_from_lkas()

                if self.action_sub:
                    self.action_sub.poll()

                now = time.time()
                if now - last_state_ts >= 1.0 / self.publish_state_hz:
                    self._send_state(frame_id)
                    last_state_ts = now

                fps_frame_count += 1
                if now - last_fps_ts >= 1.0:
                    current_fps = fps_frame_count / (now - last_fps_ts)
                    status = "PAUSED" if self.paused else f"{current_fps:.1f} FPS"
                    print(f"\r[vehicle] {status} | Frame {frame_id}", end="", flush=True)
                    fps_frame_count = 0
                    last_fps_ts = now

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
            self.car.throttle = 0.0
            self.car.steering = 0.0
            print("✓ Vehicle stopped (safety)")
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
            if _HAS_COMMON_PUB:
                self.state_pub.close()
            self.context.term()
            print("✓ ZMQ state publisher closed")
        except Exception:
            pass
        print("✓ Vehicle closed")
