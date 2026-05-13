import os
import time
import json
import cv2
import argparse
from common.config import ConfigManager
from .vehicle import Vehicle
from .constants import Hardware, Communication, Control


def main():
    parser = argparse.ArgumentParser(description="Vehicle-jetracer main loop with LKAS")
    parser.add_argument('--device', default=Hardware.CAMERA_DEVICE,
                       help=f"Camera device path (default: {Hardware.CAMERA_DEVICE})")
    parser.add_argument('--publish-state-hz', type=float, default=Communication.PUBLISH_STATE_HZ,
                       help=f"Vehicle state publish rate (default: {Communication.PUBLISH_STATE_HZ} Hz)")
    parser.add_argument('--keepalive', action='store_true', default=False,
                       help="Enable camera be alive even with pausing as sending image to LKAS (default: False)")
    parser.add_argument('--use-cpp-actuator', action='store_true', default=False,
                       help="Disable Python actuation (C++ actuator controls hardware)")
    args = parser.parse_args()

    # Load skynet-common config for LKAS settings
    cfg = ConfigManager.load()

    print("=" * 60)
    print("Vehicle-Jetracer with LKAS")
    print("=" * 60)
    print(f"  Camera: {args.device}")
    print(f"  Status Pub Port: {cfg.communication.zmq_vehicle_status_port}")
    print(f"  Action Sub Port: {cfg.communication.zmq_action_forward_port}")
    print(f"  Parameter Sub Port: {cfg.communication.zmq_param_servers_port}")
    print(f"  Image size: {cfg.camera.width}x{cfg.camera.height}")
    print(f"  Throttle Base: {cfg.throttle_policy.base}")
    print(f"  State publish rate: {args.publish_state_hz} Hz")
    print(f"  Actuator mode: {'C++ rt-actuator' if args.use_cpp_actuator else 'Python legacy'}")
    print("=" * 60)
    print("Note: LKAS broker handles frame/detection broadcasting to viewers")
    print("      Send 'stop' or 'resume' actions from viewer to control vehicle")
    print("=" * 60)

    v = Vehicle(
        device=args.device,
        status_pub_port=cfg.communication.zmq_vehicle_status_port,
        action_sub_port=cfg.communication.zmq_action_forward_port,
        param_sub_port=cfg.communication.zmq_param_servers_port,
        publish_state_hz=args.publish_state_hz,
        throttle_base=cfg.throttle_policy.base,
        image_shm_name=cfg.communication.image_shm_name,
        detection_shm_name=cfg.communication.detection_shm_name,
        control_shm_name=cfg.communication.control_shm_name,
        image_width=cfg.camera.width,
        image_height=cfg.camera.height,
        keepalive_camera=args.keepalive,
        use_cpp_actuator=args.use_cpp_actuator,
    )
    print(f"Starting vehicle loop. Throttle fixed = {v.throttle}")
    print("Press Ctrl+C to stop")
    v.run()

if __name__ == "__main__":
    main()
