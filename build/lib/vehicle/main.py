import os
import time
import json
import cv2
import argparse
from skynet_common.config import ConfigManager
from .vehicle import Vehicle
from .constants import Hardware, Communication, Control


def main():
    # Load skynet-common config for LKAS settings
    skynet_config = ConfigManager.load()
    comm = skynet_config.communication
    camera_cfg = skynet_config.camera

    parser = argparse.ArgumentParser(description="Vehicle-jetracer main loop with LKAS")
    parser.add_argument('--device', default=Hardware.CAMERA_DEVICE,
                       help=f"Camera device path (default: {Hardware.CAMERA_DEVICE})")
    parser.add_argument('--broker-status', default=Communication.BROKER_STATUS_URL,
                       help=f"ZMQ URL for LKAS broker status (default: {Communication.BROKER_STATUS_URL})")
    parser.add_argument('--action-url', default=f"tcp://localhost:{comm.zmq_action_port + 3}",
                       help=f"ZMQ URL for action commands (default: tcp://localhost:{comm.zmq_action_port + 3})")
    parser.add_argument('--publish-state-hz', type=float, default=Communication.PUBLISH_STATE_HZ,
                       help=f"Vehicle state publish rate (default: {Communication.PUBLISH_STATE_HZ} Hz)")
    parser.add_argument('--throttle', type=float, default=Control.DEFAULT_THROTTLE,
                       help=f"Fixed throttle value (default: {Control.DEFAULT_THROTTLE})")
    args = parser.parse_args()

    print("=" * 60)
    print("Vehicle-Jetracer with LKAS")
    print("=" * 60)
    print(f"  Camera: {args.device}")
    print(f"  Broker Status URL: {args.broker_status}")
    print(f"  Action URL: {args.action_url}")
    print(f"  Image size: {camera_cfg.width}x{camera_cfg.height}")
    print(f"  Throttle: {args.throttle}")
    print(f"  State publish rate: {args.publish_state_hz} Hz")
    print("=" * 60)
    print("Note: LKAS broker handles frame/detection broadcasting to viewers")
    print("      Send 'stop' or 'resume' actions from viewer to control vehicle")
    print("=" * 60)

    v = Vehicle(
        device=args.device,
        broker_status_url=args.broker_status,
        action_url=args.action_url,
        publish_state_hz=args.publish_state_hz,
        throttle=args.throttle,
        image_shm_name=comm.image_shm_name,
        detection_shm_name=comm.detection_shm_name,
        control_shm_name=comm.control_shm_name,
        image_width=camera_cfg.width,
        image_height=camera_cfg.height,
    )
    print(f"Starting vehicle loop. Throttle fixed = {v.throttle}")
    print("Press Ctrl+C to stop")
    v.run()

if __name__ == "__main__":
    main()