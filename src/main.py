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
    cfg = ConfigManager.load()
    # cfg.communication = cfg.communication
    # cfg_cam = cfg.camera
    # cfg_thr = cfg.throttle_policy

    parser = argparse.ArgumentParser(description="Vehicle-jetracer main loop with LKAS")
    parser.add_argument('--device', default=Hardware.CAMERA_DEVICE,
                       help=f"Camera device path (default: {Hardware.CAMERA_DEVICE})")
    # parser.add_argument('--broker-status', default=Communication.BROKER_STATUS_URL,
    #                    help=f"ZMQ URL for LKAS broker status (default: {Communication.BROKER_STATUS_URL})")
    # parser.add_argument('--action-url', default=f"tcp://localhost:{comm.zmq_action_port + 3}",
    #                    help=f"ZMQ URL for action commands (default: tcp://localhost:{comm.zmq_action_port + 3})")
    parser.add_argument('--publish-state-hz', type=float, default=Communication.PUBLISH_STATE_HZ,
                       help=f"Vehicle state publish rate (default: {Communication.PUBLISH_STATE_HZ} Hz)")
    # parser.add_argument('--throttle', type=float, default=Control.DEFAULT_THROTTLE,
    #                    help=f"Fixed throttle value (default: {Control.DEFAULT_THROTTLE})")
    # parser.add_argument('--throttle-base', type=float, default=throttle_cfg.base,
    #                    help=f"Base throttle value for reset (default: {throttle_cfg.base})")
    args = parser.parse_args()

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
    )
    print(f"Starting vehicle loop. Throttle fixed = {v.throttle}")
    print("Press Ctrl+C to stop")
    v.run()

if __name__ == "__main__":
    main()