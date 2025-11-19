import os
import time
import json
import cv2
import argparse
import yaml
from pathlib import Path
from skynet_common.config import ConfigManager
from .vehicle import Vehicle


def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load YAML config from Jetracer root config.yaml.
    Returns defaults when file is missing or invalid.
    """
    defaults = {
        'device': '/dev/video4',
        'broker_status': 'tcp://localhost:5562',
        'publish_state_hz': 10,
        'throttle': -0.15,
    }
    try:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.yaml"
        config_path = Path(config_path)
        if not config_path.exists():
            return defaults
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return defaults

def main():
    # Load skynet-common config for LKAS settings
    skynet_config = ConfigManager.load()
    comm = skynet_config.communication
    camera_cfg = skynet_config.camera

    # Load vehicle-specific config
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Vehicle-jetracer main loop with LKAS")
    parser.add_argument('--device', default=cfg.get('device'),
                       help=f"Camera device path (default: {cfg.get('device')})")
    parser.add_argument('--broker-status', default=cfg.get('broker_status'),
                       help=f"ZMQ URL for LKAS broker status (default: {cfg.get('broker_status')})")
    parser.add_argument('--action-url', default=f"tcp://localhost:{comm.zmq_action_port + 3}",
                       help=f"ZMQ URL for action commands (default: tcp://localhost:{comm.zmq_action_port + 3})")
    parser.add_argument('--publish-state-hz', type=float, default=cfg.get('publish_state_hz'),
                       help=f"Vehicle state publish rate (default: {cfg.get('publish_state_hz')} Hz)")
    parser.add_argument('--throttle', type=float, default=cfg.get('throttle'),
                       help=f"Fixed throttle value (default: {cfg.get('throttle')})")
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