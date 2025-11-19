"""
Vehicle Module Constants

Module-specific constants for the Jetracer vehicle.
Hardware-specific settings and runtime defaults.
"""


class Hardware:
    """Hardware configuration constants."""

    CAMERA_DEVICE = '/dev/video4'  # Default camera device path


class Communication:
    """Communication configuration constants."""

    BROKER_STATUS_URL = 'tcp://localhost:5562'  # LKAS broker status URL
    PUBLISH_STATE_HZ = 10  # Vehicle state publish rate (Hz)


class Control:
    """Control configuration constants."""

    DEFAULT_THROTTLE = 0.0  # Default throttle value (0 = stopped)


# Export all constants
__all__ = [
    'Hardware',
    'Communication',
    'Control',
]
