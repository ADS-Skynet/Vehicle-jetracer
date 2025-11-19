"""
Vehicle module for JetRacer with LKAS integration.

This module provides the Vehicle class that integrates:
- Camera capture
- LKAS control system
- NvidiaRacecar hardware interface
- ZMQ communication for status and actions
"""

from .vehicle import Vehicle
from .camera import Camera, MonochromeCamera
from .main import main

__version__ = "0.1.0"
__all__ = ["Vehicle", "Camera", "MonochromeCamera", "main"]
