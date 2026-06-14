import mmap
import os
import struct
import time


class RtControlShmWriter:
    """Minimal Python writer for the LKAS -> DCAS ring in rt_control_shm."""

    SHM_PATH = "/dev/shm/rt_control_shm"
    MAGIC = 0x5243544C
    VERSION = 2
    RING_CAPACITY = 64
    LAYOUT_SIZE = 4680
    LKAS_RING_OFFSET = 1560
    RING_ENTRIES_OFFSET = 8
    SAMPLE_SIZE = 24

    def __init__(self, shm_path: str | None = None):
        self.shm_path = shm_path or self.SHM_PATH
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None

    def open(self) -> None:
        flags = os.O_RDWR | os.O_CREAT
        self.fd = os.open(self.shm_path, flags, 0o666)

        current_size = os.fstat(self.fd).st_size
        if current_size < self.LAYOUT_SIZE:
            os.ftruncate(self.fd, self.LAYOUT_SIZE)

        self.mm = mmap.mmap(self.fd, self.LAYOUT_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
        self._initialize_if_needed()

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write_lkas(self, throttle: float, steering: float) -> None:
        if self.mm is None:
            return

        ring_offset = self.LKAS_RING_OFFSET
        head = struct.unpack_from("<I", self.mm, ring_offset)[0]
        entry_offset = ring_offset + self.RING_ENTRIES_OFFSET + ((head % self.RING_CAPACITY) * self.SAMPLE_SIZE)
        timestamp_us = int(time.monotonic() * 1_000_000)

        throttle = max(0.0, min(1.0, float(throttle)))
        steering = max(-1.0, min(1.0, float(steering)))

        struct.pack_into("<QffI", self.mm, entry_offset, timestamp_us, throttle, steering, 0)
        struct.pack_into("<I", self.mm, ring_offset, (head + 1) & 0xFFFFFFFF)

    def _initialize_if_needed(self) -> None:
        if self.mm is None:
            return

        magic = struct.unpack_from("<I", self.mm, 0)[0]
        if magic == self.MAGIC:
            return

        self.mm[:] = b"\x00" * self.LAYOUT_SIZE
        struct.pack_into("<IIII", self.mm, 0, self.MAGIC, self.VERSION, self.RING_CAPACITY, 0)
