"""Minimal client for the NDI Polaris Vega (Combined API over TCP port 8765).

Only what is needed for passive-tool tracking: INIT, tool loading
(PHRQ/PVWR/PINIT/PENA), TSTART/TSTOP and TX text-mode frames, including
passive stray markers (TX option 0x1000).
"""
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_HOST = '192.168.3.3'
DEFAULT_PORT = 8765
ROM_DIR = Path(__file__).parent / 'roms'


def crc16(data: bytes) -> int:
    """CRC-16 used by NDI replies (poly 0x8005 reflected, init 0)."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


class VegaError(RuntimeError):
    pass


@dataclass
class ToolPose:
    handle: str
    name: str
    status: str                  # 'OK', 'MISSING' or 'DISABLED'
    quat: Optional[np.ndarray] = None      # (q0, qx, qy, qz), q0 scalar
    pos: Optional[np.ndarray] = None      # mm, camera frame
    error: float = float('nan')  # RMS marker fit error, mm
    frame: int = 0

    @property
    def visible(self) -> bool:
        return self.status == 'OK'

    def rotation(self) -> np.ndarray:
        return quat_to_rot(self.quat)

    def tip(self, offset) -> np.ndarray:
        """Camera-frame position of a point fixed in the tool frame (mm)."""
        return self.rotation() @ np.asarray(offset, float) + self.pos


@dataclass
class Frame:
    t: float                     # host time.time() at receipt
    tools: list = field(default_factory=list)
    strays: Optional[np.ndarray] = None   # (N, 3) mm, passive markers not assigned to a tool
    system_status: int = 0
    raw: str = ''


def quat_to_rot(q) -> np.ndarray:
    # TX quaternions have 4 decimals (|q| ~ 0.9999): normalise or R is not a rotation
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _num(s: str, scale: float) -> float:
    return int(s) / scale


class Vega:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.names = {}          # handle -> tool name
        self._buf = b''

    # -- low level -------------------------------------------------------
    def command(self, cmd: str, timeout=10.0) -> str:
        self.sock.sendall(cmd.encode() + b'\r')
        end = time.time() + timeout
        while b'\r' not in self._buf:
            if time.time() > end:
                raise VegaError(f'timeout waiting for reply to {cmd[:20]!r}')
            self._buf += self.sock.recv(65536)
        raw, self._buf = self._buf.split(b'\r', 1)
        body, crc = raw[:-4], raw[-4:]
        if int(crc, 16) != crc16(body):
            raise VegaError(f'CRC mismatch in reply to {cmd[:20]!r}: {raw!r}')
        reply = body.decode()
        if reply.startswith('ERROR'):
            raise VegaError(f'{cmd[:20]!r} -> {reply}')
        return reply

    # -- setup -----------------------------------------------------------
    def init(self):
        self.command('INIT ', timeout=30)

    def info(self) -> str:
        return self.command('VER 4')

    def load_rom(self, path) -> str:
        """Load a passive tool definition; returns its port handle."""
        path = Path(path)
        if not path.exists():
            path = ROM_DIR / (path.name if path.suffix else path.name + '.rom')
        rom = path.read_bytes()
        rom += b'\0' * (-len(rom) % 64)
        handle = self.command('PHRQ *********1****')[:2]
        for off in range(0, len(rom), 64):
            self.command(f'PVWR {handle}{off:04X}{rom[off:off + 64].hex().upper()}')
        self.command(f'PINIT {handle}')
        self.command(f'PENA {handle}D')
        self.names[handle] = path.stem
        return handle

    TRACK_RATES = {20: 0, 30: 1, 60: 2}    # Hz -> Param.Tracking.Track Frequency

    def start(self, rate=60):
        """Start tracking at 20, 30 or 60 Hz (camera frames are always 60 Hz)."""
        self.command(f'SET Param.Tracking.Track Frequency={self.TRACK_RATES[rate]}')
        self.command('TSTART ', timeout=30)

    def stop(self):
        try:
            self.command('TSTOP ', timeout=30)
        except (VegaError, OSError):
            pass

    def close(self):
        self.stop()
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- tracking --------------------------------------------------------
    def read(self, strays=False) -> Frame:
        """Latest frame: tool transforms (TX 0x0001), plus passive stray markers
        (0x1000) if asked. The stray layout is unverified: if it does not parse,
        frame.strays is None and frame.raw holds the reply."""
        reply = self.command('TX 1001' if strays else 'TX 0001')
        return self._parse_tx(reply, time.time(), strays)

    def _parse_tx(self, r: str, t: float, strays=True) -> Frame:
        frame = Frame(t=t, raw=r)
        n = int(r[:2], 16)
        i = 2
        for _ in range(n):
            h = r[i:i + 2]; i += 2
            name = self.names.get(h, h)
            if r.startswith('MISSING', i) or r.startswith('DISABLED', i):
                status = 'MISSING' if r[i] == 'M' else 'DISABLED'
                i += len(status)
                pose = ToolPose(h, name, status)
                if status == 'MISSING':
                    i += 8                                   # port status
                    pose.frame = int(r[i:i + 8], 16); i += 8
            else:
                q = [_num(r[i + 6 * k:i + 6 * k + 6], 1e4) for k in range(4)]; i += 24
                p = [_num(r[i + 7 * k:i + 7 * k + 7], 1e2) for k in range(3)]; i += 21
                err = _num(r[i:i + 6], 1e4); i += 6
                i += 8                                       # port status
                fr = int(r[i:i + 8], 16); i += 8
                pose = ToolPose(h, name, 'OK', np.array(q), np.array(p), err, fr)
            frame.tools.append(pose)
            if r[i:i + 1] == '\n':
                i += 1
        rest = r[i:].replace('\n', '')
        if not strays:
            frame.system_status = int(rest[:4], 16)
            return frame
        try:
            frame.strays, frame.system_status = self._parse_strays(rest)
        except ValueError:
            frame.strays = None
        return frame

    @staticmethod
    def _parse_strays(rest):
        m = int(rest[:2], 16)
        j = 2 + (m + 3) // 4                                 # skip out-of-volume bits
        pts = [[_num(rest[j + 21 * k + 7 * a:j + 21 * k + 7 * a + 7], 1e2) for a in range(3)]
               for k in range(m)]
        j += 21 * m
        return np.array(pts).reshape(-1, 3), int(rest[j:j + 4], 16)
