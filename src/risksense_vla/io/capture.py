"""Video input capture utilities for webcam and file streams."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class CapturedFrame:
    frame_index: int
    timestamp: float
    bgr: np.ndarray
    source_id: str


class VideoInput:
    """Single-stream OpenCV capture with FPS pacing."""

    def __init__(self, source: int | str, width: int = 1280, height: int = 720, target_fps: int = 25):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0

    def open(self) -> None:
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        self._cap = cv2.VideoCapture(src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def stream(self) -> Iterator[CapturedFrame]:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        target_dt = 1.0 / max(1, self.target_fps)
        while True:
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield CapturedFrame(
                frame_index=self._frame_index,
                timestamp=time.time(),
                bgr=frame,
                source_id=str(self.source),
            )
            self._frame_index += 1
            elapsed = time.perf_counter() - t0
            sleep_dt = target_dt - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)


class MultiViewInput:
    """Optional multiview wrapper returning synchronized frame bundles."""

    def __init__(self, sources: list[int | str], width: int = 1280, height: int = 720, target_fps: int = 25):
        self.sources = sources
        self._inputs = [VideoInput(s, width=width, height=height, target_fps=target_fps) for s in sources]

    def open(self) -> None:
        for inp in self._inputs:
            inp.open()

    def close(self) -> None:
        for inp in self._inputs:
            inp.close()

    def stream(self) -> Iterator[list[CapturedFrame]]:
        streams = [inp.stream() for inp in self._inputs]
        while True:
            bundle: list[CapturedFrame] = []
            for stream in streams:
                try:
                    bundle.append(next(stream))
                except StopIteration:
                    return
            yield bundle


def resolve_source(value: str) -> int | str:
    """Treat numeric string as camera id, otherwise path."""
    if value.isdigit():
        return int(value)
    path = Path(value)
    return str(path)
