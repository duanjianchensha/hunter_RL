"""将 RGB 帧流写入 MP4（依赖 imageio + imageio-ffmpeg）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def mp4_writer_requires_msg() -> str:
    return "录制 MP4 需要安装：pip install imageio imageio-ffmpeg（或 pip install -e \".[viz]\"）"


class Mp4Recorder:
    """按配置 fps 追加 (H,W,3) RGB 帧，结束时 close。"""

    def __init__(self, path: str | Path, fps: float) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError as e:
            raise SystemExit(mp4_writer_requires_msg()) from e
        self._imageio = imageio
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Any = imageio.get_writer(
            str(path),
            fps=float(fps),
            codec="libx264",
            quality=8,
        )
        self.path = path
        self.frame_count = 0

    def append(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        x = np.ascontiguousarray(frame)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"期望 RGB 帧 shape (H,W,3)，收到 {x.shape}")
        self._writer.append_data(x)
        self.frame_count += 1

    def close(self) -> None:
        self._writer.close()
