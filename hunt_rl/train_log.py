"""训练脚本可选：将 stdout / stderr 同步写入日志文件（控制台仍正常输出）。"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


class _TeeIO:
    """多路写入并 flush，供 print 与 traceback 使用。"""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = len(s)
        for st in self._streams:
            st.write(s)
            st.flush()
        return n

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()

    @property
    def encoding(self) -> str:
        enc = getattr(self._streams[0], "encoding", None)
        return enc if isinstance(enc, str) else "utf-8"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streams[0], name)


@contextmanager
def tee_stdout_stderr(log_path: str | Path | None) -> Iterator[None]:
    """
    若 `log_path` 非空，则同时把 stdout、stderr 复制到该文件（UTF-8）；退出时恢复。
    """
    if not log_path:
        yield
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="\n")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _TeeIO(old_out, f)  # type: ignore[assignment]
        sys.stderr = _TeeIO(old_err, f)  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()
