from __future__ import annotations

from threading import Event, Lock, Thread
from typing import Callable, Optional


class BackgroundOffloadWorker:
    """
    最小后台 offload worker。

    当前不直接负责 GPU<->PIM DMA，只负责在独立线程里周期性推进：
    - background ready callback
    - prepared tier 的 background prebuild / activation

    这样 decode 主线程不再是唯一能推动后台迁移状态前进的入口。
    """

    def __init__(
        self,
        tick_fn: Callable[[], int],
        *,
        poll_interval_seconds: float = 0.005,
        auto_start: bool = True,
    ) -> None:
        self._tick_fn = tick_fn
        self.poll_interval_seconds = max(0.0, float(poll_interval_seconds))
        self._stop_event = Event()
        self._stats_lock = Lock()
        self._thread: Optional[Thread] = None
        self.ticks = 0
        self.work_ticks = 0
        self.last_work_items = 0
        if auto_start:
            self.start()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._run,
            name="background-offload-worker",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            work_items = int(self._tick_fn())
            with self._stats_lock:
                self.ticks += 1
                self.last_work_items = work_items
                if work_items > 0:
                    self.work_ticks += 1
            if self.poll_interval_seconds > 0:
                self._stop_event.wait(self.poll_interval_seconds)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def reset_counters(self) -> None:
        with self._stats_lock:
            self.ticks = 0
            self.work_ticks = 0
            self.last_work_items = 0

    def diagnostics(self) -> dict[str, int | float | bool]:
        with self._stats_lock:
            ticks = self.ticks
            work_ticks = self.work_ticks
            last_work_items = self.last_work_items
        return {
            "enabled": self._thread is not None and self._thread.is_alive(),
            "poll_interval_seconds": self.poll_interval_seconds,
            "ticks": ticks,
            "work_ticks": work_ticks,
            "last_work_items": last_work_items,
        }
