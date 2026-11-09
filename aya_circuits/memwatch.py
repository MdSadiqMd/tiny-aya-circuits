"""Memory budget guard for MPS runs.

The owner granted 30 GB. `check()` is called between batches: above the soft limit it frees
caches and sleeps; above the hard limit it raises `MemoryBudgetExceeded` so the caller can
checkpoint and exit instead of swapping the machine. Peak usage is recorded for manifests.
Process memory is sampled with `resource` (RSS) and `torch.mps.driver_allocated_memory`.
"""

from __future__ import annotations

import gc
import os
import resource
import subprocess
import threading
import time

import torch


class MemoryBudgetExceeded(RuntimeError):
    pass


def used_gb() -> float:
    """Current footprint: max(resident set size now, Metal driver allocation incl. cache).
    ru_maxrss is a lifetime peak, so RSS is read from `ps` instead; unified memory means the two
    views
    overlap, hence max rather than sum."""
    try:
        rss = (
            int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True))
            * 1024
            / 2**30
        )
    except Exception:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    mps = torch.mps.driver_allocated_memory() / 2**30 if torch.backends.mps.is_available() else 0.0
    return max(rss, mps)


class MemWatch:
    def __init__(
        self,
        budget_gb: float = 30.0,
        soft_fraction: float = 0.85,
        pause_s: float = 5.0,
        sample_s: float = 2.0,
    ):
        self.budget = budget_gb
        self.soft = budget_gb * soft_fraction
        self.pause_s = pause_s
        self.peak = 0.0
        self.pauses = 0
        self.over_budget_samples = 0
        # Background sampler so peaks inside long library calls (MEMIT, pillars) still reach the
        # manifest.
        self._stop = threading.Event()
        threading.Thread(target=self._sample, args=(sample_s,), daemon=True).start()

    def _sample(self, every: float) -> None:
        while not self._stop.wait(every):
            u = used_gb()
            self.peak = max(self.peak, u)
            if u > self.budget:
                self.over_budget_samples += 1
                if self.over_budget_samples in (1, 10, 100) or self.over_budget_samples % 300 == 0:
                    print(
                        f"[memwatch] {u:.1f} GB > budget {self.budget} GB "
                        f"(sample {self.over_budget_samples})",
                        flush=True,
                    )

    def check(self) -> float:
        u = used_gb()
        self.peak = max(self.peak, u)
        if u > self.soft:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            u = used_gb()
            if u > self.soft:
                self.pauses += 1
                time.sleep(self.pause_s)
        if u > self.budget:
            raise MemoryBudgetExceeded(
                f"{u:.1f} GB used > {self.budget:.1f} GB budget; stopping so the owner can decide"
            )
        return u

    def summary(self) -> dict:
        self._stop.set()
        return {
            "peak_gb": round(self.peak, 2),
            "budget_gb": self.budget,
            "pauses": self.pauses,
            "over_budget_samples": self.over_budget_samples,
        }
