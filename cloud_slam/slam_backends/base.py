"""Backend protocol + shared dataclasses for the SLAM comparison harness."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class BackendResult:
    """What every backend returns.

    poses[i] is the 4x4 world-from-scan transform to apply to clouds[i]
    (one entry per input cloud, same ordering).
    """
    poses: list[np.ndarray]
    runtime_s: float
    backend_name: str
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    """A SLAM backend — pose estimator only. Merging + colorization are
    handled by the shared post_process module, not here.
    """
    name: str

    def run(self,
            clouds: list[tuple[float, np.ndarray, np.ndarray]],
            imus: list[tuple[float, np.ndarray, np.ndarray]]
            ) -> BackendResult:
        ...
