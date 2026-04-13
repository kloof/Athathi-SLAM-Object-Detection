"""Synthetic 6-wall L-shape with a 1 cm closure gap. For Stage 8 degeneracy tests."""
import numpy as np


def walls_open_lshape(closure_gap_m: float = 0.01):
    """Return a list of 6 walls forming an L-shape with a small closure gap.

    The gap is added to the last wall's endpoint so Σ (p2 − p1) ≠ 0.
    Used to exercise Stage 8's polygon-closure solver under Manhattan
    snap constraints where all walls are axis-aligned.
    """
    pts = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 2.0],
        [2.0, 2.0],
        [2.0, 3.0],
        [0.0, 3.0],
    ])
    # Perturb the last point slightly so the polygon does not close cleanly.
    end = pts[0] + np.array([closure_gap_m, 0.0])
    sequence = [pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], end]
    walls = []
    angles = [0.0, 90.0, 180.0, 90.0, 180.0, 270.0]
    for i in range(6):
        a, b = sequence[i], sequence[i + 1]
        walls.append((a, b, angles[i], float(np.linalg.norm(b - a))))
    return walls
