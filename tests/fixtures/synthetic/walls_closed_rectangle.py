"""Synthetic 4-wall closed rectangle (Δ = 0). For Stage 8 no-op tests."""
import numpy as np


def walls_closed_rectangle(length: float = 4.0, width: float = 3.0):
    """Return a list of 4 walls forming a closed rectangle.

    Each wall is a tuple (p1, p2, angle_deg, length_m) matching the
    format produced by extract_walls().
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([length, 0.0])
    p2 = np.array([length, width])
    p3 = np.array([0.0, width])
    return [
        (p0, p1, 0.0, length),
        (p1, p2, 90.0, width),
        (p2, p3, 180.0, length),
        (p3, p0, 270.0, width),
    ]
