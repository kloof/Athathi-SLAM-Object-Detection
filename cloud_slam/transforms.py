"""Post-SLAM buffer transformation helpers.

`apply_transform_to_buffers` is the single chokepoint for applying a
(rotation, translation) pair to every buffer that moves in lockstep
after SLAM completes: the merged cloud, per-point wall labels, detected
objects, and (future M2) back-projected line endpoints.

The 16-agent review flagged that adding buffers one at a time with
inline transforms creates a silent-drift hazard: if any site forgets
one buffer, that buffer silently rotates relative to merged. This helper
is the chokepoint — new buffers add a single keyword to this signature.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def apply_transform_to_buffers(
    R: np.ndarray,
    t: Optional[np.ndarray] = None,
    *,
    merged=None,
    wall_labels: Optional[dict] = None,
    objects: Optional[Iterable[dict]] = None,
    lines: Optional[dict] = None,
) -> None:
    """Apply (R, t) to every non-None post-SLAM buffer in lockstep.

    The 16-agent review flagged that adding buffers one at a time with
    inline transforms creates a silent-drift hazard: if any site
    forgets one buffer, that buffer drifts relative to merged by a
    rotation. This helper is the single chokepoint for post-SLAM
    transforms so new buffers can be added with one signature change.

    Quaternion convention: the codebase uses scipy's (x, y, z, w)
    ordering — see box_refiner.py:134 ('format': 'xyzw'). We compose
    R_new = R @ R_obj and re-emit via scipy.spatial.transform.Rotation.

    Args:
        R: 3x3 rotation matrix (numpy array).
        t: optional 3-vector translation; if None, rotate-only.
        merged: optional Open3D PointCloud (rotated in place, translated if t given).
        wall_labels: optional dict with 'xyz' key holding an (N, 3) ndarray.
                     Operates on wall_labels['xyz'] in place. The numeric
                     dtype is preserved by casting the intermediate to
                     float64 and back to the original dtype (current
                     pipeline uses float32 for wall_labels['xyz']).
        objects: optional iterable of dicts with 'center' (3-vec list/array)
                 and optional 'orientation.quaternion' (xyzw 4-vec). Updated
                 in place. Translation does not affect orientation.
        lines: optional dict with 'start' and 'end' keys holding (N, 3)
               ndarrays. Lines are expected to ALREADY be in the current
               world frame — this helper applies post-SLAM global rotations
               only; it must NOT be used to promote lines from sensor to
               world frame. (Lines buffer is pre-wired for M2 and will be
               None until that milestone lands.)
    """
    # Local import to keep module import cost minimal if the helper
    # isn't called (it will always be, but preserves the original
    # pattern where scipy/open3d are imported lazily in main()).
    from scipy.spatial.transform import Rotation as _SciRot

    if merged is not None:
        merged.rotate(R, center=(0, 0, 0))
        if t is not None:
            merged.translate(t)

    if wall_labels is not None:
        xyz = wall_labels.get('xyz')
        if xyz is not None and len(xyz) > 0:
            orig_dtype = xyz.dtype
            rotated = xyz.astype(np.float64) @ R.T
            if t is not None:
                rotated = rotated + np.asarray(t, dtype=np.float64)
            wall_labels['xyz'] = rotated.astype(orig_dtype)

    if objects is not None:
        t_arr = None if t is None else np.asarray(t, dtype=np.float64)
        for obj in objects:
            c = np.array(obj['center'], dtype=np.float64)
            c_new = R @ c
            if t_arr is not None:
                c_new = c_new + t_arr
            obj['center'] = c_new.tolist()
            if 'orientation' in obj and 'quaternion' in obj['orientation']:
                q = np.array(obj['orientation']['quaternion'], dtype=np.float64)
                R_obj = _SciRot.from_quat(q).as_matrix()
                R_new = R @ R_obj
                obj['orientation']['quaternion'] = (
                    _SciRot.from_matrix(R_new).as_quat().tolist())

    if lines is not None:
        for key in ('start', 'end'):
            arr = lines.get(key)
            if arr is None or len(arr) == 0:
                continue
            orig_dtype = arr.dtype
            rotated = np.asarray(arr, dtype=np.float64) @ R.T
            if t is not None:
                rotated = rotated + np.asarray(t, dtype=np.float64)
            lines[key] = rotated.astype(orig_dtype)
