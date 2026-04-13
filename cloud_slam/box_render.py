"""Box wireframe point-cloud sampling helper.

Exposes `create_box_points`, used by `scripts/detect_and_slam.py` to sample
points along the 12 edges of an oriented bounding box so the detections
are visible as red wireframes in the PLY output.

Extracted from `scripts/detect_and_slam.py` during the M0d package split
so the helper lives with the other render/export utilities. Zero behavior
change.
"""

import numpy as np


def create_box_points(center, dimensions, quat_xyzw, spacing=0.01):
    """Sample points along OBB edges for PLY visualization."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(quat_xyzw).as_matrix()
    dx, dy, dz = dimensions / 2

    # 8 corners of the box in local frame
    corners_local = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
    ])

    # Transform to world frame
    corners = (R @ corners_local.T).T + center

    # 12 edges of a box
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]

    points = []
    for a, b in edges:
        dist = np.linalg.norm(corners[b] - corners[a])
        n_pts = max(int(dist / spacing), 2)
        for t in np.linspace(0, 1, n_pts):
            points.append(corners[a] + t * (corners[b] - corners[a]))

    return np.array(points)
