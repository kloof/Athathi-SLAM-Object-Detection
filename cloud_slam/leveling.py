"""
Floor-based leveling for the merged SLAM cloud.

Uses IMU gravity as a prior and RANSAC to find the actual floor plane, then
computes the rotation that maps the floor normal to +Z and (optionally) the
Z shift that places the floor at Z=0.

Also applies the same transform to a tracker's accumulated per-object points
so downstream OBB fitting happens in the already-leveled frame — which avoids
the post-hoc quaternion rotation dance that has been the root of several OBB
orientation bugs.

If RANSAC cannot produce a sensible floor plane, falls back to IMU gravity
alone (matches legacy behavior).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from cloud_slam.room_structure import detect_room
from cloud_slam.frustum import estimate_gravity


Z_UP = np.array([0.0, 0.0, 1.0])


def level_by_floor(pcd, imus=None, *, distance_thresh=0.03,
                   shift_floor_to_zero=True, fallback_to_imu=True,
                   max_prior_angle_deg=45.0):
    """
    Compute a rotation and Z shift that level the cloud so the floor is at Z=0.

    Strategy:
      1. Estimate gravity from IMU (if provided) as a prior.
      2. Run detect_room() with that prior — RANSAC finds the lowest horizontal
         plane and returns it as room.floor.
      3. Sanity-check: the RANSAC floor normal must be within
         max_prior_angle_deg of the IMU prior. If not, fall back to IMU.
      4. Compute R_level = align_vectors(+Z, target_normal).
      5. Compute z_shift = (R_level @ floor_centroid)[2] so floor lands at Z=0.

    Args:
        pcd:                  Open3D PointCloud in the raw SLAM world frame.
        imus:                 list of (timestamp, gyro, acc) — IMU prior and
                              fallback. Can be None or empty.
        distance_thresh:      RANSAC inlier threshold (m).
        shift_floor_to_zero:  If True, include a Z shift so the floor is at Z=0.
        fallback_to_imu:      If True and RANSAC fails, use IMU gravity alone.
        max_prior_angle_deg:  Max tolerable angle between RANSAC floor normal
                              and IMU prior. Beyond this, RANSAC is rejected.

    Returns:
        (R_level, z_shift, method)
            R_level: (3,3) rotation matrix. Apply via pcd.rotate(R, center=(0,0,0)).
            z_shift: float — apply via pcd.translate((0, 0, -z_shift)).
            method:  "ransac_floor" | "imu_fallback" | "identity"
    """
    # 1. IMU prior
    if imus:
        gravity_imu = estimate_gravity(imus)
    else:
        gravity_imu = Z_UP.copy()

    # 2. RANSAC floor detection with IMU prior
    floor_plane = None
    try:
        room = detect_room(pcd, gravity_up=gravity_imu,
                           distance_threshold=distance_thresh)
        floor_plane = room.floor
    except Exception as e:
        print(f"[LEVEL] detect_room raised: {e}")

    # 3. Validate RANSAC result
    use_ransac = False
    target_normal = None
    if floor_plane is not None:
        fn = floor_plane.normal.copy()
        # Ensure the normal points in the same hemisphere as the IMU up vector
        if fn @ gravity_imu < 0:
            fn = -fn
        angle_rad = np.arccos(np.clip(fn @ gravity_imu, -1.0, 1.0))
        angle_deg = float(np.degrees(angle_rad))
        if angle_deg < max_prior_angle_deg:
            use_ransac = True
            target_normal = fn
            print(f"[LEVEL] RANSAC floor normal: [{fn[0]:+.3f}, {fn[1]:+.3f}, "
                  f"{fn[2]:+.3f}] ({angle_deg:.2f}° from IMU gravity, "
                  f"{floor_plane.num_inliers} inliers)")
        else:
            print(f"[LEVEL] RANSAC floor normal {angle_deg:.1f}° from IMU "
                  f"gravity — rejecting, falling back to IMU")

    if not use_ransac:
        if not fallback_to_imu:
            raise ValueError("No floor plane found and fallback_to_imu=False")
        target_normal = gravity_imu
        method = "imu_fallback"
    else:
        method = "ransac_floor"

    # 4. Compute rotation
    if np.allclose(target_normal, Z_UP, atol=1e-4):
        R_level = np.eye(3)
        if method == "ransac_floor" and np.allclose(gravity_imu, Z_UP, atol=1e-3):
            method = "identity"
    else:
        R_level = Rotation.align_vectors([Z_UP], [target_normal])[0].as_matrix()

    # 5. Z shift so floor lands at Z=0
    z_shift = 0.0
    if shift_floor_to_zero:
        if floor_plane is not None:
            floor_centroid_leveled = R_level @ floor_plane.centroid
            z_shift = float(floor_centroid_leveled[2])
        else:
            # No floor plane (IMU fallback with no room detected) —
            # estimate floor Z as the 1st percentile of leveled points.
            pts = np.asarray(pcd.points)
            if len(pts) > 100:
                pts_z_leveled = (R_level @ pts.T)[2]
                z_shift = float(np.percentile(pts_z_leveled, 1.0))

    return R_level, z_shift, method


def apply_leveling_to_tracker(tracker, R_level, z_shift):
    """
    Apply the leveling transform to every tracked object's accumulated points
    and Kalman position state, in place.

    Point buffers are in world frame (see TrackedObject.point_buffer —
    populated via world_pts in detect_pipeline.on_frame). After this call,
    all accumulated data is in the leveled frame and downstream OBB fitting
    can run with gravity_up=[0,0,1].

    The per-object Kalman covariance is rotated for correctness even though
    no further update() calls occur after SLAM. The yaw accumulators
    (_yaw_sin_sum/_cos_sum) are left untouched: they are not consumed by
    the current refine_objects pipeline. If that changes, they would need to
    be recomputed from the rotated points.
    """
    if tracker is None:
        return

    shift = np.array([0.0, 0.0, -float(z_shift)])
    identity_rotation = np.allclose(R_level, np.eye(3), atol=1e-6)
    zero_shift = abs(z_shift) < 1e-6

    if identity_rotation and zero_shift:
        return  # nothing to do

    for obj in tracker.objects.values():
        new_buffer = []
        for pts in obj.point_buffer:
            if pts.size == 0:
                new_buffer.append(pts)
                continue
            pts_new = (R_level @ pts.T).T + shift
            new_buffer.append(pts_new)
        obj.point_buffer = new_buffer

        obj.x = R_level @ obj.x + shift
        obj.P = R_level @ obj.P @ R_level.T
