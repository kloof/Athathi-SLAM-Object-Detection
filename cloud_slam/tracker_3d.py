"""
3D object tracking with per-object point accumulation and position-only Kalman filter.

Designed for static indoor objects observed from a moving sensor.
"""

import numpy as np
import open3d as o3d
from collections import defaultdict


class TrackedObject:
    """State for a single tracked object."""

    def __init__(self, track_id, center, class_name, frame_idx, confidence=1.0):
        self.track_id = track_id
        self.class_votes = defaultdict(float)
        self.class_votes[class_name] += confidence
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.observation_count = 1

        # Position-only Kalman filter: state=[x,y,z]
        self.x = center.copy()                       # state estimate
        self.P = np.diag([0.04, 0.04, 0.04])         # covariance (init = R)
        self.Q = np.diag([0.0001, 0.0001, 0.0001])   # process noise (tiny, static objects)
        self.R = np.diag([0.04, 0.04, 0.04])         # measurement noise

        # Point accumulation buffer (world frame)
        self.point_buffer = []
        self.max_points = 500

        # Orientation tracking: yaw (rotation around gravity axis)
        # Accumulated as (sin, cos) components for correct circular averaging
        self._yaw_sin_sum = 0.0
        self._yaw_cos_sum = 0.0
        self._yaw_weight_sum = 0.0

    @property
    def class_name(self):
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def status(self):
        if self.observation_count >= 3:
            return 'confirmed'
        return 'tentative'

    @property
    def tracked_yaw(self):
        """Smoothed yaw from all observations (circular mean). None if no yaw data."""
        if self._yaw_weight_sum < 1e-6:
            return None
        # Recover angle from doubled-angle sin/cos sums (handles 180° ambiguity)
        return 0.5 * np.arctan2(
            self._yaw_sin_sum / self._yaw_weight_sum,
            self._yaw_cos_sum / self._yaw_weight_sum,
        )

    def update(self, center, world_pts, class_name, frame_idx, confidence=1.0,
               gravity_up=None):
        """Update tracker with a new observation."""
        # Mahalanobis gating (chi2, 3 dof, 99th percentile = 11.34)
        innovation = center - self.x
        S = self.P + self.R
        d_mahal = innovation @ np.linalg.inv(S) @ innovation
        if d_mahal > 11.34:
            # Outlier — reject entirely (no vote, no count, no points)
            self.P = self.P + self.Q
            return

        # Observation accepted — update votes and count
        self.class_votes[class_name] += confidence
        self.last_frame = frame_idx
        self.observation_count += 1

        # Kalman update
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(3) - K) @ self.P + self.Q

        # Orientation: estimate yaw from frustum points projected onto horizontal plane
        if gravity_up is not None and len(world_pts) >= 5:
            yaw = _estimate_yaw(world_pts, gravity_up)
            if yaw is not None:
                w = confidence * min(len(world_pts), 50) / 50
                self._yaw_sin_sum += w * np.sin(2 * yaw)
                self._yaw_cos_sum += w * np.cos(2 * yaw)
                self._yaw_weight_sum += w

        # Accumulate points
        self.point_buffer.append(world_pts)
        total = sum(len(p) for p in self.point_buffer)
        if total > self.max_points:
            self._downsample_buffer()

    def _downsample_buffer(self):
        """Voxel downsample accumulated points to stay within budget."""
        all_pts = np.concatenate(self.point_buffer, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd = pcd.voxel_down_sample(0.02)
        self.point_buffer = [np.asarray(pcd.points)]

    def get_accumulated_points(self):
        """Get all accumulated world-frame points."""
        if not self.point_buffer:
            return np.empty((0, 3))
        return np.concatenate(self.point_buffer, axis=0)


def _estimate_yaw(world_pts, gravity_up):
    """Estimate object yaw from frustum points in the gravity-aligned frame.

    Returns yaw angle in the gravity-aligned coordinate system where Z=up.
    """
    from scipy.spatial.transform import Rotation as _Rot

    g = gravity_up / np.linalg.norm(gravity_up)
    z = np.array([0.0, 0.0, 1.0])

    # Build rotation to align gravity with Z
    if np.allclose(g, z, atol=0.01):
        R_grav = np.eye(3)
    elif np.allclose(g, -z, atol=0.01):
        R_grav = np.diag([1.0, -1.0, -1.0])
    else:
        R_grav = _Rot.align_vectors([z], [g])[0].as_matrix()

    # Transform points to gravity-aligned frame and do PCA on XY
    pts_grav = (R_grav @ world_pts.T).T
    pts_xy = pts_grav[:, :2] - pts_grav[:, :2].mean(axis=0)

    cov = pts_xy.T @ pts_xy
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # largest eigenvalue direction in XY

    norm = np.linalg.norm(principal)
    if norm < 1e-6:
        return None

    return np.arctan2(principal[1], principal[0])


class ObjectTracker3D:
    """Manages all tracked objects across frames."""

    def __init__(self):
        self.objects = {}  # track_id -> TrackedObject

    def update(self, track_id, center, world_pts, class_name, frame_idx,
               confidence=1.0, gravity_up=None):
        """Update or create a tracked object."""
        if track_id in self.objects:
            self.objects[track_id].update(
                center, world_pts, class_name, frame_idx, confidence, gravity_up
            )
        else:
            obj = TrackedObject(track_id, center, class_name, frame_idx, confidence)
            obj.point_buffer.append(world_pts)
            if gravity_up is not None and len(world_pts) >= 5:
                yaw = _estimate_yaw(world_pts, gravity_up)
                if yaw is not None:
                    w = confidence * min(len(world_pts), 50) / 50
                    obj._yaw_sin_sum = w * np.sin(2 * yaw)
                    obj._yaw_cos_sum = w * np.cos(2 * yaw)
                    obj._yaw_weight_sum = w
            self.objects[track_id] = obj

    def merge_fragmented_tracks(self, merge_dist=0.5, cross_class_dist=0.3):
        """
        Post-processing: merge nearby track IDs.

        Two passes:
        1. Same class + nearby centroids (merge_dist) — handles BoT-SORT ID fragmentation
        2. Different class + very close centroids (cross_class_dist) — handles
           the same object detected as different classes in different frames
        """
        for same_class_only, threshold in [(True, merge_dist), (False, cross_class_dist)]:
            track_ids = list(self.objects.keys())
            merged = set()

            for i, tid_a in enumerate(track_ids):
                if tid_a in merged:
                    continue
                obj_a = self.objects[tid_a]
                for tid_b in track_ids[i + 1:]:
                    if tid_b in merged:
                        continue
                    obj_b = self.objects[tid_b]

                    if same_class_only and obj_a.class_name != obj_b.class_name:
                        continue

                    centroid_a = obj_a.get_accumulated_points().mean(axis=0) if obj_a.point_buffer else obj_a.x
                    centroid_b = obj_b.get_accumulated_points().mean(axis=0) if obj_b.point_buffer else obj_b.x
                    dist = np.linalg.norm(centroid_a - centroid_b)
                    if dist > threshold:
                        continue

                    # Merge b into a (keep the one with more observations)
                    if obj_b.observation_count > obj_a.observation_count:
                        obj_a, obj_b = obj_b, obj_a
                        tid_a, tid_b = tid_b, tid_a

                    # Transfer state
                    for cls, cnt in obj_b.class_votes.items():
                        obj_a.class_votes[cls] += cnt
                    obj_a.point_buffer.extend(obj_b.point_buffer)
                    obj_a.observation_count += obj_b.observation_count
                    obj_a.first_frame = min(obj_a.first_frame, obj_b.first_frame)
                    obj_a.last_frame = max(obj_a.last_frame, obj_b.last_frame)

                    merged.add(tid_b)

            for tid in merged:
                del self.objects[tid]

    def get_final_objects(self, **kwargs):
        """
        Get finalized object list. Only returns confirmed objects (3+ observations).
        OBB fitting is deferred to box_refiner for single-pass computation.

        Returns:
            list of dicts with object properties (no OBB yet — added by refiner)
        """
        results = []
        for tid, obj in self.objects.items():
            if obj.status != 'confirmed':
                continue

            pts = obj.get_accumulated_points()
            if len(pts) < 10:
                continue

            results.append({
                'track_id': int(tid),
                'class': obj.class_name,
                'confidence': float(max(obj.class_votes.values()) / max(sum(obj.class_votes.values()), 1e-6)),
                'center': obj.x.tolist(),
                'num_points': len(pts),
                'num_observations': obj.observation_count,
                'first_frame': obj.first_frame,
                'last_frame': obj.last_frame,
            })

        return results
