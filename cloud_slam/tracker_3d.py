"""
3D object tracking with per-object point accumulation and position-only Kalman filter.

Designed for static indoor objects observed from a moving sensor.
"""

import numpy as np
import open3d as o3d
from collections import defaultdict


class TrackedObject:
    """State for a single tracked object."""

    def __init__(self, track_id, center, class_name, frame_idx):
        self.track_id = track_id
        self.class_votes = defaultdict(int)
        self.class_votes[class_name] += 1
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

    @property
    def class_name(self):
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def status(self):
        if self.observation_count >= 3:
            return 'confirmed'
        return 'tentative'

    def update(self, center, world_pts, class_name, frame_idx):
        """Update tracker with a new observation."""
        self.class_votes[class_name] += 1
        self.last_frame = frame_idx
        self.observation_count += 1

        # Mahalanobis gating (chi2, 3 dof, 99th percentile = 11.34)
        innovation = center - self.x
        S = self.P + self.R
        d_mahal = innovation @ np.linalg.inv(S) @ innovation
        if d_mahal > 11.34:
            # Outlier — skip Kalman update, still accumulate process noise
            self.P = self.P + self.Q
            return

        # Kalman update
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(3) - K) @ self.P + self.Q

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


class ObjectTracker3D:
    """Manages all tracked objects across frames."""

    def __init__(self):
        self.objects = {}  # track_id -> TrackedObject

    def update(self, track_id, center, world_pts, class_name, frame_idx):
        """Update or create a tracked object."""
        if track_id in self.objects:
            self.objects[track_id].update(center, world_pts, class_name, frame_idx)
        else:
            obj = TrackedObject(track_id, center, class_name, frame_idx)
            obj.point_buffer.append(world_pts)
            self.objects[track_id] = obj

    def merge_fragmented_tracks(self, merge_dist=0.5):
        """
        Post-processing: merge track IDs with same class + nearby centroids.
        Handles ByteTrack/BoT-SORT ID fragmentation from occlusion.
        """
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

                # Same class and close centroids
                if obj_a.class_name != obj_b.class_name:
                    continue
                dist = np.linalg.norm(obj_a.x - obj_b.x)
                if dist > merge_dist:
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
                'confidence': float(max(obj.class_votes.values()) / obj.observation_count),
                'center': obj.x.tolist(),
                'num_points': len(pts),
                'num_observations': obj.observation_count,
                'first_frame': obj.first_frame,
                'last_frame': obj.last_frame,
            })

        return results
