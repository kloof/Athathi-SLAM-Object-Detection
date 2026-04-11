#!/usr/bin/env python3
"""
Benchmark: Original vs Incremental map management.
Same Open3D ICP, but with:
1. Incremental voxel map (don't rebuild from scratch)
2. Cached normals on the map (only recompute when map changes)
"""

import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, '/home/klof/lidar_slam/cloud_slam')
from mcap_reader import read_mcap


class IncrementalVoxelMap:
    """Voxel grid that supports incremental insertion without full rebuild."""

    def __init__(self, voxel_size=0.05):
        self.voxel_size = voxel_size
        self.voxels = {}  # (ix, iy, iz) -> representative point
        self._cloud = None
        self._cloud_dirty = True
        self._normals_cloud = None
        self._normals_dirty = True

    def insert(self, points):
        """Add new points to the voxel map. Only keeps one point per voxel."""
        inv = 1.0 / self.voxel_size
        for i in range(len(points)):
            x, y, z = points[i]
            key = (int(np.floor(x * inv)),
                   int(np.floor(y * inv)),
                   int(np.floor(z * inv)))
            if key not in self.voxels:
                self.voxels[key] = (x, y, z)
                self._cloud_dirty = True
                self._normals_dirty = True

    def insert_bulk(self, points):
        """Vectorized insertion — much faster than per-point."""
        inv = 1.0 / self.voxel_size
        keys = np.floor(points * inv).astype(np.int64)
        for i in range(len(keys)):
            k = (keys[i, 0], keys[i, 1], keys[i, 2])
            if k not in self.voxels:
                self.voxels[k] = (points[i, 0], points[i, 1], points[i, 2])
                self._cloud_dirty = True
                self._normals_dirty = True

    def get_cloud(self):
        """Get Open3D point cloud (cached)."""
        if self._cloud_dirty or self._cloud is None:
            pts = np.array(list(self.voxels.values()), dtype=np.float64)
            self._cloud = o3d.geometry.PointCloud()
            if len(pts) > 0:
                self._cloud.points = o3d.utility.Vector3dVector(pts)
            self._cloud_dirty = False
            self._normals_dirty = True  # cloud changed, normals invalid
        return self._cloud

    def get_cloud_with_normals(self):
        """Get cloud with normals (cached — only recomputes when map changes)."""
        cloud = self.get_cloud()
        if self._normals_dirty and len(cloud.points) > 10:
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            self._normals_dirty = False
            self._normals_cloud = cloud
        return self._normals_cloud if self._normals_cloud is not None else cloud

    def size(self):
        return len(self.voxels)


def run_incremental(clouds, imus):
    """Same ICP algorithm but with incremental map + cached normals."""
    t0 = time.time()

    imu_times = np.array([t for t, _, _ in imus])
    imu_gyro = np.array([g for _, g, _ in imus])

    merged = o3d.geometry.PointCloud()  # full resolution for final output
    vmap = IncrementalVoxelMap(voxel_size=0.05)
    T_current = np.eye(4)
    prev_stamp = None

    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(xyz)

        if len(scan.points) < 10:
            continue

        if prev_stamp is not None and vmap.size() > 10:
            dt = stamp - prev_stamp
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                dR = Rotation.from_rotvec(imu_gyro[mask].mean(axis=0) * dt).as_matrix()
            else:
                dR = np.eye(3)

            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            # Downsample scan
            scan_down = scan.voxel_down_sample(0.1)
            scan_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

            # Get map with cached normals
            map_cloud = vmap.get_cloud_with_normals()

            if map_cloud is not None and len(map_cloud.points) > 10:
                result = o3d.pipelines.registration.registration_icp(
                    scan_down, map_cloud, 0.3, T_guess,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))

                if result.fitness > 0.1:
                    T_current = result.transformation
                else:
                    T_current = T_guess

        # Transform to world frame
        scan_world = o3d.geometry.PointCloud(scan)
        scan_world.transform(T_current)

        # Accumulate full-res for output
        merged += scan_world

        # Incrementally insert into voxel map (every frame, but cheap)
        world_pts = np.asarray(scan_world.points)
        vmap.insert_bulk(world_pts)

        prev_stamp = stamp

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  Frame {i+1}/{len(clouds)} ({fps:.0f} fps, map={vmap.size()} voxels)")

    elapsed = time.time() - t0
    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return merged, elapsed


def run_original(clouds, imus):
    """Original approach for comparison."""
    t0 = time.time()

    imu_times = np.array([t for t, _, _ in imus])
    imu_gyro = np.array([g for _, g, _ in imus])

    merged = o3d.geometry.PointCloud()
    T_current = np.eye(4)
    prev_stamp = None
    map_cloud = o3d.geometry.PointCloud()

    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(xyz)

        if len(scan.points) < 10:
            continue

        if prev_stamp is not None and len(map_cloud.points) > 0:
            dt = stamp - prev_stamp
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                dR = Rotation.from_rotvec(imu_gyro[mask].mean(axis=0) * dt).as_matrix()
            else:
                dR = np.eye(3)

            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            scan_down = scan.voxel_down_sample(0.1)
            map_down = map_cloud.voxel_down_sample(0.1)
            scan_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            map_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

            result = o3d.pipelines.registration.registration_icp(
                scan_down, map_down, 0.3, T_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))

            if result.fitness > 0.1:
                T_current = result.transformation
            else:
                T_current = T_guess

        scan_world = o3d.geometry.PointCloud(scan)
        scan_world.transform(T_current)
        merged += scan_world

        if i % 5 == 0:
            map_cloud = merged.voxel_down_sample(0.05)

        prev_stamp = stamp

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  Frame {i+1}/{len(clouds)} ({fps:.0f} fps, {len(merged.points)} pts)")

    elapsed = time.time() - t0
    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return merged, elapsed


def measure_quality(pcd, name):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    remaining = pcd
    rms_values = []
    for _ in range(3):
        if len(remaining.points) < 500: break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        pts = np.asarray(remaining.select_by_index(inliers).points)
        rms = np.sqrt(np.mean((a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d)**2)) * 1000
        rms_values.append(rms)
        remaining = remaining.select_by_index(inliers, invert=True)

    avg_rms = np.mean(rms_values) if rms_values else 0
    print(f"\n  {name}")
    print(f"  Points:       {len(pcd.points)}")
    print(f"  Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    print(f"  Avg wall RMS: {avg_rms:.2f} mm")
    for i, rms in enumerate(rms_values):
        print(f"    Plane {i+1}: {rms:.2f} mm")


# Main
BAG_DIR = "/home/klof/lidar_slam/data/scan_002"
print("Reading MCAP (scan_002, 120MB)...")
t0 = time.time()
clouds, imus = read_mcap(BAG_DIR)
print(f"Read {len(clouds)} clouds + {len(imus)} IMU in {time.time()-t0:.1f}s\n")

print("=" * 60)
print("Running ORIGINAL (rebuild map every 5 frames)...")
pcd_orig, time_orig = run_original(clouds, imus)
measure_quality(pcd_orig, f"ORIGINAL ({time_orig:.1f}s)")
o3d.io.write_point_cloud("/tmp/bench_original2.ply", pcd_orig)

print("\n" + "=" * 60)
print("Running INCREMENTAL (voxel map + cached normals)...")
pcd_incr, time_incr = run_incremental(clouds, imus)
measure_quality(pcd_incr, f"INCREMENTAL ({time_incr:.1f}s)")
o3d.io.write_point_cloud("/tmp/bench_incremental.ply", pcd_incr)

print(f"\n{'=' * 60}")
print(f"  SPEEDUP: {time_orig/time_incr:.1f}x ({time_orig:.1f}s -> {time_incr:.1f}s)")
print(f"{'=' * 60}")
