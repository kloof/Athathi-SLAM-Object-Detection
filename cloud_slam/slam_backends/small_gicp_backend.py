"""small_gicp backend — GICP scan-to-map with IMU-gyro init.

GICP (Generalized ICP) matches distributions of points, not individual
points — empirically more robust to sparse sampling and under-constrained
surface geometry than plain point-to-plane ICP. small_gicp is a
multi-threaded C++ implementation with clean Python bindings.

Scan-to-map: we accumulate a voxel-downsampled rolling local map and
register each new scan to it. IMU gyro integration seeds the initial
guess between frames, matching the baseline's motion-model strategy.

Deskew is done with the shared deskew_scan for consistency with other
backends.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

import small_gicp

from cloud_slam.deskew import deskew_scan
from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult


@register("small_gicp")
class SmallGICPBackend:
    name = "small_gicp"
    description = ("small_gicp Generalized-ICP scan-to-map, "
                   "IMU-gyro init, rolling local map")

    def __init__(self,
                 downsample: float = 0.1,
                 max_correspondence_distance: float = 0.4,
                 map_voxel: float = 0.1,
                 map_update_interval: int = 3,
                 map_max_points: int = 300_000,
                 num_threads: int = 4,
                 max_iterations: int = 30,
                 registration_type: str = "GICP"):
        self.downsample = downsample
        self.max_corr = max_correspondence_distance
        self.map_voxel = map_voxel
        self.map_update_interval = map_update_interval
        self.map_max_points = map_max_points
        self.num_threads = num_threads
        self.max_iterations = max_iterations
        self.registration_type = registration_type

    def _voxel_down(self, xyz: np.ndarray, voxel: float) -> np.ndarray:
        if len(xyz) == 0:
            return xyz
        keys = np.floor(xyz / voxel).astype(np.int64)
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        return xyz[np.sort(first_idx)]

    def run(self,
            clouds: list[tuple[float, np.ndarray, np.ndarray]],
            imus: list[tuple[float, np.ndarray, np.ndarray]]
            ) -> BackendResult:
        t0 = time.time()

        if imus:
            imu_times = np.array([t for t, _, _ in imus], dtype=np.float64)
            imu_gyros = np.array([g for _, g, _ in imus], dtype=np.float64)
        else:
            imu_times = np.array([], dtype=np.float64)
            imu_gyros = np.empty((0, 3), dtype=np.float64)

        poses: list[np.ndarray] = []
        T_current = np.eye(4)
        prev_stamp: float | None = None
        map_xyz = np.empty((0, 3), dtype=np.float64)
        n_success = 0
        n_frames_aligned = 0

        for i, (stamp, xyz, time_offsets) in enumerate(clouds):
            xyz = np.asarray(xyz, dtype=np.float64)

            if len(imu_times) > 0 and len(time_offsets) > 0 and len(xyz) > 0:
                xyz_desk = deskew_scan(xyz, time_offsets, stamp,
                                       imu_times, imu_gyros)
            else:
                xyz_desk = xyz

            if len(xyz_desk) < 10:
                poses.append(T_current.copy())
                prev_stamp = stamp
                continue

            if len(map_xyz) > 0 and prev_stamp is not None:
                dt = stamp - prev_stamp
                mask = (imu_times >= prev_stamp) & (imu_times < stamp)
                if mask.any():
                    avg_gyro = imu_gyros[mask].mean(axis=0)
                    dR = Rotation.from_rotvec(avg_gyro * dt).as_matrix()
                else:
                    dR = np.eye(3)
                T_guess = T_current.copy()
                T_guess[:3, :3] = T_current[:3, :3] @ dR

                try:
                    result = small_gicp.align(
                        target_points=map_xyz,
                        source_points=xyz_desk,
                        init_T_target_source=T_guess,
                        registration_type=self.registration_type,
                        downsampling_resolution=self.downsample,
                        max_correspondence_distance=self.max_corr,
                        num_threads=self.num_threads,
                        max_iterations=self.max_iterations,
                    )
                    if result.converged:
                        T_current = np.asarray(result.T_target_source)
                        n_success += 1
                    else:
                        T_current = T_guess
                except Exception as exc:
                    print(f"  [small_gicp] align exc frame {i}: {exc}")
                    T_current = T_guess
                n_frames_aligned += 1

            poses.append(T_current.copy())

            if i % self.map_update_interval == 0:
                xyz_world = xyz_desk @ T_current[:3, :3].T + T_current[:3, 3]
                map_xyz = np.vstack([map_xyz, xyz_world])
                map_xyz = self._voxel_down(map_xyz, self.map_voxel)
                if len(map_xyz) > self.map_max_points:
                    map_xyz = map_xyz[-self.map_max_points:]

            prev_stamp = stamp

        convergence = (n_success / n_frames_aligned) if n_frames_aligned else 0.0
        return BackendResult(
            poses=poses,
            runtime_s=time.time() - t0,
            backend_name=self.name,
            extra={
                "small_gicp_convergence_rate": round(convergence, 3),
                "small_gicp_map_final_points": int(len(map_xyz)),
                "registration_type": self.registration_type,
                "num_threads": self.num_threads,
            },
        )
