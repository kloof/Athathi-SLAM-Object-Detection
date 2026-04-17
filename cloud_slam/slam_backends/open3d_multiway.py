"""Open3D multiway registration backend — scan-to-scan ICP + pose graph.

Registers each scan to its immediate predecessor (odometry edge), then
adds sparse loop-closure candidate edges at a fixed stride (every N
frames). Runs a global Levenberg-Marquardt pose graph optimization over
all edges at the end.

Pros vs scan-to-map ICP:
  - Global consistency: drift accumulated in any segment is distributed
    across the whole trajectory.
  - Loop-closure edges correct revisits directly.

Cons:
  - Slower than pure odometry (ICP for every loop-closure candidate).
  - Quality of loop closures depends on the stride and on fitness
    filtering — a bad loop edge can pull the graph.

Uses the shared deskew_scan so input geometry matches other backends.
"""

import time

import numpy as np
import open3d as o3d

from cloud_slam.deskew import deskew_scan
from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult


def _register_pair(source_pcd: o3d.geometry.PointCloud,
                   target_pcd: o3d.geometry.PointCloud,
                   voxel: float,
                   T_init: np.ndarray = np.eye(4)
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-plane ICP returning (transformation, information, fitness)."""
    src = source_pcd.voxel_down_sample(voxel)
    tgt = target_pcd.voxel_down_sample(voxel)
    if len(src.points) < 10 or len(tgt.points) < 10:
        return T_init, np.eye(6), 0.0
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * 2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * 2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel * 3,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=30),
    )
    info = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src, tgt, voxel * 3, result.transformation))
    return result.transformation, info, result.fitness


@register("open3d_multiway")
class Open3DMultiwayBackend:
    name = "open3d_multiway"
    description = ("scan-to-scan point-to-plane ICP + loop-closure candidate "
                   "edges + global pose graph optimization")

    def __init__(self,
                 voxel: float = 0.1,
                 loop_stride: int = 20,
                 loop_min_fitness: float = 0.3,
                 loop_radius: int = 3):
        self.voxel = voxel
        self.loop_stride = loop_stride
        self.loop_min_fitness = loop_min_fitness
        self.loop_radius = loop_radius

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

        pcds: list[o3d.geometry.PointCloud] = []
        for stamp, xyz, time_offsets in clouds:
            xyz = np.asarray(xyz, dtype=np.float64)
            if len(imu_times) > 0 and len(time_offsets) > 0 and len(xyz) > 0:
                xyz = deskew_scan(xyz, time_offsets, stamp,
                                  imu_times, imu_gyros)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcds.append(pcd)

        n = len(pcds)
        poses: list[np.ndarray] = [np.eye(4)]
        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

        # Odometry edges (source=i-1, target=i).
        for i in range(1, n):
            T_rel, info, fitness = _register_pair(
                pcds[i], pcds[i - 1], self.voxel)
            T_world = poses[-1] @ T_rel
            poses.append(T_world)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(T_world))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id=i - 1,
                    target_node_id=i,
                    transformation=T_rel,
                    information=info,
                    uncertain=False))

        # Sparse loop-closure candidate edges.
        loops_added = 0
        for tgt in range(0, n, self.loop_stride):
            for src in range(tgt + self.loop_stride, n, self.loop_stride):
                # Use relative pose estimate from odometry chain as init.
                T_init = np.linalg.inv(poses[tgt]) @ poses[src]
                T_rel, info, fitness = _register_pair(
                    pcds[src], pcds[tgt], self.voxel, T_init=T_init)
                if fitness < self.loop_min_fitness:
                    continue
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_node_id=tgt,
                        target_node_id=src,
                        transformation=T_rel,
                        information=info,
                        uncertain=True))
                loops_added += 1

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel * 3,
            edge_prune_threshold=0.25,
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

        optimized_poses = [np.array(node.pose) for node in pose_graph.nodes]
        if len(optimized_poses) != len(clouds):
            raise RuntimeError(
                f"open3d_multiway: pose count {len(optimized_poses)} "
                f"!= cloud count {len(clouds)}")

        return BackendResult(
            poses=optimized_poses,
            runtime_s=time.time() - t0,
            backend_name=self.name,
            extra={
                "multiway_voxel": self.voxel,
                "multiway_loop_stride": self.loop_stride,
                "multiway_loops_added": loops_added,
            },
        )
