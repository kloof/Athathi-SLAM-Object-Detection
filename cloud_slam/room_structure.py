"""
Room structure detection: floor, walls, ceiling via sequential RANSAC.

Detects planar surfaces in the merged point cloud and classifies them
by their orientation relative to gravity.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Plane:
    """A detected planar surface."""
    normal: np.ndarray
    offset: float           # d in ax+by+cz+d=0
    centroid: np.ndarray
    num_inliers: int


@dataclass
class RoomStructure:
    """Complete room structure detection result."""
    floor: Optional[Plane] = None
    ceiling: Optional[Plane] = None
    walls: List[Plane] = field(default_factory=list)
    gravity_up: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    floor_height: float = 0.0
    ceiling_height: float = 3.0
    # M4a: horizontal planes between floor+1m and the selected ceiling.
    # These are dropped soffits / coffers / trays / HVAC bulkheads /
    # countertops — not the structural ceiling itself. Empty list by
    # default (pre-M4a behavior). Populated from `floor_candidates`
    # with the floor and ceiling picks removed.
    intermediate_horiz: List[Plane] = field(default_factory=list)


def detect_room(merged_pcd, gravity_up=None, voxel_size=0.03,
                distance_threshold=None, max_walls=8, min_inlier_ratio=0.01):
    """
    Detect room structure from merged point cloud.

    Args:
        merged_pcd: Open3D PointCloud (the full SLAM map)
        gravity_up: (3,) unit vector for "up" direction
        voxel_size: downsample resolution for RANSAC (0.03m = fast)
        distance_threshold: RANSAC inlier threshold
        max_walls: maximum wall planes to detect
        min_inlier_ratio: stop when inlier count drops below this fraction

    Returns:
        RoomStructure with detected planes
    """
    if gravity_up is None:
        gravity_up = np.array([0., 0., 1.])

    # Default RANSAC threshold matches voxel quantization
    if distance_threshold is None:
        distance_threshold = voxel_size

    # Downsample for fast RANSAC
    pcd = merged_pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_total = len(points)

    if n_total < 100:
        return RoomStructure(gravity_up=gravity_up)

    # Classify points by normal orientation
    dots = np.abs(normals @ gravity_up)
    horizontal_mask = dots > 0.8   # floor/ceiling candidates
    vertical_mask = dots < 0.3     # wall candidates

    result = RoomStructure(gravity_up=gravity_up)

    # --- Detect floor and ceiling from horizontal points ---
    horiz_indices = np.where(horizontal_mask)[0]
    if len(horiz_indices) > 100:
        horiz_pcd = pcd.select_by_index(horiz_indices)
        horiz_pts = points[horiz_indices]

        # Find ALL horizontal plane candidates, pick the LOWEST as floor
        # and the HIGHEST (above floor+1m) as ceiling.
        #
        # M4a: loop caps bumped 5 → 10 and min-inliers lowered 100 → 50 so
        # dropped ceiling features (perimeter soffits / coffered ceilings /
        # tray ceilings / HVAC bulkheads) don't starve the actual ceiling
        # out of the candidate list. The actual structural ceiling is
        # often heavily occluded by the dropped feature and surfaces with
        # only ~50-100 inliers, so min=100 used to silently reject it and
        # the soffit would win as "the ceiling".
        #
        # The LOWEST plane is still overwhelmingly the true floor because
        # no horizontal plane below the actual floor exists in normal scans.
        floor_candidates = []
        remaining_horiz = horiz_pcd
        for _ in range(10):
            if len(remaining_horiz.points) < 50:
                break
            candidate = _ransac_plane(remaining_horiz, distance_threshold)
            if candidate is None or candidate.num_inliers < 50:
                break
            floor_candidates.append(candidate)
            remaining_horiz = _remove_plane_inliers(
                remaining_horiz, candidate, distance_threshold
            )

        if floor_candidates:
            # Sort by height along gravity — lowest first
            floor_candidates.sort(key=lambda p: p.centroid @ gravity_up)
            floor_plane = floor_candidates[0]
            floor_h = floor_plane.centroid @ gravity_up
            result.floor = floor_plane
            result.floor_height = float(floor_h)

            # Ceiling = highest horizontal plane above floor + 1.0m
            ceil_candidates = [
                p for p in floor_candidates[1:]
                if (p.centroid @ gravity_up) > floor_h + 1.0
            ]
            ceil_plane = None
            if ceil_candidates:
                ceil_plane = max(ceil_candidates, key=lambda p: p.centroid @ gravity_up)
                result.ceiling = ceil_plane
                result.ceiling_height = float(ceil_plane.centroid @ gravity_up)

            # M4a: surface dropped soffits / coffers / trays so the
            # floorplan schema can emit them as `secondary_ceiling_features`.
            # These are horizontal planes strictly above floor+1m but NOT
            # the highest (which we just picked as the ceiling).
            if ceil_plane is not None:
                result.intermediate_horiz = [
                    p for p in ceil_candidates if p is not ceil_plane
                ]
            else:
                # No ceiling picked — every "above floor+1m" plane is an
                # intermediate horizontal feature in its own right.
                result.intermediate_horiz = list(ceil_candidates)

    # --- Detect walls from vertical points ---
    vert_indices = np.where(vertical_mask)[0]
    if len(vert_indices) > 100:
        remaining = pcd.select_by_index(vert_indices)
        min_inliers = max(int(len(vert_indices) * min_inlier_ratio), 50)

        for _ in range(max_walls):
            if len(remaining.points) < min_inliers:
                break

            plane = _ransac_plane(remaining, distance_threshold * 1.5)
            if plane is None or plane.num_inliers < min_inliers:
                break

            # Verify it's vertical
            if abs(plane.normal @ gravity_up) > 0.3:
                # Not a wall — remove and continue
                remaining = _remove_plane_inliers(remaining, plane, distance_threshold * 1.5)
                continue

            result.walls.append(plane)
            remaining = _remove_plane_inliers(remaining, plane, distance_threshold * 1.5)

    return result


def _ransac_plane(pcd, distance_threshold, ransac_n=3, num_iterations=1000):
    """Run RANSAC plane fitting, return Plane or None."""
    if len(pcd.points) < ransac_n:
        return None

    try:
        # M0b: reseed Open3D's RANSAC RNG each call so the pipeline is
        # byte-for-byte reproducible. Open3D 0.19 lacks a `seed=` kwarg
        # on segment_plane(), so the global seed API is the only lever.
        # See docs/plans/roomplan-quality.md (M0b).
        try:
            o3d.utility.random.seed(42)
        except AttributeError:
            pass
        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
    except Exception:
        return None

    if len(inliers) < 10:
        return None

    normal = np.array(model[:3])
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None
    normal /= norm

    inlier_pts = np.asarray(pcd.points)[inliers]
    return Plane(
        normal=normal,
        offset=float(model[3] / norm),
        centroid=inlier_pts.mean(axis=0),
        num_inliers=len(inliers),
    )


def _remove_plane_inliers(pcd, plane, distance_threshold):
    """Remove points close to a detected plane."""
    pts = np.asarray(pcd.points)
    dists = np.abs(pts @ plane.normal + plane.offset)
    keep = dists > distance_threshold
    return pcd.select_by_index(np.where(keep)[0])
