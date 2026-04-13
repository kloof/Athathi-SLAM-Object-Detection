#!/usr/bin/env python3
"""
Combined SLAM + YOLOE 3D object detection.

Processes a lidar-camera rosbag to produce a colored point cloud map
with 3D bounding boxes for detected objects.

Usage:
    python3 detect_and_slam.py /path/to/rosbag /path/to/output /path/to/calibration [--classes "person,chair,table"]
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import open3d as o3d

# Add parent dir so cloud_slam imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cloud_slam.box_render import create_box_points
from cloud_slam.frustum import estimate_gravity
from cloud_slam.transforms import apply_transform_to_buffers


def _load_calibration_info(calibration_dir):
    """Extract M0a calibration-block fields from the scan's extrinsics.yaml.

    Reads `method` and `calibration_date` (if present). Derives
    `age_days` as today-date minus calibration-date when parseable.
    All fields are optional — anything missing falls back to the
    M0a defaults inside `cloud_slam.floorplan.schema._build_calibration_block`.
    """
    import datetime
    import yaml

    info = {}
    ext_path = os.path.join(calibration_dir, "extrinsics.yaml")
    try:
        with open(ext_path) as f:
            ext = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Floorplan] calibration_info: couldn't read {ext_path}: {e}")
        return info

    if 'method' in ext:
        info['method'] = str(ext['method'])
    if 'calibration_date' in ext:
        date_str = str(ext['calibration_date'])
        info['calibration_date'] = date_str
        try:
            cal_date = datetime.date.fromisoformat(date_str)
            info['age_days'] = (datetime.date.today() - cal_date).days
        except ValueError:
            # Unparseable date → leave age_days unset (falls back to None).
            pass
    return info


def main():
    # M0b: RANSAC seeding for byte-for-byte regression reproducibility.
    # See docs/plans/roomplan-quality.md (M0b) — Open3D >=0.15 exposes
    # o3d.utility.random.seed() which seeds the internal Mersenne
    # twister that segment_plane() samples from. numpy/random are
    # seeded too for any downstream helpers. Without this, rerunning
    # the pipeline on the same scan gives non-deterministic floorplan
    # corners.
    # Note: RANSAC normals are reproducible with these seeds; downstream
    # float-order non-associativity still causes ~1e-8 drift in final object centers.
    _SEED = 42
    np.random.seed(_SEED)
    random.seed(_SEED)
    try:
        o3d.utility.random.seed(_SEED)
    except AttributeError:
        pass  # Open3D < 0.15 has no seed API

    parser = argparse.ArgumentParser(description="SLAM + YOLOE 3D Object Detection")
    parser.add_argument("rosbag", help="Path to rosbag directory or .mcap file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("calibration", help="Path to calibration directory")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names (default: indoor furniture)")
    parser.add_argument("--voxel-size", type=float, default=0.005,
                        help="Final voxel size in meters (default: 0.005)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLOE confidence threshold (default: 0.3)")
    parser.add_argument("--leveling", choices=["legacy", "floor"], default="legacy",
                        help="Leveling strategy. 'legacy' (default) = IMU "
                             "gravity estimate applied AFTER refinement (the "
                             "stable, 2-months-in-the-making path). 'floor' = "
                             "RANSAC floor detection (with IMU prior) applied "
                             "BEFORE refinement and shifts floor to Z=0 — "
                             "experimental, sharper leveling.")
    parser.add_argument("--label-walls", action="store_true",
                        help="Enable vision-based wall typing. Runs "
                             "Mask2Former (ADE20K) on each camera frame, "
                             "projects wall/window/door/glass pixel labels "
                             "onto lidar points, and passes them into the "
                             "floorplan refiner so each D_refined wall gets "
                             "a `type` field in the metadata and a colored "
                             "edge in floorplan_refined.png. Opt-in — off "
                             "by default, zero regression otherwise.")
    parser.add_argument("--no-stage8", action="store_true",
                        help="Disable the Stage 8 polygon-closure solver "
                             "(M1a). Without this flag, Stage 8 runs by "
                             "default and no-ops when |Δ| < 10 mm, so the "
                             "default path is byte-identical on already-"
                             "closed polygons. Use this flag for A/B "
                             "regression comparison against the pre-M1a "
                             "baseline.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  SLAM + YOLOE 3D Object Detection")
    print("=" * 60)

    # Load calibration
    from cloud_slam.colorizer import load_calibration
    calib = load_calibration(
        os.path.join(args.calibration, "intrinsics.yaml"),
        os.path.join(args.calibration, "extrinsics.yaml"),
    )
    print(f"[INFO] Loaded calibration from {args.calibration}")

    # Read MCAP
    t0 = time.time()
    from cloud_slam.mcap_reader import read_mcap
    clouds, imus, images = read_mcap(args.rosbag)
    t_read = time.time() - t0
    print(f"[INFO] Read {len(clouds)} clouds + {len(imus)} IMU + {len(images)} images in {t_read:.1f}s")

    # Configure detector
    detector_config = {'conf': args.conf}
    if args.classes:
        detector_config['classes'] = [c.strip() for c in args.classes.split(',')]

    # Optional vision wall segmenter — built only if the user opted in.
    # Any failure during segmenter init will flip it to a disabled state
    # internally, so the pipeline still runs (just without vision labels).
    wall_segmenter = None
    if args.label_walls:
        from cloud_slam.wall_segmenter import WallSegmenter
        wall_segmenter = WallSegmenter()  # lazy-loads on first segment() call
        print("[INFO] Vision wall labeling ENABLED "
              "(Mask2Former ADE20K — loads on first frame)")

    # Run pipeline
    from cloud_slam.pipelines.detect_pipeline import run
    merged, poses, objects, stats, wall_labels = run(
        clouds, imus, images, calib,
        voxel_size=args.voxel_size,
        detector_config=detector_config,
        leveling_mode=args.leveling,
        wall_segmenter=wall_segmenter,
    )

    # M4b: extract pose translations as an (N, 3) array up front. The
    # subsequent R_level / R_align / RANSAC-leveler transforms rotate
    # `merged` in place but leave `poses` (a list of 4x4 T_world_lidar)
    # pointing at the raw SLAM frame. We need the pose XY to match the
    # FINAL leveled-aligned frame of `merged` so the floorplan
    # trajectory-containment filter compares apples to apples.
    if poses is not None and len(poses) > 0:
        _pose_xyz = np.asarray(
            [np.asarray(p)[:3, 3] for p in poses], dtype=np.float64)
    else:
        _pose_xyz = np.zeros((0, 3), dtype=np.float64)

    t_total = time.time() - t0

    # --- M0c: per-scan calibration verification (must run BEFORE leveling) ---
    # Reprojection IoU only makes sense while the merged cloud's world frame
    # still matches the per-frame poses. Once we apply R_level / R_align /
    # RANSAC-leveler below, the cloud rotates but `poses` keep pointing at
    # the raw frame, so we verify here first. The returned dict gets merged
    # into `calibration_info` and surfaces in the floorplan JSON schema.
    calibration_info = _load_calibration_info(args.calibration)
    try:
        from cloud_slam.calibration import (
            verify_calibration, should_warn, warning_message,
        )
        from cloud_slam.projection import project_lidar_to_camera

        frame_masks = (wall_labels.get('frame_wall_masks')
                       if wall_labels is not None else None)
        if frame_masks:
            # Geometric wall inliers on the pre-leveling merged cloud.
            # Uses room_structure.detect_room (RANSAC + IMU gravity prior)
            # to pick wall planes; we concat their inliers into a single
            # (N, 3) array for reprojection. Avoids circularity: these
            # points are chosen by geometry, not by Mask2Former.
            from cloud_slam.room_structure import detect_room as _detect_room

            try:
                # RANSAC wall detection with IMU-derived gravity prior (the
                # merged cloud hasn't been leveled yet at this point — its
                # gravity still points in the raw-sensor direction, which
                # is NOT [0, 0, 1]). detect_room's default would be
                # [0, 0, 1], which would misclassify floor+ceiling as walls
                # on the tilted L2 scanner.
                _gravity_up_for_rooms = estimate_gravity(imus)
                _room_raw = _detect_room(merged, gravity_up=_gravity_up_for_rooms)
                # Plane dataclass doesn't store inlier points — only the
                # plane equation + count. Recover inliers by thresholding
                # the full merged cloud with `|p.n + d| < tol`.
                _all_pts = np.asarray(merged.points, dtype=np.float64)
                _wall_pts_list = []
                _wall_tol = 0.05  # 5 cm — tight enough to pin to wall surface
                for _wall in (_room_raw.walls or []):
                    _n = np.asarray(_wall.normal, dtype=np.float64)
                    _d = float(_wall.offset)
                    _dist = np.abs(_all_pts @ _n + _d)
                    _mask = _dist < _wall_tol
                    if not _mask.any():
                        continue
                    _wall_pts_list.append(_all_pts[_mask])
                if _wall_pts_list:
                    _merged_wall_pts = np.concatenate(_wall_pts_list, axis=0)
                    # Cap to 50k points — reprojection is O(N per frame),
                    # and the IoU signal saturates well before 50k points.
                    if len(_merged_wall_pts) > 50_000:
                        _rng = np.random.default_rng(42)
                        _idx = _rng.choice(len(_merged_wall_pts),
                                           50_000, replace=False)
                        _merged_wall_pts = _merged_wall_pts[_idx]
                else:
                    _merged_wall_pts = np.zeros((0, 3), dtype=np.float64)
            except Exception as _e:
                print(f"[Calibration] detect_room failed "
                      f"({type(_e).__name__}: {_e}); skipping verification")
                _merged_wall_pts = np.zeros((0, 3), dtype=np.float64)

            if len(_merged_wall_pts) > 0:
                # Build the parallel lists expected by verify_calibration.
                # images_with_poses: (None, pose, wall_mask) per sampled frame.
                # lidar_wall_inliers_per_frame: same merged array reused per
                # slot (M0c pragmatic simplification — the docstring documents
                # this). The `project_fn` adapter converts world→lidar via
                # inverse pose, then calls project_lidar_to_camera.
                def _project_world_to_image(pts_world, pose_l2w):
                    R = pose_l2w[:3, :3]
                    t = pose_l2w[:3, 3]
                    # world -> lidar frame: p_l = R^T (p_w - t)
                    pts_lidar = (R.T @ (pts_world - t).T).T
                    _cam, pixels, in_front = project_lidar_to_camera(
                        pts_lidar, calib)
                    return pixels, in_front

                # Build a list of max(frame_idx)+1 slots so that
                # verify_calibration can index by frame_idx. The iterable
                # only yields entries for sampled frames, so we build
                # those slots explicitly.
                _max_idx = max(fi for fi, _, _ in frame_masks)
                _triplets = [(None, None, None)] * (_max_idx + 1)
                _inliers_slots = [np.zeros((0, 3))] * (_max_idx + 1)
                for _fi, _pose, _mask in frame_masks:
                    _triplets[_fi] = (None, _pose, _mask)
                    _inliers_slots[_fi] = _merged_wall_pts

                _img_shape = None
                for _fi, _, _mask in frame_masks:
                    _img_shape = _mask.shape[:2]
                    break

                cal_result = verify_calibration(
                    images_with_poses=_triplets,
                    lidar_wall_inliers_per_frame=_inliers_slots,
                    project_fn=_project_world_to_image,
                    image_shape=_img_shape,
                    sample_every=1,  # triplets already subsampled
                )
                print(f"[Calibration] reprojection IoU mean="
                      f"{cal_result['reprojection_iou_mean']}, "
                      f"min={cal_result['reprojection_iou_min']}, "
                      f"frames={cal_result['reprojection_iou_frames_checked']}, "
                      f"tier={cal_result['accuracy_tier']}")
                # Merge into calibration_info so the floorplan schema picks
                # up the IoU fields. Renamed keys match what
                # schema._build_calibration_block already looks up.
                calibration_info['reprojection_iou_mean'] = (
                    cal_result['reprojection_iou_mean'])
                calibration_info['reprojection_iou_min'] = (
                    cal_result['reprojection_iou_min'])
                calibration_info['reprojection_iou_frames_checked'] = (
                    cal_result['reprojection_iou_frames_checked'])
                calibration_info['accuracy_tier'] = cal_result['accuracy_tier']
                # Stash the built calibration block for the end-of-run warning.
                calibration_info['_verified_block'] = {
                    'reprojection_iou_mean': cal_result['reprojection_iou_mean'],
                    'reprojection_iou_min': cal_result['reprojection_iou_min'],
                    'reprojection_iou_frames_checked': cal_result[
                        'reprojection_iou_frames_checked'],
                    'accuracy_tier': cal_result['accuracy_tier'],
                }
        else:
            if wall_segmenter is not None:
                print("[Calibration] no per-frame wall masks accumulated; "
                      "skipping reprojection-IoU verification")
    except Exception as e:
        print(f"[Calibration] verification failed "
              f"({type(e).__name__}: {e}); calibration IoU fields stay null")
    # ------------------------------------------------------------------

    from scipy.spatial.transform import Rotation as SciRot
    z_up = np.array([0.0, 0.0, 1.0])

    if args.leveling == "legacy":
        # Level the scan: rotate so IMU-estimated gravity aligns with Z-up.
        # This is the original, working path — untouched by the floor-leveling
        # feature. In "floor" mode the pipeline has already leveled merged +
        # tracker and left us in a Z-up frame with the floor at Z=0.
        gravity_up = estimate_gravity(imus)
        if not np.allclose(gravity_up, z_up, atol=0.01):
            R_level = SciRot.align_vectors([z_up], [gravity_up])[0].as_matrix()
            print(f"[INFO] Leveling scan (gravity was {gravity_up})")

            # Lockstep transform: merged + wall_labels + objects.
            apply_transform_to_buffers(
                R_level,
                merged=merged,
                wall_labels=wall_labels,
                objects=objects,
            )
            if _pose_xyz.shape[0] > 0:
                _pose_xyz = _pose_xyz @ R_level.T

    # Align room to axes: detect wall direction on the LEVELED cloud
    from cloud_slam.room_structure import detect_room
    from cloud_slam.manhattan import estimate_manhattan_frame

    room_leveled = detect_room(merged, z_up)
    if room_leveled.walls:
        manhattan_leveled = estimate_manhattan_frame(room_leveled.walls, z_up)
        if manhattan_leveled.confidence > 0.3:
            # Manhattan X direction in the leveled frame
            mx = manhattan_leveled.R.T[:, 0]
            wall_yaw = np.arctan2(mx[1], mx[0])
            snapped_wall = round(wall_yaw / (np.pi / 2)) * (np.pi / 2)
            residual = wall_yaw - snapped_wall
            if abs(residual) > 0.01:
                R_align = SciRot.from_euler('z', -residual).as_matrix()
                print(f"[INFO] Aligning room to axes (rotating {np.degrees(-residual):.1f}° around Z)")
                # Lockstep transform: merged + wall_labels + objects.
                apply_transform_to_buffers(
                    R_align,
                    merged=merged,
                    wall_labels=wall_labels,
                    objects=objects,
                )
                if _pose_xyz.shape[0] > 0:
                    _pose_xyz = _pose_xyz @ R_align.T

            # Re-snap object quaternions to the leveled+aligned Manhattan directions.
            # NOTE: yaw-snap is a *quaternion-only* operation that rebuilds each
            # object's orientation from scratch (not a rotation applied to an
            # existing one), so it intentionally stays OUTSIDE
            # apply_transform_to_buffers() — it is not a lockstep transform.
            # The OBBs were fit in the raw world frame, so after R_level their yaw
            # doesn't match the leveled Manhattan. Fix by snapping each object's yaw
            # to the nearest 90° (which are now the aligned wall directions).
            for obj in objects:
                if 'orientation' not in obj:
                    continue
                q = np.array(obj['orientation']['quaternion'])
                R_obj = SciRot.from_quat(q).as_matrix()
                # Extract yaw (Z rotation in leveled+aligned frame)
                obj_yaw = np.arctan2(R_obj[1, 0], R_obj[0, 0])
                snapped_yaw = round(obj_yaw / (np.pi / 2)) * (np.pi / 2)
                # Rebuild quaternion: keep gravity alignment, fix yaw
                R_snapped = SciRot.from_euler('z', snapped_yaw).as_matrix()
                obj['orientation']['quaternion'] = SciRot.from_matrix(R_snapped).as_quat().tolist()

    # Save colored map
    map_path = os.path.join(args.output, "colored_map.ply")
    o3d.io.write_point_cloud(map_path, merged)
    print(f"[INFO] Saved colored map: {map_path} ({len(merged.points)} points)")

    # Post-process: run standalone RANSAC floor leveler on the merged cloud
    # in-memory (level.py's own PLY reader doesn't handle uchar RGB, so we
    # keep its algorithm verbatim but use Open3D for I/O).
    #
    # After producing the refined leveled PLY we also apply the same
    # rotation + z-shift to `merged` in place and to every object center /
    # quaternion, so that the downstream `objects.json` and
    # `map_with_boxes.ply` end up in the same "floor at Z=0" frame.
    try:
        from cloud_slam.level import detect_floor_plane, level_points
        print("[Level] Running RANSAC floor leveler on map...")
        xyz = np.asarray(merged.points)
        result = detect_floor_plane(xyz)
        if result is None:
            raise RuntimeError("No horizontal plane found")
        normal, _ = result
        leveled_xyz, _, z_shift = level_points(xyz, normal)

        # Re-derive the rotation matrix level_points computed internally
        # (it only returns the magnitude in degrees).
        R_level = SciRot.align_vectors(
            [np.array([0.0, 0.0, 1.0])], [normal])[0].as_matrix()

        # Write the refined leveled PLY as a separate file.
        leveled_pcd = o3d.geometry.PointCloud()
        leveled_pcd.points = o3d.utility.Vector3dVector(leveled_xyz.astype(np.float64))
        if merged.has_colors():
            leveled_pcd.colors = merged.colors
        leveled_path = os.path.join(args.output, "colored_map_leveled.ply")
        o3d.io.write_point_cloud(leveled_path, leveled_pcd)
        print(f"[Level] Saved leveled map: {leveled_path}")

        # Propagate the same transform to `merged` and to objects so that
        # the subsequently-saved objects.json and map_with_boxes.ply align
        # with the leveled cloud.
        # NOTE: `merged.points` is directly assigned from leveled_pcd (which
        # is in the new frame already), so we pass merged=None to the helper
        # here — rotating it again would double-apply. wall_labels / objects
        # still need R_level + shift_vec.
        merged.points = leveled_pcd.points
        shift_vec = np.array([0.0, 0.0, -z_shift])
        apply_transform_to_buffers(
            R_level,
            shift_vec,
            wall_labels=wall_labels,
            objects=objects,
        )
        if _pose_xyz.shape[0] > 0:
            _pose_xyz = (_pose_xyz @ R_level.T) + shift_vec
    except Exception as e:
        print(f"[WARNING] level.py post-process failed: {e}")

    # Post-process: 2D floor plan extraction.
    # Uses the ENHANCED algorithm (Stages 1-7 RANSAC wall refinement +
    # learned-dominant-angle tolerance snap, producing variant D_refined
    # with per-wall `snapped_to`/`residual_m`/`confidence` fields). Robust
    # floor/ceiling detection via detect_room() + IMU gravity prior — not
    # the fragile top-2 Z-histogram peaks that broke on furniture-heavy
    # rooms. Operates on the in-memory `merged` cloud (already Z-up).
    # Output: <scan_root>/results/<scan_name>/ (user-chosen convention).
    try:
        from cloud_slam.floorplan import generate_floorplan
        rosbag_parent = os.path.dirname(os.path.abspath(args.rosbag))
        scan_name = os.path.basename(rosbag_parent)
        scan_root = os.path.dirname(rosbag_parent)
        fp_out_dir = os.path.join(scan_root, "results", scan_name)
        os.makedirs(fp_out_dir, exist_ok=True)
        print(f"[Floorplan] Generating floor plan -> {fp_out_dir}")
        # Pass the vision-labeled buffer only when the --label-walls flag
        # actually produced labels; otherwise leave it None so the
        # floorplan path is byte-identical to the vision-less regression
        # baseline.
        fp_wall_labels = (wall_labels
                          if wall_labels is not None
                          and len(wall_labels.get('labels', [])) > 0
                          else None)
        # M0a calibration block: pull `method` / `calibration_date` from
        # the scan's extrinsics.yaml if present. Compute `age_days` from
        # today's date so downstream converters can flag drifted
        # calibrations. Any missing field falls back to M0a defaults.
        # M0c: `calibration_info` already loaded + enriched above (pre-
        # leveling verification) — we just pass it through. Strip the
        # internal `_verified_block` key so only the documented contract
        # leaks into the floorplan schema builder.
        calibration_info_fp = {
            k: v for k, v in calibration_info.items()
            if not k.startswith('_')
        }
        # M4a: gather scan-quality inputs — vision health from the
        # segmenter (None when --label-walls wasn't passed), time-sync
        # dts from the SLAM stats, and the dropped-frame count. Any
        # missing signal falls through as None / 0, and the floorplan
        # schema builder emits nulls — the JSON always carries the
        # block so consumers can unconditionally read it.
        vision_health = (wall_segmenter.get_vision_health()
                         if wall_segmenter is not None else None)
        time_sync_dts = stats.get('time_sync_dts')
        time_sync_dropped = int(stats.get('time_sync_dropped', 0) or 0)
        # M4b: forward SLAM trajectory poses so the floorplan stage can
        # drop next-room "phantom" walls outside the buffered hull of the
        # scanner's path. `_pose_xyz` was extracted from `poses` before
        # leveling and rotated in lockstep with every transform applied
        # to `merged` above — so its XY frame matches the final leveled
        # + aligned cloud the floorplan stage operates on.
        _variants, fp_meta = generate_floorplan(
            merged,
            fp_out_dir,
            name="floorplan",
            gravity_up=np.array([0.0, 0.0, 1.0]),
            wall_labels=fp_wall_labels,
            calibration_info=calibration_info_fp,
            run_stage8=not args.no_stage8,
            vision_health=vision_health,
            time_sync_dts=time_sync_dts,
            time_sync_dropped=time_sync_dropped,
            poses=_pose_xyz,
            verbose=False,
        )
        v = fp_meta['variants']
        print(f"[Floorplan] Saved ("
              f"A={v['A_natural']['area_m2']}m²/{v['A_natural']['n_walls']}w, "
              f"D={v['D_refined']['area_m2']}m²/{v['D_refined']['n_walls']}w)")
        if 'vision_wall_point_count' in fp_meta:
            print(f"[Floorplan] Vision: "
                  f"wall_pts={fp_meta['vision_wall_point_count']}, "
                  f"wall_blobs={fp_meta['vision_wall_blob_count']}")
    except Exception as e:
        print(f"[WARNING] floorplan.py post-process failed: {e}")

    # Save objects.json
    objects_path = os.path.join(args.output, "objects.json")
    output_json = {
        "format_version": "1.0",
        "coordinate_frame": "world",
        "stats": stats,
        "objects": objects,
    }
    with open(objects_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"[INFO] Saved {len(objects)} objects: {objects_path}")

    # Save map with box wireframes
    if objects:
        box_points_list = []
        box_colors_list = []
        for obj in objects:
            pts = create_box_points(
                np.array(obj['center']),
                np.array(obj['dimensions']),
                np.array(obj['orientation']['quaternion']),
            )
            box_points_list.append(pts)
            # Bright red for box edges
            box_colors_list.append(np.full((len(pts), 3), [1.0, 0.0, 0.0]))

        all_box_pts = np.concatenate(box_points_list)
        all_box_colors = np.concatenate(box_colors_list)

        # Combine map + box points
        map_pts = np.asarray(merged.points)
        map_colors = np.asarray(merged.colors) if merged.has_colors() else np.full((len(map_pts), 3), 0.5)

        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(np.vstack([map_pts, all_box_pts]))
        combined.colors = o3d.utility.Vector3dVector(np.vstack([map_colors, all_box_colors]))

        boxes_map_path = os.path.join(args.output, "map_with_boxes.ply")
        o3d.io.write_point_cloud(boxes_map_path, combined)
        print(f"[INFO] Saved map with boxes: {boxes_map_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE — Total: {t_total:.1f}s")
    print(f"  Points: {len(merged.points)}")
    print(f"  Objects detected: {len(objects)}")
    for obj in objects:
        dims = obj['dimensions']
        print(f"    [{obj['track_id']}] {obj['class']} at "
              f"({obj['center'][0]:.2f}, {obj['center'][1]:.2f}, {obj['center'][2]:.2f}) "
              f"size {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}m "
              f"({obj['num_observations']} obs)")
    print(f"{'='*60}")

    # M0c: loud warning if reprojection IoU was below 0.6. Emitted LAST so
    # it's the final thing the user sees — calibration drift silently caps
    # M2's wall-edge accuracy, and a mid-run message buried in floorplan
    # logs is easy to miss.
    _verified = calibration_info.get('_verified_block')
    if _verified is not None:
        try:
            from cloud_slam.calibration import should_warn, warning_message
            if should_warn(_verified):
                print()
                print(warning_message(_verified))
        except Exception:
            # Never let the warning path break the pipeline.
            pass

    # M4a: loud warning when the scan_quality block shows low-quality
    # vision, bad time-sync, or too many poorly-seen walls. Format
    # matches the M0c calibration banner for visual consistency.
    sq = {}
    try:
        sq = fp_meta.get('scan_quality', {}) if 'fp_meta' in locals() else {}
    except Exception:
        sq = {}
    try:
        if sq:
            _issues = []
            _v = sq.get('vision', {})
            _fq = _v.get('frame_quality_pct')
            if _fq is not None and float(_fq) < 30.0:
                _issues.append(
                    f"vision frame_quality_pct={_fq:.1f}% (<30%) — "
                    "most pixels labeled 'other'; check camera/lighting")
            _t = sq.get('time_sync', {})
            _p95 = _t.get('p95_ms')
            if _p95 is not None and float(_p95) > 120.0:
                _issues.append(
                    f"time_sync p95_ms={_p95:.1f} ms (>120 ms) — "
                    "approaching the 150 ms hard wall")
            _w = sq.get('walls_camera_coverage', {})
            _nlow = _w.get('n_walls_with_low_coverage', 0)
            if int(_nlow or 0) > 2:
                _issues.append(
                    f"walls_camera_coverage.n_walls_with_low_coverage="
                    f"{_nlow} (>2) — too many walls seen in <10 frames")
            if _issues:
                print()
                print("!" * 60)
                print("!!  SCAN QUALITY WARNING (M4a)")
                for _i in _issues:
                    print(f"!!  - {_i}")
                print("!!  Inspect floorplan_metadata.json → scan_quality "
                      "for details.")
                print("!" * 60)
    except Exception:
        # Never let the warning path break the pipeline.
        pass


if __name__ == "__main__":
    main()
