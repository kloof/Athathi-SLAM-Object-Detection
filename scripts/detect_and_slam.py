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
import argparse
import numpy as np

# Add parent dir so cloud_slam imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cloud_slam.box_render import create_box_points


def main():
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

    t_total = time.time() - t0

    import open3d as o3d
    from scipy.spatial.transform import Rotation as SciRot
    z_up = np.array([0.0, 0.0, 1.0])

    if args.leveling == "legacy":
        # Level the scan: rotate so IMU-estimated gravity aligns with Z-up.
        # This is the original, working path — untouched by the floor-leveling
        # feature. In "floor" mode the pipeline has already leveled merged +
        # tracker and left us in a Z-up frame with the floor at Z=0.
        from cloud_slam.frustum import estimate_gravity

        gravity_up = estimate_gravity(imus)
        if not np.allclose(gravity_up, z_up, atol=0.01):
            R_level = SciRot.align_vectors([z_up], [gravity_up])[0].as_matrix()
            print(f"[INFO] Leveling scan (gravity was {gravity_up})")

            # Rotate the point cloud
            merged.rotate(R_level, center=(0, 0, 0))

            # Rotate vision-labeled points in lockstep
            if wall_labels is not None and len(wall_labels['xyz']) > 0:
                wall_labels['xyz'] = (
                    wall_labels['xyz'].astype(np.float64) @ R_level.T
                ).astype(np.float32)

            # Rotate all object centers and orientations
            for obj in objects:
                c = np.array(obj['center'])
                obj['center'] = (R_level @ c).tolist()
                if 'orientation' in obj:
                    q = np.array(obj['orientation']['quaternion'])
                    R_obj = SciRot.from_quat(q).as_matrix()
                    R_new = R_level @ R_obj
                    obj['orientation']['quaternion'] = SciRot.from_matrix(R_new).as_quat().tolist()

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
                merged.rotate(R_align, center=(0, 0, 0))
                # Rotate vision-labeled points in lockstep
                if wall_labels is not None and len(wall_labels['xyz']) > 0:
                    wall_labels['xyz'] = (
                        wall_labels['xyz'].astype(np.float64) @ R_align.T
                    ).astype(np.float32)
                for obj in objects:
                    c = np.array(obj['center'])
                    obj['center'] = (R_align @ c).tolist()
                    if 'orientation' in obj:
                        q = np.array(obj['orientation']['quaternion'])
                        R_obj = SciRot.from_quat(q).as_matrix()
                        R_new = R_align @ R_obj
                        obj['orientation']['quaternion'] = SciRot.from_matrix(R_new).as_quat().tolist()

            # Re-snap object quaternions to the leveled+aligned Manhattan directions.
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
        merged.points = leveled_pcd.points
        shift_vec = np.array([0.0, 0.0, -z_shift])
        # Propagate to vision-labeled points as well so the floorplan
        # receives labels in the same coordinate frame as the merged cloud.
        if wall_labels is not None and len(wall_labels['xyz']) > 0:
            wall_labels['xyz'] = (
                (wall_labels['xyz'].astype(np.float64) @ R_level.T)
                + shift_vec
            ).astype(np.float32)
        for obj in objects:
            c = np.array(obj['center'])
            obj['center'] = (R_level @ c + shift_vec).tolist()
            if 'orientation' in obj:
                q = np.array(obj['orientation']['quaternion'])
                R_obj = SciRot.from_quat(q).as_matrix()
                R_new = R_level @ R_obj
                obj['orientation']['quaternion'] = (
                    SciRot.from_matrix(R_new).as_quat().tolist())
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
        _variants, fp_meta = generate_floorplan(
            merged,
            fp_out_dir,
            name="floorplan",
            gravity_up=np.array([0.0, 0.0, 1.0]),
            wall_labels=fp_wall_labels,
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


if __name__ == "__main__":
    main()
