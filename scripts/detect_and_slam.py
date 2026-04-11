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

    # Run pipeline
    from cloud_slam.pipelines.detect_pipeline import run
    merged, poses, objects, stats = run(
        clouds, imus, images, calib,
        voxel_size=args.voxel_size,
        detector_config=detector_config,
    )

    t_total = time.time() - t0

    # Save colored map
    import open3d as o3d
    map_path = os.path.join(args.output, "colored_map.ply")
    o3d.io.write_point_cloud(map_path, merged)
    print(f"[INFO] Saved colored map: {map_path} ({len(merged.points)} points)")

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
