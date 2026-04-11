#!/usr/bin/env python3
"""
Visualize YOLO detections + LiDAR projection overlay on camera frames.

Produces a side-by-side video:
  Left:  YOLO detections (bboxes + masks + track IDs)
  Right: LiDAR points projected onto the camera image (colored by depth)

Usage:
    python3 scripts/visualize_detections.py <rosbag> <output_dir> <calibration_dir>
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Color palette for track IDs
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
]


def color_for_id(track_id):
    return COLORS[track_id % len(COLORS)]


def draw_detections(image, detections):
    """Draw bboxes, masks, class labels, and track IDs on the image."""
    vis = image.copy()
    for det in detections:
        color = color_for_id(det.track_id)
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)

        # Draw mask with transparency
        if det.mask is not None:
            h, w = vis.shape[:2]
            mask_resized = cv2.resize(det.mask.astype(np.float32), (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5
            overlay = vis.copy()
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)
            vis = overlay

        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def draw_lidar_projection(image, xyz, calib):
    """Project lidar points onto the camera image, colored by depth."""
    from cloud_slam.projection import project_lidar_to_camera

    vis = image.copy()
    if len(xyz) == 0:
        return vis

    pts_cam, pixels, in_front = project_lidar_to_camera(xyz, calib)
    W, H = calib['image_size']

    u = pixels[:, 0]
    v = pixels[:, 1]
    valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    if not valid.any():
        return vis

    u_valid = u[valid].astype(int)
    v_valid = v[valid].astype(int)
    depths = pts_cam[valid, 2]

    # Color by depth (close=red, far=blue)
    d_min, d_max = depths.min(), min(depths.max(), 8.0)
    if d_max - d_min < 0.1:
        d_max = d_min + 1.0
    norm_d = np.clip((depths - d_min) / (d_max - d_min), 0, 1)

    for i in range(len(u_valid)):
        t = norm_d[i]
        r = int(255 * (1 - t))
        g = int(255 * min(2 * t, 2 * (1 - t)))
        b = int(255 * t)
        cv2.circle(vis, (u_valid[i], v_valid[i]), 2, (b, g, r), -1)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detections + LiDAR projection"
    )
    parser.add_argument("rosbag", help="Path to rosbag directory or .mcap file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("calibration", help="Path to calibration directory")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLOE confidence threshold (default: 0.3)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Output video FPS (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load calibration
    from cloud_slam.colorizer import load_calibration
    calib = load_calibration(
        os.path.join(args.calibration, "intrinsics.yaml"),
        os.path.join(args.calibration, "extrinsics.yaml"),
    )
    W, H = calib['image_size']
    print(f"[INFO] Camera: {W}x{H}")

    # Read MCAP
    from cloud_slam.mcap_reader import read_mcap
    t0 = time.time()
    clouds, imus, images = read_mcap(args.rosbag)
    print(f"[INFO] Read {len(clouds)} clouds + {len(images)} images in {time.time()-t0:.1f}s")

    # Init detector
    from cloud_slam.detector import YOLODetector
    detector = YOLODetector(conf=args.conf)

    # Match images to nearest lidar cloud
    cloud_stamps = np.array([c[0] for c in clouds])

    # Video writers
    side_by_side_w = W * 2
    video_path = os.path.join(args.output, "detections.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, args.fps, (side_by_side_w, H))

    det_only_path = os.path.join(args.output, "detections_only.mp4")
    writer_det = cv2.VideoWriter(det_only_path, fourcc, args.fps, (W, H))

    print(f"[INFO] Processing {len(images)} frames...")
    n_det_total = 0

    for i, (stamp, compressed, fmt) in enumerate(images):
        # Decode image
        arr = np.frombuffer(compressed, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Run YOLO
        detections = detector.detect_and_track(frame)
        n_det_total += len(detections)

        # Draw detections
        det_vis = draw_detections(frame, detections)

        # Find nearest lidar cloud
        idx = np.argmin(np.abs(cloud_stamps - stamp))
        dt = abs(cloud_stamps[idx] - stamp)
        if dt < 0.15:
            xyz = clouds[idx][1]
        else:
            xyz = np.empty((0, 3))

        # Draw lidar projection
        lidar_vis = draw_lidar_projection(frame, xyz, calib)

        # Side-by-side
        # Add labels
        cv2.putText(det_vis, "YOLO Detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(det_vis, f"Frame {i}/{len(images)}  |  {len(detections)} objects",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(lidar_vis, "LiDAR Projection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2, cv2.LINE_AA)
        if len(xyz) > 0:
            cv2.putText(lidar_vis, f"{len(xyz)} points  |  dt={dt*1000:.0f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        combined = np.hstack([det_vis, lidar_vis])
        writer.write(combined)
        writer_det.write(det_vis)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(images)}] {n_det_total} detections so far")

    writer.release()
    writer_det.release()

    print(f"\n[DONE] {n_det_total} total detections across {len(images)} frames")
    print(f"  Side-by-side video: {video_path}")
    print(f"  Detection-only video: {det_only_path}")


if __name__ == "__main__":
    main()
