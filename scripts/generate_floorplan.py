#!/usr/bin/env python3
"""
Standalone floor-plan extraction from a LiDAR point cloud.

Wraps cloud_slam.floorplan.generate_floorplan. Accepts an already-leveled
.ply (the output of detect_and_slam.py is always Z-up) and optionally an
original rosbag for the IMU gravity prior.

Usage:
    python3 scripts/generate_floorplan.py IN.ply -o OUT_DIR/
    python3 scripts/generate_floorplan.py IN.ply -o OUT_DIR/ --rosbag PATH
    python3 scripts/generate_floorplan.py IN.ply --resolution 0.03 --snap 45
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description='Extract a 2D floor plan (PNG + JSON) from a LiDAR '
                    'point cloud. Floor/ceiling detection uses RANSAC + '
                    'IMU gravity (robust to furniture peaks).')
    parser.add_argument('input', help='Input PLY or PCD file')
    parser.add_argument('-o', '--output', default='output/',
                        help='Output directory (default: output/)')
    parser.add_argument('--name', default=None,
                        help='Output file basename (default: derived from input)')
    parser.add_argument('--rosbag', default=None,
                        help='Optional MCAP/rosbag path for IMU gravity prior. '
                             'Not needed for clouds produced by detect_and_slam.py '
                             '(those are already Z-up). If omitted, gravity_up '
                             'is assumed to be [0, 0, 1] — correct only for '
                             'already-leveled clouds.')
    parser.add_argument('--resolution', type=float, default=0.03,
                        help='Grid resolution (m). Default: 0.03')
    parser.add_argument('--epsilon', type=float, default=0.012,
                        help='Contour simplification ratio. Default: 0.012')
    parser.add_argument('--snap', type=float, default=45,
                        help='Angle snap (deg). 0 disables. Default: 45')
    parser.add_argument('--voxel', type=float, default=0.03,
                        help='Voxel downsample (m). Default: 0.03')
    parser.add_argument('--ceil-band', type=float, default=0.12,
                        help='Ceiling Z band +/- m. Default: 0.12')
    parser.add_argument('--close-kernel', type=int, default=11,
                        help='Morph close kernel. Default: 11')
    parser.add_argument('--angle-flex', type=float, default=3.0,
                        help='Max deviation from snap grid. Default: 3.0')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    import open3d as o3d
    from cloud_slam.floorplan import generate_floorplan

    if not args.quiet:
        print(f"Loading {args.input} ...")
    pcd = o3d.io.read_point_cloud(args.input)
    if len(pcd.points) == 0:
        print(f"Error: no points in {args.input} "
              f"(file exists but is empty or unreadable)", file=sys.stderr)
        sys.exit(1)

    # IMU prior (optional)
    imus = None
    gravity_up = None
    if args.rosbag:
        from cloud_slam.mcap_reader import read_mcap
        if not args.quiet:
            print(f"Reading IMU from {args.rosbag} ...")
        _clouds, imus, _images = read_mcap(args.rosbag)
        if not args.quiet:
            print(f"  got {len(imus)} IMU samples")
    else:
        # Clouds from detect_and_slam.py are already leveled to Z-up.
        gravity_up = np.array([0.0, 0.0, 1.0])

    name = args.name or os.path.splitext(os.path.basename(args.input))[0]

    generate_floorplan(
        pcd,
        args.output,
        name=name,
        gravity_up=gravity_up,
        imus=imus,
        resolution=args.resolution,
        epsilon=args.epsilon,
        snap_angle=args.snap,
        voxel_size=args.voxel,
        ceil_band=args.ceil_band,
        close_kernel=args.close_kernel,
        angle_flex=args.angle_flex,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
