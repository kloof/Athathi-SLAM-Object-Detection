#!/usr/bin/env python3
"""
Standalone CLI: level a PLY, then run the floorplan pipeline on it.

Dead-simple wiring: level_ply(input) -> leveled PLY -> floorplan.run().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description='Level a PLY and extract a 2D floor plan (PNG + JSON).')
    parser.add_argument('input', help='Input PLY file')
    parser.add_argument('-o', '--output', default='output/',
                        help='Output directory (default: output/)')
    parser.add_argument('--name', default=None,
                        help='Output basename (default: from input filename)')
    parser.add_argument('--skip-level', action='store_true',
                        help='Skip the leveling step (use raw PLY as-is)')
    parser.add_argument('--level-distance-thresh', type=float, default=0.03,
                        help='RANSAC floor threshold for leveling. Default 0.03')
    # floorplan.py passthroughs
    parser.add_argument('--resolution', type=float, default=0.03)
    parser.add_argument('--epsilon', type=float, default=0.012)
    parser.add_argument('--snap', type=float, default=45)
    parser.add_argument('--voxel', type=float, default=0.03)
    parser.add_argument('--ceil-band', type=float, default=0.12)
    parser.add_argument('--close-kernel', type=int, default=11)
    parser.add_argument('--angle-flex', type=float, default=3.0)
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Error: input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    # 1. Level the PLY (writes a sibling _leveled.ply next to the input).
    if args.skip_level:
        leveled_path = args.input
    else:
        import open3d as o3d
        import numpy as np
        from cloud_slam.floorplan_level import level_ply
        base = args.name or os.path.splitext(os.path.basename(args.input))[0]
        # level.py's PLY reader chokes on mixed prop types (float x/y/z +
        # uchar r/g/b). Strip colors via Open3D first, then feed level.py
        # a clean xyz-only PLY.
        xyz_only_path = os.path.join(args.output, f'{base}_xyz.ply')
        _raw = o3d.io.read_point_cloud(args.input)
        _xyz = o3d.geometry.PointCloud()
        _xyz.points = _raw.points
        o3d.io.write_point_cloud(xyz_only_path, _xyz,
                                 write_ascii=False, compressed=False)
        leveled_path = os.path.join(args.output, f'{base}_leveled.ply')
        level_ply(xyz_only_path, output_path=leveled_path,
                  distance_thresh=args.level_distance_thresh)
        os.remove(xyz_only_path)

    # 2. Run the original floorplan pipeline on the leveled PLY.
    from cloud_slam.floorplan import run
    run(leveled_path, args.output,
        resolution=args.resolution, epsilon=args.epsilon,
        snap_angle=args.snap, voxel_size=args.voxel,
        ceil_band=args.ceil_band, close_kernel=args.close_kernel,
        angle_flex=args.angle_flex, verbose=not args.quiet)


if __name__ == '__main__':
    main()
