#!/bin/bash
# ============================================================
# Post-Processing Pipeline: rosbag -> FAST-LIO2 -> Merge -> Final PCD
#
# Usage: ./process_scan.sh /path/to/rosbag_directory [playback_rate]
#
# The rosbag directory should contain metadata.yaml + .mcap files
# playback_rate defaults to 1.0 (realtime)
#
# Output:
#   <rosbag_dir>/../slam_output/    - per-frame PCD + trajectory
#   <rosbag_dir>/../final_map.pcd   - merged point cloud
#   <rosbag_dir>/../final_map.ply   - PLY copy
# ============================================================

set -e

BAG_PATH="${1:?Usage: ./process_scan.sh /path/to/rosbag_directory [playback_rate]}"
RATE="${2:-5.0}"
SESSION_DIR="$(cd "$(dirname "$BAG_PATH")" && pwd)"
BAG_PATH="$(cd "$BAG_PATH" 2>/dev/null && pwd || echo "$BAG_PATH")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate bag path
if [ ! -f "${BAG_PATH}/metadata.yaml" ]; then
    echo "[ERROR] No metadata.yaml found in ${BAG_PATH}"
    exit 1
fi

echo "============================================"
echo "  LiDAR SLAM Post-Processing Pipeline"
echo "============================================"
echo "[INFO] Rosbag: ${BAG_PATH}"
echo "[INFO] Output: ${SESSION_DIR}"
echo "[INFO] Rate:   ${RATE}x"
echo ""

# Source ROS2
source /opt/ros/humble/setup.bash
source ~/lidar_slam/slam_ws/install/setup.bash

# FAST-LIO2 writes to its source dir
FAST_LIO_SRC="${HOME}/lidar_slam/slam_ws/src/FAST_LIO"
SLAM_OUTPUT="${SESSION_DIR}/slam_output"
mkdir -p "${SLAM_OUTPUT}"

# Clean previous FAST-LIO2 output
rm -f "${FAST_LIO_SRC}/PCD/"*.pcd 2>/dev/null || true
rm -f "${FAST_LIO_SRC}/Log/pos_log.txt" 2>/dev/null || true

# ============================================================
# STAGE 1: Run FAST-LIO2
# ============================================================
echo "========== STAGE 1: FAST-LIO2 SLAM =========="

ros2 launch fast_lio mapping.launch.py \
    config_file:=unitree_l2.yaml \
    use_sim_time:=true \
    rviz:=false \
    &
SLAM_PID=$!
sleep 5

echo "[INFO] Playing rosbag at ${RATE}x..."
ros2 bag play "${BAG_PATH}" --clock --rate "${RATE}" 2>&1 | grep -v "^$" || true

echo "[INFO] Waiting for SLAM to finish processing..."
sleep 8
kill $SLAM_PID 2>/dev/null || true
wait $SLAM_PID 2>/dev/null || true

# Copy output
PCD_COUNT=$(ls "${FAST_LIO_SRC}/PCD/"*.pcd 2>/dev/null | wc -l)
echo "[INFO] Generated ${PCD_COUNT} per-frame PCD files"

if [ "$PCD_COUNT" -eq 0 ]; then
    echo "[ERROR] No PCD files generated. Check FAST-LIO2 output."
    exit 1
fi

cp "${FAST_LIO_SRC}/PCD/"*.pcd "${SLAM_OUTPUT}/"
cp "${FAST_LIO_SRC}/Log/pos_log.txt" "${SLAM_OUTPUT}/" 2>/dev/null || true

POSE_COUNT=$(wc -l < "${SLAM_OUTPUT}/pos_log.txt" 2>/dev/null || echo "0")
echo "[INFO] Trajectory: ${POSE_COUNT} poses"
echo "[DONE] FAST-LIO2 complete"

# ============================================================
# STAGE 2: Merge + Refine
# ============================================================
echo ""
echo "========== STAGE 2: MERGE + REFINE =========="
python3 "${SCRIPT_DIR}/merge_and_refine.py" "${SLAM_OUTPUT}" "${SESSION_DIR}"

# ============================================================
# STAGE 3: Verify (optional)
# ============================================================
echo ""
echo "========== STAGE 3: QUALITY CHECK =========="
python3 "${SCRIPT_DIR}/verify_scan.py" "${SESSION_DIR}/final_map.pcd"

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE"
echo "============================================"
echo "  Final map PCD: ${SESSION_DIR}/final_map.pcd"
echo "  Final map PLY: ${SESSION_DIR}/final_map.ply"
echo "  Per-frame PCDs: ${SLAM_OUTPUT}/"
echo "  Trajectory:     ${SLAM_OUTPUT}/pos_log.txt"
echo ""
echo "  View in CloudCompare: cloudcompare ${SESSION_DIR}/final_map.pcd"
echo "============================================"
