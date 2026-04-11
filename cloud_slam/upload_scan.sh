#!/bin/bash
# Upload MCAP to Cloud SLAM service and download result.
#
# Usage: ./upload_scan.sh /path/to/rosbag_dir [server_url]
#
# Server defaults to localhost:8080

BAG_DIR="${1:?Usage: ./upload_scan.sh /path/to/rosbag_dir [server_url]}"
SERVER="${2:-http://localhost:8080}"

# Find MCAP file
MCAP_FILE=$(ls "${BAG_DIR}"/*.mcap 2>/dev/null | head -1)
if [ -z "$MCAP_FILE" ]; then
    echo "[ERROR] No .mcap file found in ${BAG_DIR}"
    exit 1
fi

FILE_SIZE=$(du -sh "$MCAP_FILE" | cut -f1)
echo "============================================"
echo "  Cloud SLAM Upload"
echo "============================================"
echo "  File:   ${MCAP_FILE} (${FILE_SIZE})"
echo "  Server: ${SERVER}"
echo ""

# Compress and upload
echo "[INFO] Compressing and uploading..."
RESPONSE=$(gzip -c "$MCAP_FILE" | curl -s -X POST \
    "${SERVER}/api/slam" \
    -F "file=@-;filename=scan.mcap.gz")

echo "[INFO] Server response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# Check if we got a download URL (GCS mode) or a direct file
if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('download_url',''))" 2>/dev/null | grep -q "http"; then
    # GCS mode — download from signed URL
    URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['download_url'])")
    OUTPUT_FILE="${BAG_DIR}/cloud_result.ply"
    echo ""
    echo "[INFO] Downloading result from GCS..."
    wget -q "$URL" -O "$OUTPUT_FILE"
    echo "[DONE] Saved: ${OUTPUT_FILE}"
else
    # Direct mode — response IS the file (save the curl output differently)
    OUTPUT_FILE="${BAG_DIR}/cloud_result.ply"
    gzip -c "$MCAP_FILE" | curl -s -X POST \
        "${SERVER}/api/slam" \
        -F "file=@-;filename=scan.mcap.gz" \
        -o "$OUTPUT_FILE"
    echo "[DONE] Saved: ${OUTPUT_FILE}"
fi

if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    RESULT_SIZE=$(du -sh "$OUTPUT_FILE" | cut -f1)
    echo "[INFO] Result size: ${RESULT_SIZE}"
fi
