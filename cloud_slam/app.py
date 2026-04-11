"""
Cloud SLAM Service — Single endpoint, GCS-backed file storage.

POST /api/slam             — Upload MCAP directly (up to ~32MB due to Cloud Run proxy)
POST /api/slam/from-gcs    — Process MCAP already uploaded to GCS (no size limit)
GET  /api/upload-url       — Get a pre-signed upload URL for large files
GET  /api/result/{id}      — Get download URL for existing result
GET  /api/health           — Health check
"""

import io
import os
import gzip
import json
import time
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mcap_reader import read_mcap
from pipelines import icp_imu_pipeline

app = FastAPI(title="Cloud SLAM Service", version="2.1.0")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
job_stats = {}


def get_gcs_client():
    from google.cloud import storage
    return storage.Client()


def upload_to_gcs(local_path, blob_name):
    """Upload file to GCS and return a public URL."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


def download_from_gcs(blob_name, local_path):
    """Download a file from GCS to local path."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise HTTPException(404, f"File not found in GCS: {blob_name}")
    blob.download_to_filename(local_path)


def get_public_url(blob_name):
    """Get public URL for an existing blob."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    return blob.public_url


def generate_upload_url(blob_name):
    """Generate a resumable upload URL for large files."""
    from datetime import timedelta
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    # Use resumable upload URL (works without signing credentials)
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{GCS_BUCKET}/o?uploadType=media&name={blob_name}"
    return url


def process_mcap_file(mcap_local_path, job_id, voxel_size, output_format, t_start, extra_stats=None):
    """Core processing logic shared by both upload paths."""
    tmpdir = str(Path(mcap_local_path).parent)

    # Decompress if gzipped
    raw_path = Path(tmpdir) / "scan.mcap"
    with open(mcap_local_path, "rb") as f:
        header = f.read(2)

    if header == b'\x1f\x8b':
        with gzip.open(mcap_local_path, "rb") as gz_in:
            with open(raw_path, "wb") as f_out:
                shutil.copyfileobj(gz_in, f_out)
        input_size = raw_path.stat().st_size
    else:
        raw_path = Path(mcap_local_path)
        input_size = Path(mcap_local_path).stat().st_size

    t_decompress = time.time() - t_start

    # Read MCAP
    clouds, imus = read_mcap(raw_path)
    if not clouds:
        raise HTTPException(400, "No point cloud data found in MCAP file")

    t_read = time.time() - t_start

    # Run SLAM
    merged, poses, stats = icp_imu_pipeline.run(clouds, imus, voxel_size=voxel_size)

    # Write output
    import open3d as o3d
    out_path = Path(tmpdir) / f"result.{output_format}"
    o3d.io.write_point_cloud(str(out_path), merged)
    raw_size = out_path.stat().st_size

    # Gzip compress the output
    gz_path = Path(tmpdir) / f"result.{output_format}.gz"
    with open(out_path, "rb") as f_in:
        with gzip.open(str(gz_path), "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_size = gz_path.stat().st_size

    stats["job_id"] = job_id
    stats["decompress_time_s"] = round(t_decompress, 2)
    stats["mcap_read_time_s"] = round(t_read - t_decompress, 2)
    stats["output_format"] = output_format
    stats["input_size_mb"] = round(input_size / 1e6, 2)
    stats["output_size_mb"] = round(raw_size / 1e6, 2)
    stats["output_compressed_mb"] = round(gz_size / 1e6, 2)

    if extra_stats:
        stats.update(extra_stats)

    if GCS_BUCKET:
        blob_name_gz = f"results/{job_id}.{output_format}.gz"
        upload_to_gcs(str(gz_path), blob_name_gz)

        blob_name_raw = f"results/{job_id}.{output_format}"
        upload_to_gcs(str(out_path), blob_name_raw)

        stats["storage"] = "gcs"
        stats["blob_name"] = blob_name_raw
        stats["download_url"] = get_public_url(blob_name_raw)
        stats["download_url_gz"] = get_public_url(blob_name_gz)
        stats["total_time_s"] = round(time.time() - t_start, 2)

        job_stats[job_id] = stats
        return JSONResponse(stats)
    else:
        stats["storage"] = "direct"
        stats["total_time_s"] = round(time.time() - t_start, 2)

        result_bytes = gz_path.read_bytes()
        filename = f"slam_{job_id}.{output_format}.gz"
        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-SLAM-Stats": json.dumps(stats),
            },
        )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "algorithm": "Open3D point-to-plane ICP + IMU gyro integration",
        "storage": "gcs" if GCS_BUCKET else "direct",
        "endpoints": {
            "POST /api/slam": "Direct upload (files up to ~32MB, or gzip larger ones under 32MB)",
            "POST /api/slam/from-gcs": "Process from GCS (no size limit — upload to GCS first)",
            "GET /api/upload-url": "Get a GCS upload URL for large files",
            "GET /api/result/{job_id}": "Get download URLs for a completed job",
        },
    }


@app.get("/api/upload-url")
def get_upload_url():
    """
    Get a GCS path to upload a large MCAP file directly to the bucket.
    Use gsutil or curl to upload, then call POST /api/slam/from-gcs with the blob name.
    """
    if not GCS_BUCKET:
        raise HTTPException(501, "GCS not configured")

    upload_id = uuid.uuid4().hex[:12]
    blob_name = f"uploads/{upload_id}.mcap"

    return {
        "upload_id": upload_id,
        "blob_name": blob_name,
        "bucket": GCS_BUCKET,
        "gsutil_command": f"gsutil cp your_scan.mcap gs://{GCS_BUCKET}/{blob_name}",
        "curl_command": f"gzip -c your_scan.mcap | gsutil cp - gs://{GCS_BUCKET}/{blob_name}.gz",
        "then_process": f"curl -X POST '{{}}/api/slam/from-gcs?blob_name={blob_name}'",
    }


@app.post("/api/slam")
async def process_slam_direct(
    request: Request,
    file: UploadFile = File(...),
    voxel_size: float = Query(0.005, ge=0.001, le=0.1),
    output_format: str = Query("ply", pattern="^(ply|pcd)$"),
):
    """Direct file upload — works for files up to ~32MB (Cloud Run proxy limit).
    For larger files, gzip them first or use /api/slam/from-gcs."""
    job_id = uuid.uuid4().hex[:12]
    t_start = time.time()

    tmpdir = tempfile.mkdtemp(prefix="slam_")
    try:
        mcap_path = Path(tmpdir) / "upload.bin"
        total_bytes = 0
        with open(mcap_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                total_bytes += len(chunk)
                if total_bytes > 1024 * 1024 * 1024:
                    raise HTTPException(413, "File exceeds 1GB limit")

        t_upload = time.time() - t_start
        return process_mcap_file(
            str(mcap_path), job_id, voxel_size, output_format, t_start,
            extra_stats={"upload_time_s": round(t_upload, 2), "upload_method": "direct"}
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/api/slam/from-gcs")
def process_slam_from_gcs(
    blob_name: str = Query(..., description="GCS blob name (e.g., uploads/abc123.mcap)"),
    voxel_size: float = Query(0.005, ge=0.001, le=0.1),
    output_format: str = Query("ply", pattern="^(ply|pcd)$"),
):
    """Process an MCAP file already uploaded to GCS. No size limit."""
    if not GCS_BUCKET:
        raise HTTPException(501, "GCS not configured")

    job_id = uuid.uuid4().hex[:12]
    t_start = time.time()

    tmpdir = tempfile.mkdtemp(prefix="slam_")
    try:
        # Download from GCS
        local_path = Path(tmpdir) / "scan_from_gcs.mcap"
        download_from_gcs(blob_name, str(local_path))
        t_download = time.time() - t_start

        return process_mcap_file(
            str(local_path), job_id, voxel_size, output_format, t_start,
            extra_stats={"gcs_download_time_s": round(t_download, 2), "upload_method": "gcs"}
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    """Get download URLs for a previously processed scan."""
    if not GCS_BUCKET:
        raise HTTPException(501, "GCS not configured")

    stats = job_stats.get(job_id)
    if not stats:
        raise HTTPException(404, f"Job {job_id} not found")

    blob_name = stats.get("blob_name")
    if not blob_name:
        raise HTTPException(404, "No blob name recorded")

    url = get_public_url(blob_name)
    if not url:
        raise HTTPException(404, "Result file not found in GCS")

    stats["download_url"] = url
    blob_gz = blob_name + ".gz"
    gz_url = get_public_url(blob_gz)
    if gz_url:
        stats["download_url_gz"] = gz_url

    return JSONResponse(stats)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
