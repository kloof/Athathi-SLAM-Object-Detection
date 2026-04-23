"""SpatialLM input-preparation + inference pipeline.

End-to-end: rosbag -> KISS-ICP SLAM -> floorplan-based crop -> Manhattan
yaw-align -> colored-priority 1cm voxel -> K-NN gray-color interpolation
-> SpatialLM 1.1 inference (Qwen structure + Llama objects merged) ->
colored PLY with embedded wireframe bounding boxes.

Entry points:
    cloud_slam.spatiallm_pipeline.crop:           crop_to_floorplan()
    cloud_slam.spatiallm_pipeline.manhattan:      yaw_align_by_walls()
    cloud_slam.spatiallm_pipeline.voxel:          colored_priority_voxel()
    cloud_slam.spatiallm_pipeline.interpolate:    interpolate_gray_colors()
    cloud_slam.spatiallm_pipeline.infer:          run_spatiallm()
    cloud_slam.spatiallm_pipeline.merge:          merge_layouts()
    cloud_slam.spatiallm_pipeline.embed:          embed_bboxes_in_ply()

CLI: scripts/rosbag_to_bboxes.py
"""
