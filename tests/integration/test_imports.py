"""Smoke test: every public module of cloud_slam imports cleanly.

Guards against import-time errors introduced by the M0d package split
and future milestones' additions.
"""
import pytest


def test_floorplan_package_imports():
    from cloud_slam.floorplan import generate_floorplan, SCHEMA_VERSION
    assert generate_floorplan is not None
    assert SCHEMA_VERSION is not None


def test_floorplan_submodules_import():
    from cloud_slam.floorplan import render, refine, openings, schema, config
    assert render is not None
    assert refine is not None
    assert openings is not None
    assert schema is not None
    assert config is not None


def test_box_render_imports():
    from cloud_slam.box_render import create_box_points
    assert create_box_points is not None


def test_scripts_detect_and_slam_imports_cleanly():
    # Importing the module without running main() should succeed.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "detect_and_slam_mod", "scripts/detect_and_slam.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
    assert hasattr(mod, "create_box_points")
