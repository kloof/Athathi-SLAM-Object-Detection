"""
Per-class size priors for indoor furniture with Bayesian refinement.

Uses typical furniture dimensions to correct noisy bounding boxes
from frustum-based 3D detection.
"""

import numpy as np
from typing import Optional, Tuple

# Per-class size priors: [width, depth, height] in meters
# Width >= depth by convention. Height is gravity-aligned.
SIZE_PRIORS = {
    "chair":   {"mean": [0.50, 0.50, 0.85], "std": [0.10, 0.10, 0.15]},
    "table":   {"mean": [1.20, 0.70, 0.75], "std": [0.40, 0.20, 0.05]},
    "desk":    {"mean": [1.20, 0.60, 0.75], "std": [0.30, 0.15, 0.05]},
    "sofa":    {"mean": [2.00, 0.85, 0.85], "std": [0.50, 0.15, 0.15]},
    "bed":     {"mean": [2.00, 1.50, 0.55], "std": [0.20, 0.30, 0.15]},
    "shelf":   {"mean": [0.80, 0.40, 1.50], "std": [0.40, 0.10, 0.50]},
    "monitor": {"mean": [0.55, 0.05, 0.35], "std": [0.20, 0.03, 0.10]},
    "door":    {"mean": [0.85, 0.05, 2.05], "std": [0.15, 0.02, 0.15]},
    "person":  {"mean": [0.45, 0.30, 1.70], "std": [0.10, 0.10, 0.20]},
    "lamp":    {"mean": [0.30, 0.30, 0.50], "std": [0.15, 0.15, 0.30]},
    "plant":   {"mean": [0.40, 0.40, 0.60], "std": [0.20, 0.20, 0.40]},
}

# Classes that sit on the floor
FLOOR_CONTACT = {"chair", "table", "desk", "sofa", "bed", "shelf", "door", "person", "plant"}

# Classes that typically sit against a wall
WALL_ADJACENT = {"shelf", "desk", "monitor", "door"}


def assign_dimensions(obb_dims, obb_rotation, gravity_up):
    """
    Map 3 OBB extents to [width, depth, height] using gravity alignment.

    Height = axis most aligned with gravity.
    Width = larger of the two horizontal extents.
    Depth = smaller horizontal extent.

    Returns:
        (ordered_dims, axis_mapping) where axis_mapping[i] = original OBB axis index
    """
    # Find which OBB axis aligns with gravity
    if obb_rotation is not None and obb_rotation.shape == (3, 3):
        dots = np.abs(obb_rotation.T @ gravity_up)
    else:
        dots = np.abs(gravity_up)

    h_idx = int(np.argmax(dots))
    horiz = [i for i in range(3) if i != h_idx]

    if obb_dims[horiz[0]] >= obb_dims[horiz[1]]:
        w_idx, d_idx = horiz[0], horiz[1]
    else:
        w_idx, d_idx = horiz[1], horiz[0]

    ordered = np.array([obb_dims[w_idx], obb_dims[d_idx], obb_dims[h_idx]])
    return ordered, (w_idx, d_idx, h_idx)


def refine_dimensions(observed, class_name, num_points=100, confidence=0.5):
    """
    Bayesian refinement of observed dimensions using class priors.

    Few points → trust prior more. Many points → trust observation.

    Args:
        observed: (3,) [width, depth, height] in meters
        class_name: YOLOE class label
        num_points: number of lidar points in accumulated frustum
        confidence: YOLOE detection confidence

    Returns:
        (refined_dims, sigma_deviation)
    """
    if class_name not in SIZE_PRIORS:
        return observed.copy(), np.zeros(3)

    prior = SIZE_PRIORS[class_name]
    prior_mean = np.array(prior["mean"])
    prior_std = np.array(prior["std"])

    # Observation uncertainty: inversely proportional to sqrt(points) and confidence
    point_factor = max(1.0, np.sqrt(num_points / 50.0))
    conf_factor = max(0.3, confidence)
    obs_std = (0.3 * np.maximum(observed, 0.05)) / (point_factor * conf_factor)
    obs_std = np.maximum(obs_std, 0.05)  # floor at 5cm

    # Bayesian: posterior = weighted combination
    prior_prec = 1.0 / (prior_std ** 2)
    obs_prec = 1.0 / (obs_std ** 2)
    posterior_mean = (prior_prec * prior_mean + obs_prec * observed) / (prior_prec + obs_prec)

    sigma_dev = (observed - prior_mean) / prior_std

    return posterior_mean, sigma_dev


def validate_dimensions(dims, class_name, sigma_threshold=3.0):
    """
    Check if dimensions are physically plausible for the class.

    Returns:
        (is_valid, rejection_reason)
    """
    # Hard limits
    if np.any(dims > 4.0):
        return False, f"dimension exceeds 4m: {dims}"
    if np.any(dims < 0.01):
        return False, f"dimension below 1cm: {dims}"

    # Volume sanity check — reject impossibly small objects
    volume = float(np.prod(dims))
    min_volumes = {
        "chair": 0.05, "table": 0.05, "desk": 0.05, "sofa": 0.10,
        "bed": 0.10, "shelf": 0.02, "monitor": 0.001, "door": 0.01,
        "person": 0.02, "lamp": 0.005, "plant": 0.005,
    }
    min_vol = min_volumes.get(class_name, 0.01)
    if volume < min_vol:
        return False, f"volume {volume:.4f}m³ below minimum {min_vol}m³ for {class_name}"

    if class_name not in SIZE_PRIORS:
        return True, None

    prior = SIZE_PRIORS[class_name]
    prior_mean = np.array(prior["mean"])
    prior_std = np.array(prior["std"])

    bad_count = 0
    for i in range(3):
        dev = abs(dims[i] - prior_mean[i]) / prior_std[i]
        if dev > sigma_threshold:
            bad_count += 1

    if bad_count >= 3:
        return False, f"all dimensions deviate > {sigma_threshold}σ from {class_name} prior"
    if bad_count >= 2 and np.max(np.abs((dims - prior_mean) / prior_std)) > 5.0:
        return False, f"multiple extreme deviations for {class_name}"

    return True, None


def is_floor_contact(class_name):
    """Check if this class should sit on the floor."""
    return class_name in FLOOR_CONTACT


def is_wall_adjacent(class_name):
    """Check if this class typically sits against a wall."""
    return class_name in WALL_ADJACENT
