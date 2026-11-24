"""Fit the visible right-hand pointer as a ray in controller space."""

from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spatial_geometry import stereo_context
from targets import TargetSolver
from world_map import BASELINE, K


def _beam_line(image):
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (82, 45, 90), (106, 255, 255))
    height, width = mask.shape
    mask[:round(height * .64)] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, 21, minLineLength=44, maxLineGap=10)
    if lines is None:
        raise ValueError("The right-hand beam could not be traced in both eyes")
    candidates = []
    for x, y, u, v in lines.reshape(-1, 4):
        if y > v:
            x, y, u, v = u, v, x, y
        if v-y < 27 or u < width*.50 or v < height*.74:
            continue
        candidates.append((np.hypot(u-x, v-y), x, y, u, v))
    if not candidates:
        raise ValueError("Could not distinguish the right-hand beam from other cyan features")
    _, x, y, u, v = max(candidates)
    line = np.cross([x, y, 1.], [u, v, 1.])
    rows, cols = np.nonzero(mask)
    near = np.abs(line[0]*cols + line[1]*rows + line[2]) <= 2.2*np.linalg.norm(line[:2])
    near &= (rows > y-12) & (rows < v+12)
    points = np.column_stack((cols[near], rows[near]))
    if len(points) >= 24:
        dx, dy, px, py = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
        line = np.cross([px, py, 1.], [px+dx, py+dy, 1.])
    return line / np.linalg.norm(line[:2])


def beam_tip(image):
    """Find the start of the continuous cyan streak on the fitted beam."""
    try:
        a, b, c = _beam_line(image)
    except ValueError:
        return None
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (82, 45, 125), (106, 255, 255))
    rows, columns = np.nonzero(mask)
    on_line = np.abs(a*columns+b*rows+c) < 2.2
    supported = np.unique(rows[on_line])
    runs = np.split(supported, np.flatnonzero(np.diff(supported) > 3)+1)
    streak = max(runs, key=len)
    if len(streak) < 45 or abs(a) < .1:
        return None
    y = int(streak[0])
    x = -(b*y+c)/a
    return float(x/(image.width-1)), float(y/(image.height-1))


def _tracking_transform(pair, commanded_head):
    _, _, left, spacing = stereo_context(pair)
    eye_axes = left[:3, :3]
    head_center = left[:3, 3] + eye_axes[:, 0] * spacing / 2
    head_rotation = Rotation.from_quat(commanded_head[3:], scalar_first=True).as_matrix()
    tracking_basis = eye_axes @ np.diag([1., -1., -1.]) @ head_rotation.T
    scale = spacing / BASELINE
    origin = head_center - tracking_basis @ np.asarray(commanded_head[:3]) * scale
    return left, spacing, tracking_basis, origin


def beam_planes(pair, commanded_poses):
    """Convert the two image lines into planes in right-controller coordinates."""
    if commanded_poses is None:
        raise ValueError("No controller pose associated with the image")
    left, spacing, tracking_basis, tracking_origin = _tracking_transform(pair, commanded_poses[0])
    controller = commanded_poses[2]
    basis = tracking_basis @ Rotation.from_quat(controller[3:], scalar_first=True).as_matrix()
    scale = spacing / BASELINE
    origin = tracking_origin + tracking_basis @ np.asarray(controller[:3]) * scale
    planes = []
    for i in range(2):
        image = pair.crop((640*i, 0, 640*(i+1), 720))
        normal = left[:3, :3] @ (K.T @ _beam_line(image))
        normal /= np.linalg.norm(normal)
        eye = left[:3, 3] + left[:3, 0] * i * spacing
        planes.append((normal @ basis, float(normal @ (eye-origin) / scale)))
    return planes, spacing


def fit_beam(samples):
    """Fit a ray to multiple stereo observations and discard inconsistent samples."""
    if len(samples) < 4:
        raise ValueError("At least four different visible beam poses are required")
    normals = np.asarray([[n for n, _ in planes] for planes, _ in samples])
    offsets = np.asarray([[d for _, d in planes] for planes, _ in samples])
    spacing = float(np.median([spacing for _, spacing in samples]))
    best = None
    for i, j in combinations(range(len(samples)), 2):
        indices = [i, j]
        _, singular, directions = np.linalg.svd(normals[indices].reshape(-1, 3), full_matrices=False)
        if singular[1] < .06:
            continue
        direction = directions[-1]
        anchor = np.linalg.lstsq(np.vstack((normals[indices].reshape(-1, 3), direction)),
                                 np.r_[offsets[indices].ravel(), 0.], rcond=None)[0]
        angular = np.max(np.abs(normals @ direction), axis=1)
        spatial = np.max(np.abs(normals @ anchor - offsets), axis=1)
        keep = (angular < .032) & (spatial < .015)
        score = int(keep.sum()), -float(np.sum((angular + spatial)[keep]))
        if best is None or score > best[0]:
            best = score, keep
    if best is None or best[0][0] < max(4, len(samples)*.6):
        raise ValueError("Beam observations did not agree on a stable controller ray")
    keep = best[1]
    n, d = normals[keep].reshape(-1, 3), offsets[keep].ravel()
    _, singular, directions = np.linalg.svd(n, full_matrices=False)
    if singular[1] < .12:
        raise ValueError("Sample a wider range of controller angles")
    direction = directions[-1]
    if direction[2] > 0:
        direction = -direction
    anchor = np.linalg.lstsq(np.vstack((n, direction)), np.r_[d, 0.], rcond=None)[0]
    residual = float(np.sqrt(np.mean((n @ anchor-d)**2)))
    if np.linalg.norm(anchor) > .25 or direction[2] > -.75 or residual > .012:
        raise ValueError("Fitted beam is inconsistent with a forward controller ray")
    return {"direction": direction.tolist(), "offset_m": anchor.tolist(), "eye_spacing_m": spacing,
            "samples_used": int(keep.sum()), "samples_total": len(samples), "plane_residual_m": residual}


class BeamRangeError(ValueError):
    pass


def validate_beam_calibration(calibration):
    """Validate a saved calibration before using it to command a pointer."""
    if not isinstance(calibration, dict):
        raise ValueError("Expected a beam calibration object")
    direction = np.asarray(calibration.get("direction"), dtype=float)
    offset = np.asarray(calibration.get("offset_m"), dtype=float)
    spacing = calibration.get("eye_spacing_m")
    if (direction.shape != (3,) or offset.shape != (3,) or not np.isfinite(direction).all()
            or not np.isfinite(offset).all() or abs(np.linalg.norm(direction) - 1) > .01
            or direction[2] > -.75 or np.linalg.norm(offset) > .25
            or not isinstance(spacing, (int, float)) or not .004 < spacing < .15):
        raise ValueError("Invalid fitted controller ray or rendered eye spacing")
    return calibration


def solve_aim(pair, commanded_head, target_world, calibration, side="right"):
    """Find the existing two-axis pointer command that intersects a world target."""
    _, spacing, basis, origin = _tracking_transform(pair, commanded_head)
    if abs(spacing - calibration["eye_spacing_m"]) > .002:
        raise ValueError("Rendered eye spacing changed. Recalibrate for this avatar.")
    scale = spacing / BASELINE
    target = basis.T @ (np.asarray(target_world)-origin) / scale
    offset = np.asarray(calibration["offset_m"])
    direction = np.asarray(calibration["direction"])
    if side == "left":
        reflection = np.array([-1, 1, 1])
        offset = offset * reflection
        direction = direction * reflection

    def residual(xy):
        pose = TargetSolver.pointer_pose(commanded_head, *xy, side)
        rotation = Rotation.from_quat(pose[3:], scalar_first=True)
        vector = target - pose[:3] - rotation.apply(offset)
        return np.cross(rotation.apply(direction), vector)

    result = least_squares(residual, [.5, .18], bounds=([0., 0.], [1., 1.]), max_nfev=36)
    pose = TargetSolver.pointer_pose(commanded_head, *result.x, side)
    rotation = Rotation.from_quat(pose[3:], scalar_first=True)
    hit_is_forward = rotation.apply(direction) @ (target-pose[:3]-rotation.apply(offset)) > 0
    if not result.success or np.linalg.norm(result.fun) > .014 or not hit_is_forward:
        raise BeamRangeError("World target is outside the calibrated aiming range")
    return result.x.tolist()
