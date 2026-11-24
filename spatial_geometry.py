"""Measure visible surfaces and palm positions in VRChat world coordinates."""

import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_pose import read_eye
from world_map import BASELINE, K

WORLD_UP = np.array([0., 1., 0.])
FLIP_Y = np.diag([1., -1., 1., 1.])


def stereo_context(pair):
    if pair.size != (1280, 720):
        raise ValueError("A calibrated stereo pair is required")
    left, right = np.hsplit(np.asarray(pair.convert("RGB")), 2)
    huds = [read_eye(eye) for eye in (left, right)]
    if any(hud is None for hud in huds):
        raise ValueError("Enable Tracking HUD and capture both unobstructed eyes")
    first, second = huds
    relative = np.linalg.solve(first["pose"], second["pose"])
    baseline = float(relative[0, 3])
    if (not .004 < baseline < .15 or np.linalg.norm(relative[1:3, 3]) > .001
            or not np.allclose(relative[:3, :3], np.eye(3), atol=.001)):
        raise ValueError("The two rendered eye poses do not form a rectified stereo pair")
    images = [cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY) for eye in (left, right)]
    # The camera axes are right/down/forward. Return coordinates in VRChat's Y-up world.
    return images, first, FLIP_Y @ first["pose"], baseline


def _overlays(flipped, width, height):
    top = .78 if flipped else .06
    bottom = .94 if flipped else .22
    palm_top = .86 if flipped else .06
    palm_bottom = .94 if flipped else .14
    return [(width * x1, height * y1, width * x2, height * y2) for x1, y1, x2, y2 in
            ((.04, top, .24, bottom), (.25, palm_top, .66, palm_bottom))]


def _peaks(reference, other, x, y, side, maximum):
    radius = 5
    width = reference.shape[1]
    start = max(radius, x - maximum) if side < 0 else x + 2
    stop = x - 2 if side < 0 else min(width - radius - 1, x + maximum)
    if stop - start < 3 or not radius <= x < width - radius:
        return None
    patch = reference[y-radius:y+radius+1, x-radius:x+radius+1]
    if patch.std() < 6:
        return None
    stripe = other[y-radius:y+radius+1, start-radius:stop+radius+1]
    scores = cv2.matchTemplate(stripe, patch, cv2.TM_CCOEFF_NORMED)[0]
    index = int(scores.argmax())
    best = float(scores[index])
    alternatives = np.concatenate((scores[:max(index-2, 0)], scores[index+3:]))
    if best < .78 or (alternatives.size and best - alternatives.max() < .045):
        return None
    shift = 0.
    if 0 < index < len(scores) - 1:
        before, after = scores[index-1], scores[index+1]
        curvature = before - 2 * best + after
        if curvature < -1e-5:
            shift = float(np.clip(.5 * (before-after) / curvature, -.5, .5))
    return start + index + shift


def match_stereo_pixel(left, right, x, y, maximum):
    """Find a unique, reciprocal horizontal match for one left-eye pixel."""
    if not 5 <= y < left.shape[0]-5:
        return None
    match = _peaks(left, right, x, y, -1, maximum)
    if match is not None:
        reverse = _peaks(right, left, round(match), y, 1, maximum)
        if reverse is not None and abs(reverse-x) <= 1.25:
            return match
    return None


def measure_surface(pair, x, y, radius_px=14):
    """Fit a local surface near a normalized left-eye image point."""
    return measure_surface_in_context(stereo_context(pair), x, y, radius_px)


def measure_surface_in_context(context, x, y, radius_px=14):
    """Fit a point using an already validated, decoded stereo pair."""
    if not (0 <= x < 1 and 0 <= y < 1 and 6 <= radius_px <= 32):
        raise ValueError("Use normalized left-eye coordinates and a radius between 6 and 32 pixels")
    images, hud, pose, baseline = context
    left, right = images
    height, width = left.shape
    cx, cy = round(x * (width - 1)), round(y * (height - 1))
    overlays = _overlays(hud["flipped_patch"], width, height)
    maximum = min(100, math.ceil(K[0, 0] * baseline / .2))
    points = []
    for py in range(max(5, cy-radius_px), min(height-5, cy+radius_px+1), 3):
        for px in range(max(5, cx-radius_px), min(width-5, cx+radius_px+1), 3):
            if any(y1-5 <= py <= y2+5 and x1-5 <= px <= x2+maximum+5
                   for x1, y1, x2, y2 in overlays):
                continue
            match = match_stereo_pixel(left, right, px, py, maximum)
            if match is None:
                continue
            depth = K[0, 0] * baseline / (px - match)
            points.append([px, py, depth])
    if len(points) < 8:
        raise ValueError("Insufficient unambiguous stereo matches near the point")
    samples = np.asarray(points)
    offsets = np.linalg.norm(samples[:, :2] - [cx, cy], axis=1)
    nearby = samples[np.argsort(offsets)[:min(len(samples), 12)]]
    if offsets.min() > radius_px * .6:
        raise ValueError("The selected point has too few stereo matches nearby")
    center_depth = np.median(nearby[:, 2])
    surface = samples[np.abs(samples[:, 2]-center_depth) <= max(.035, center_depth * .10)]
    if len(surface) < max(8, len(samples) * .6):
        raise ValueError("Multiple depths cross the selected region")
    rays = np.column_stack((surface[:, :2], np.ones(len(surface)))) @ np.linalg.inv(K).T
    cloud = (rays * surface[:, 2, None]) @ pose[:3, :3].T + pose[:3, 3]
    centroid = cloud.mean(axis=0)
    _, singular, axes = np.linalg.svd(cloud - centroid, full_matrices=False)
    normal = axes[-1]
    ray = pose[:3, :3] @ np.linalg.solve(K, [cx, cy, 1])
    if np.dot(normal, ray) > 0:
        normal = -normal
    alignment = float(np.dot(normal, ray))
    plane_error = float(np.sqrt(np.mean(((cloud - centroid) @ normal) ** 2)))
    depth = float(center_depth)
    reliable_plane = singular[1] > .012 and abs(alignment) > .12 and plane_error <= .025
    if reliable_plane:
        intersect = float(np.dot(normal, centroid - pose[:3, 3]) / alignment)
        if abs(intersect - depth) <= max(.035, depth * .08):
            depth = intersect
        else:
            reliable_plane = False
    return {
        "world_xyz_m": (pose[:3, 3] + ray * depth).tolist(),
        "surface_normal": normal.tolist() if reliable_plane else None,
        "depth_m": depth,
        "matched_samples": len(surface),
        "depth_spread_m": float(np.median(np.abs(surface[:, 2]-center_depth))),
        "plane_residual_m": plane_error if reliable_plane else None,
        "eye_spacing_m": baseline,
    }


def palm_step(pair, commanded_poses, side, goal):
    """Convert a measured palm error to one bounded controller correction."""
    _, hud, pose, baseline = stereo_context(pair)
    if side not in hud["palms"] or commanded_poses is None:
        raise ValueError(f"No measured {side} palm and matching controller pose")
    palm = hud["palms"][side][0]
    error = np.asarray(goal) - palm
    distance = float(np.linalg.norm(error))
    forward = pose[:3, 2].copy()
    forward[1] = 0
    if np.linalg.norm(forward) < .25:
        raise ValueError("Look closer to level to align the world and controller headings")
    forward /= np.linalg.norm(forward)
    right = np.cross(WORLD_UP, forward)
    local_error = np.array([right @ error, error[1], -forward @ error]) * (BASELINE / baseline)
    size = np.linalg.norm(local_error)
    if size > .16:
        local_error *= .16 / size

    head = Rotation.from_quat(commanded_poses[0][3:], scalar_first=True)
    head_forward = head.apply([0., 0., -1.])
    yaw = math.atan2(head_forward[0], -head_forward[2])
    controller = commanded_poses[1 if side == "left" else 2]
    heading = Rotation.from_rotvec([0., yaw, 0.])
    local_wrist = heading.apply(controller[:3]) + local_error
    relative = heading * Rotation.from_quat(controller[3:], scalar_first=True)
    wrist_yaw, pitch, roll = relative.as_euler("YXZ", degrees=True)
    return distance, palm.tolist(), [*local_wrist.tolist(), -wrist_yaw, pitch, roll]
