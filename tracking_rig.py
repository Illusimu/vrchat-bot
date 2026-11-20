"""Expand endpoint-only procedural poses to the eleven virtual devices."""
import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ("Head", "Left hand", "Right hand", "Waist", "Left foot", "Right foot",
         "Chest", "Left upper arm", "Right upper arm", "Left knee", "Right knee")


def unit(vector, fallback):
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-7 else np.asarray(fallback)


def limb_mount(start, end, half_length, pole, neutral_axis, fraction, torso):
    # Solve an equal-length two-link chain in torso space. The pole selects
    # the anatomical bend direction even when the whole body is inverted.
    delta = torso.inv().apply(end - start)
    axis = unit(delta, neutral_axis)
    bend = pole - np.dot(pole, axis) * axis
    if np.linalg.norm(bend) < 1e-6:
        bend = np.cross(axis, [1, 0, 0])
    midpoint = delta / 2 + unit(bend, [0, 0, 1]) * np.sqrt(max(0, half_length**2 - delta.dot(delta)/4))
    direction = unit(midpoint, neutral_axis)
    cross = np.cross(neutral_axis, direction)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(neutral_axis, direction))
    orientation = torso * Rotation.from_rotvec(unit(cross, [0, 0, 1]) * angle)
    return (* (start + torso.apply(fraction * midpoint)), *orientation.as_quat(scalar_first=True))


def procedural_trackers(poses):
    points = np.asarray(poses)[:, :3]
    torso = Rotation.from_quat(poses[3][3:], scalar_first=True)
    chest = points[3] + .61 * (points[0] - points[3])
    extra = [(*chest, *poses[3][3:])]
    for side, hand in ((-1, 1), (1, 2)):
        shoulder = chest + torso.apply([.17*side, .015, 0])
        extra.append(limb_mount(shoulder, points[hand], .28, np.array([side*.5, -.4, .8]),
                                [side, 0, 0], .82, torso))
    for side, foot in ((-1, 4), (1, 5)):
        hip = points[3] + torso.apply([side*.13, 0, 0])
        extra.append(limb_mount(hip, points[foot], .445, np.array([0, 0, -1]),
                                [0, -1, 0], .95, torso))
    return (*poses[:6], *extra)
