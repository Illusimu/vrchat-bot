"""Generate local SOMA clips and retarget the eleven SteamVR devices."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "build/motions/generated"
FPS = 30
# SOMA77 joint order, excluding the BVH wrapper root. Distal upper-arm
# and thigh mounts take the rotation of the segment carrying the tracker.
TRACKERS = [6, 14, 42, 0, 69, 74, 3, 12, 40, 67, 72]
MOUNTS = [(7, 13, .82), (8, 41, .82), (9, 68, .95), (10, 73, .95)]


def constraints(rows, frames):
    # SOMA30 in its rest frame faces +Z with anatomical left at +X.
    neutral = np.zeros((30, 3))
    neutral[11, 2], neutral[17, 2] = -1.15, 1.15
    poses = {0: neutral.copy(), frames - 1: neutral.copy()}
    roots = {}
    if not rows or len(rows) > 36:
        raise ValueError("A new motion needs 1 to 36 joint keyframes")
    for row in rows:
        if len(row) != 5 or not np.isfinite(row).all():
            raise ValueError("Invalid joint keyframe")
        phase, joint, *angles = row
        if not 0 <= phase <= 1 or joint != int(joint) or not 0 <= joint <= 30 or max(map(abs,angles)) > 180:
            raise ValueError("Joint keyframe outside the SOMA range")
        index = round(phase * (frames - 1))
        pose = poses.setdefault(index, neutral.copy())
        if joint == 30:
            if not (abs(angles[0]) <= 2 and .15 <= angles[1] <= 2.5 and abs(angles[2]) <= 2):
                raise ValueError("Hip position exceeds the clip envelope")
            roots[index] = list(map(float,angles))
        else:
            pose[int(joint)] = np.radians(angles)
    indices = sorted(poses)
    return [dict(type="fullbody", frame_indices=indices,
                 local_joints_rot=[poses[i].tolist() for i in indices],
                 root_positions=[roots.get(i,[0.,1.,0.]) for i in indices])]


class MotionJob:
    def __init__(self, name, seconds, rows):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) or not 1 <= seconds <= 8:
            raise ValueError("Invalid motion name or duration")
        LIBRARY.mkdir(parents=True, exist_ok=True)
        self.path = LIBRARY / (name + ".npz")
        self.process = self.temp = None
        if self.path.exists():
            return
        data = constraints(rows, round(FPS * seconds))
        self.temp = tempfile.TemporaryDirectory(dir=LIBRARY)
        directory = Path(self.temp.name)
        spec = directory / "keyframes.json"
        spec.write_text(json.dumps(data))
        try:
            with (directory / "worker.log").open("w") as log:
                self.process = subprocess.Popen(
                    [ROOT / "build/kimodo-env/Scripts/python.exe", ROOT / "kimodo_motion.py",
                     "--duration", str(seconds), "--constraints", spec, "--output", directory / "clip.npz"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, HF_HUB_OFFLINE="1"), creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as error:
            self.close()
            raise ValueError(f"Cannot launch local Kimodo worker: {error}") from error

    def poll(self):
        if self.process is None:
            return self.path
        code = self.process.poll()
        if code is None:
            return None
        directory = Path(self.temp.name)
        if code:
            detail = (directory / "worker.log").read_text(errors="replace").strip().splitlines()
            raise ValueError("Kimodo failed: " + (detail[-1] if detail else str(code)))
        os.replace(directory / "clip.npz", self.path)
        self.close()
        return self.path

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.process = None
        if self.temp:
            self.temp.cleanup()
            self.temp = None


def mix(a, b, t):
    result = a + (b - a) * t
    qa, qb = a[:, 3:], b[:, 3:]
    q = qa * (1-t) + qb * np.where((qa*qb).sum(axis=1, keepdims=True)<0,-1,1) * t
    result[:,3:] = q / np.linalg.norm(q,axis=1,keepdims=True)
    return result


class Playback:
    def __init__(self, path, initial, yaw):
        with np.load(path, allow_pickle=False) as clip:
            joints = clip["posed_joints"]
            if joints.ndim != 3 or joints.shape[1:] != (77, 3):
                raise ValueError("Expected a SOMA77 joint sequence")
            xyz = joints[:, TRACKERS, :].copy()
            for device, distal, fraction in MOUNTS:
                xyz[:,device] += fraction * (joints[:,distal] - xyz[:,device])
            across = joints[0,67] - joints[0,72]
            mats = clip["global_rot_mats"][:, TRACKERS, :, :]
        if len(xyz)<2 or not np.isfinite(xyz).all() or not np.isfinite(mats).all():
            raise ValueError("Unusable SOMA motion")
        initial = np.asarray(initial)
        # Rotate SOMA left (+X) onto OpenVR left (-X), then apply pilot heading.
        # Hip lateral axis is preferable to crossed feet for the initial heading.
        basis = Rotation.from_euler("y", math.pi - math.radians(yaw) + math.atan2(across[2], across[0])).as_matrix()
        # Keep the generator's floor, including when the first pose is prone.
        origin = xyz[0,3].copy()
        origin[1] = 0
        anchor = initial[3,:3].copy()
        anchor[1] = 0
        positions = (xyz - origin) @ basis.T + anchor
        # SOMA global rotations deform the standard T pose. The fixed mount
        # basis is a half turn about Y (SOMA faces +Z, OpenVR faces -Z).
        mount = Rotation.from_euler("y", math.pi).as_matrix()
        orientations = basis @ mats @ mount.T
        quat = Rotation.from_matrix(orientations.reshape(-1,3,3)).as_quat().reshape(len(xyz),len(TRACKERS),4)
        self.frames = np.concatenate((positions, quat[:,:,[3,0,1,2]]), axis=2)
        self.duration = (len(xyz)-1)/FPS
        self.blend = .25

    def sample(self, seconds, base):
        index = np.clip((seconds-self.blend)*FPS,0,len(self.frames)-1)
        i = int(index)
        motion = mix(self.frames[i], self.frames[min(i+1,len(self.frames)-1)],index-i)
        weight = np.clip(min(seconds, self.duration+2*self.blend-seconds)/self.blend,0,1)
        return mix(np.asarray(base), motion, weight*weight*(3-2*weight))
