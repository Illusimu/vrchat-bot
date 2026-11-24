"""Resolve desktop input and optional gestures into a SteamVR device frame."""

import math
import struct
from motion import Locomotion, STAND
from exercises import exercise
from task_motion import TaskPose
from tracking_rig import NAMES as DEVICE_NAMES, procedural_trackers

PORT = 29745
FRAME = struct.Struct("<4s99f")
GESTURE_LENGTH = {"wave": 1.6, "nod": 0.65, "backflip": 1.75, "jumping-jacks": 3.2, "push-ups": 5.6, "leg-lift": 2.4}
SET_SIZE = {"jumping-jacks": 2, "push-ups": 3}


def gesture_count(name, repetitions=None):
    if name not in GESTURE_LENGTH:
        raise ValueError(f"Unknown gesture {name!r}")
    count = SET_SIZE.get(name, 1) if repetitions is None else repetitions
    if type(count) is not int or not 1 <= count <= 5:
        raise ValueError("Gesture repetitions must be a whole number from 1 through 5")
    return count


def gesture_duration(name, repetitions):
    if name == "push-ups":
        return GESTURE_LENGTH[name] * (.48 + .52 * repetitions / SET_SIZE[name])
    return GESTURE_LENGTH[name] * repetitions / SET_SIZE.get(name, 1)


def rotation(yaw, pitch=0, roll=0):
    """OpenVR quaternion (w, x, y, z), with positive yaw turning right."""
    y, p, r = (math.radians(angle) / 2 for angle in (-yaw, pitch, roll))
    cy, cp, cr = map(math.cos, (y, p, r))
    sy, sp, sr = map(math.sin, (y, p, r))
    return (
        cy * cp * cr + sy * sp * sr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
        cy * cp * sr - sy * sp * cr,
    )


class Body:
    def __init__(self):
        self.yaw = self.pitch = self.crouch = 0.0
        self.gestures = {}
        self.motion = Locomotion()
        self.rest = TaskPose()

    def request_gesture(self, name, now, repetitions=None):
        """Request a finite animation. Call from the same loop as update."""
        count = gesture_count(name, repetitions)
        duration = gesture_duration(name, count)
        self.gestures[name] = (now, duration, count)
        return duration

    def release(self):
        self.gestures.clear()
        self.crouch = 0.0

    def update(self, keys, mouse, now, dt, *, steady=False):
        dt = min(dt, 0.05)
        strafe = int("d" in keys) - int("a" in keys)
        forward = int("w" in keys) - int("s" in keys)
        speed = math.hypot(strafe, forward)
        if speed > 1:
            strafe, forward = strafe / speed, forward / speed
        if not steady and (speed or "t" in keys):
            self.rest.change("standing")
        if speed and not self.rest.settled:
            strafe = forward = 0.
        if steady:
            strafe = forward = 0.
            movement = STAND
        else:
            movement = self.motion.update(strafe, forward, dt)
            self.crouch += (("c" in keys) * 0.42 - self.crouch) * min(dt * 12, 1)

        look_x = 110 * dt * (int("right" in keys) - int("left" in keys)) + mouse[0] * 0.22
        look_y = 85 * dt * (int("up" in keys) - int("down" in keys)) - mouse[1] * 0.22
        self.yaw = (self.yaw + look_x + 180) % 360 - 180
        self.pitch = max(-80, min(80, self.pitch + look_y))
        if look_x or look_y:
            self.gestures.pop("nod", None)

        calibrating = "t" in keys
        left = self.hand_controls(keys, "q", "z", "67890", "g", "v", strafe, forward)
        right = self.hand_controls(keys, "e", "x", "12345", "space", "b")
        if calibrating:
            movement = STAND
            self.gestures.clear()
            self.crouch = 0.0
            # Keep the triggers available to confirm VRChat full-body calibration.
            left = (0.0, 0.0, left[2], *(0.0,) * 8)
            right = (0.0, 0.0, right[2], *(0.0,) * 8)
        if any(right):
            self.gestures.pop("wave", None)

        nod = 0. if steady else self.progress("nod", now)
        wave = 0. if steady else self.progress("wave", now)
        head_pitch = 0.0 if calibrating else self.pitch - 16 * math.sin(math.pi * nod)
        wave_height = math.sin(math.pi * wave)
        wave_roll = wave_height * math.sin(6 * math.pi * wave) * 24
        bob = movement.rise
        hand_y = 1.15 - self.crouch * 0.55 + bob + movement.hand_rise
        hand_x = 0.32
        hand_z = -0.30
        if calibrating:
            hand_x, hand_y, hand_z = 0.71, 1.38, 0.0

        # All six targets share one standing origin and one body heading.
        # OpenVR uses metres, Y up, and negative Z forward.
        theta = math.radians(self.yaw)
        sine, cosine = math.sin(theta), math.cos(theta)
        def pose(x, y, z, pitch=0, roll=0):
            position = (cosine * x - sine * z, y, sine * x + cosine * z)
            return position + rotation(self.yaw, pitch, roll)

        if not steady:
            for name in ("backflip","jumping-jacks","push-ups","leg-lift"):
                phase=self.progress(name,now)
                if phase:
                    count = self.gestures[name][2]
                    return procedural_trackers(tuple(pose(*p) for p in exercise(name, phase, count))), left, right
        dx, dz = self.motion.direction
        swing_x = dx * movement.stride
        swing_z = -dz * movement.stride
        endpoints = (
            (0, 1.65-self.crouch+bob, 0, head_pitch, 0),
            (-hand_x, hand_y, hand_z-swing_z*.35, 0, 0),
            (hand_x+wave_height*.16, hand_y+wave_height*.33, hand_z+swing_z*.35, 0, wave_roll),
            (movement.sway,.95-self.crouch*.65+bob,self.crouch*.2,0,0),
            (-.13+swing_x,.06+movement.left_lift,swing_z,0,0),
            (.13-swing_x,.06+movement.right_lift,-swing_z,0,0),
        )
        offsets = self.rest.sample(0. if steady else dt)
        poses = tuple(pose(*(a+b for a,b in zip(endpoint, offset))) for endpoint,offset in zip(endpoints,offsets))
        return procedural_trackers(poses), left, right

    def progress(self, name, now):
        if name in self.gestures:
            started, duration, count = self.gestures[name]
            progress = (now - started) / duration
            if 0 <= progress < 1:
                return progress if name in SET_SIZE else (progress * count) % 1
            if progress >= 1:
                del self.gestures[name]
        return 0.0

    @staticmethod
    def hand_controls(keys, grip_key, trigger_key, fingers, button_a, button_b, x=0, y=0):
        grip = grip_key in keys
        trigger = trigger_key in keys
        curls = tuple(
            key in keys or grip or (index == 1 and trigger)
            for index, key in enumerate(fingers)
        )
        return x, y, trigger, grip, button_a in keys, button_b in keys, *curls


def pack_frame(poses, left, right):
    return FRAME.pack(b"DB02", *(value for pose in poses for value in pose), *left, *right)
