"""Short tracking-space clips connected by matching foot-contact poses."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    sway: float = 0.0
    rise: float = 0.0
    stride: float = 0.0
    left_lift: float = 0.0
    right_lift: float = 0.0
    hand_rise: float = 0.0


STAND = Pose()
LEFT = Pose(-.025, .008, -.10)
RIGHT = Pose(.025, .008, .10)
POSES = {"standing": STAND, "left contact": LEFT, "right contact": RIGHT}


@dataclass(frozen=True)
class Clip:
    name: str
    start: str
    end: str
    seconds: float
    middle: Pose

    def sample(self, fraction):
        first, last = POSES[self.start], POSES[self.end]
        if fraction <= .5:
            a, b, t = first, self.middle, fraction * 2
        else:
            a, b, t = self.middle, last, fraction * 2 - 1
        weight = t * t * (3 - 2 * t)
        return Pose(**{k: getattr(a, k) + (getattr(b, k) - getattr(a, k)) * weight
                       for k in Pose.__dataclass_fields__})


CLIPS = (
    Clip("Breathe", "standing", "standing", 2.8, Pose(rise=.008, hand_rise=.01)),
    Clip("Shift weight", "standing", "standing", 2.4, Pose(sway=.018, hand_rise=-.008)),
    Clip("Step off", "standing", "right contact", .24, Pose(left_lift=.035)),
    Clip("Right step", "right contact", "left contact", .34, Pose(right_lift=.065, rise=.02)),
    Clip("Left step", "left contact", "right contact", .34, Pose(left_lift=.065, rise=.02)),
    Clip("Settle left", "left contact", "standing", .23, Pose(right_lift=.024)),
    Clip("Settle right", "right contact", "standing", .23, Pose(left_lift=.024)),
)


def path(start, end):
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        node, route = queue.popleft()
        if node == end:
            return route
        for clip in CLIPS:
            if clip.start == node and clip.end not in seen:
                seen.add(clip.end)
                queue.append((clip.end, route + [clip]))
    raise ValueError(f"No motion path from {start} to {end}")


class Locomotion:
    def __init__(self):
        self.clip = CLIPS[0]
        self.elapsed = 0.0
        self.idle_index = 0
        self.direction = (0., 0.)
        self.entry_pose = None

    def update(self, x, z, dt):
        moving = bool(x or z)
        # Blend direction even when reversing midway through a step.
        if moving:
            self.direction = tuple(a + (b - a) * min(1, dt * 12)
                                   for a, b in zip(self.direction, (x, z)))
        if moving and self.clip.start == self.clip.end:
            # Preserve the sampled idle pose when interrupting a long idle loop.
            self.entry_pose = self.clip.sample(self.elapsed / self.clip.seconds)
            self.clip = path("standing", "right contact")[0]
            self.elapsed = 0.
        self.elapsed += dt
        while self.elapsed >= self.clip.seconds:
            self.elapsed -= self.clip.seconds
            node = self.clip.end
            self.entry_pose = None
            if moving:
                goal = "left contact" if node == "right contact" else "right contact"
                self.clip = path(node, goal)[0]
            elif node != "standing":
                self.clip = path(node, "standing")[0]
            else:
                self.idle_index = 1 - self.idle_index
                self.clip = CLIPS[self.idle_index]
        t = self.elapsed / self.clip.seconds
        pose = self.clip.sample(t)
        entry = self.entry_pose
        if entry:
            weight = 1 - t * t * (3 - 2 * t)
            pose = Pose(**{k: getattr(pose, k) + weight * getattr(entry, k)
                           for k in Pose.__dataclass_fields__})
        return pose
