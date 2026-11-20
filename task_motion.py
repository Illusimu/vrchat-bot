"""Body-relative task poses in metres, with timed contact and release events."""
import math
import numpy as np
from exercises import smooth, symmetric

STANDING = "standing"
RESTS = ("standing", "sitting", "kneeling", "lying")


def rest_offsets(name, height=.45):
    # Additive xyz and pitch on the six endpoints. Feet keep the floor.
    zero = (0., 0., 0., 0., 0.)
    if name == "sitting":
        drop = height - .95
        return symmetric((0,drop,.16,0,0), (0,drop+.04,.03,0,0),
                         (0,drop,.20,0,0), (0,0,-.25,0,0))
    if name == "kneeling":
        return symmetric((0,-.59,-.21,-22,0), (0,-.55,-.02,-20,0),
                         (0,-.54,0,-16,0), (0,.02,.35,-68,0))
    if name == "lying":
        return symmetric((0,-1.42,-.64,-82,0), (0,-1,-.36,-65,0),
                         (0,-.74,0,-82,0), (0,.05,.84,-70,0))
    return [zero]*6


class TaskPose:
    def __init__(self):
        self.name = STANDING
        self.height = .45
        self.current = np.zeros((6,5))
        self.start = self.current.copy()
        self.end = self.current.copy()
        self.elapsed = self.duration = self.idle_time = 0.

    def change(self, name, height=.45):
        if name not in RESTS or not .20 <= height <= .85:
            raise ValueError("Invalid rest pose or seat height")
        endpoint = np.array(rest_offsets(name,height))
        if not np.array_equal(endpoint,self.end):
            self.start = self.current.copy()
            self.end = endpoint
            self.elapsed = 0.
            self.duration = max(.45, float(np.max(abs(endpoint[:,:3]-self.start[:,:3]))) * 1.25)
        self.name, self.height = name, height

    def sample(self, dt):
        self.elapsed = min(self.duration,self.elapsed+dt)
        t = smooth(self.elapsed/self.duration) if self.duration else 1.
        self.current = self.start+(self.end-self.start)*t
        if self.settled and self.name != STANDING:
            self.idle_time += dt
            slow = math.sin(self.idle_time*.37)
            breath = math.sin(self.idle_time*1.5)
            idle = self.current.copy()
            # Leave the support points planted as the head, torso and hands rest.
            if self.name == "lying":
                idle[0,3] += slow*2
                idle[3,1] += breath*.003
            else:
                idle[0,0] += slow*.012
                idle[0,1] += breath*.005
                idle[1:3,1] += breath*.004
            return idle
        self.idle_time = 0.
        return self.current

    @property
    def settled(self):
        return self.elapsed >= self.duration


def through(points, times, phase):
    """Ease into each waypoint; the throw release is mid-segment."""
    for i, end in enumerate(times[1:]):
        if phase <= end:
            f = smooth((phase-times[i])/(end-times[i]))
            return points[i]+(points[i+1]-points[i])*f
    return points[-1]


class HandAction:
    def __init__(self, kind, side, target, current, heading):
        self.kind = kind
        self.sign = -1 if side == "left" else 1
        self.index = 1 if side == "left" else 2
        self.grip = "q" if side == "left" else "e"
        self.duration = {"pickup":2.4,"place":1.9,"throw":1.15}[kind]
        self.target = np.array(target, dtype=float)
        yaw = math.radians(heading)
        self.basis = np.array([[math.cos(yaw),0,-math.sin(yaw)],[0,1,0],[math.sin(yaw),0,math.cos(yaw)]])
        self.start = self.basis.T @ current[:3]
        self.carry = np.array([self.sign*.32,1.15,-.30])
        x,y,z = self.target
        if abs(x-self.sign*.19)>.65 or not .035<=y<=1.95 or not -.76<=z<=.25:
            raise ValueError("Hand target outside reach; approach and observe again")

    def step(self, t, body, solver):
        target, carry = self.target,self.carry
        if self.kind == "throw":
            wind = np.array([self.sign*.36,1.52,.04])
            points = [self.start,wind,target,carry]
            times = [0,.43,.77,1]
            if t >= .62:
                solver.grips.discard(self.grip)
        else:
            # Lower the pelvis before final contact. Resume standing while lifting.
            low = target[1]<.77
            if low:
                body.rest.change("kneeling" if t<.68 else "standing")
            near = target+np.array([0,.09,.045])
            points = [self.start,near,target,target,carry]
            times = [0,.38,.54,.65,1]
            if t>=.55:
                if self.kind=="pickup":
                    solver.grips.add(self.grip)
                else:
                    solver.grips.discard(self.grip)
        xyz = self.basis @ through(points,times,t)
        solver.hands[self.index] = (*xyz,*self.orientation)
