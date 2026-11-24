"""Execute one planner target and report its measured outcome."""
import math
from concurrent.futures import ThreadPoolExecutor
from menu_selection import Selection

import cv2
import numpy as np

from body import rotation
from tracking_rig import procedural_trackers, shoulder_position
from task_motion import HandAction
from generated_motion import MotionJob, Playback
from scipy.spatial.transform import Rotation


def quaternion_mix(start, end, t):
    end = np.asarray(end)
    if np.dot(start, end) < 0:
        end = -end
    q = (1 - t) * np.asarray(start) + t * end
    return tuple(q / np.linalg.norm(q))


class TargetSolver:
    walk_keys = {"forward": "w", "backward": "s", "left": "a", "right": "d"}

    def __init__(self):
        self.job = self.dolly = self.selection = None
        self.reader = ThreadPoolExecutor(max_workers=1)
        self.reset()

    def reset(self):
        if self.selection and self.selection.pending:
            self.selection.pending.cancel()
        self.selection = None
        if self.job:
            self.job.close()
        if self.dolly:
            self.dolly.stop()
        self.job = self.playback = None
        self.clip_time = 0
        self.target = None
        self.carrying = set()
        self.hands = {}
        self.grips = set()
        self.keys = set()
        self.pulse_end = 0
        self.trigger = None
        self.menu_aimed = False

    def set_grip(self, side, closed):
        """Change the simulated controller input, independent of object contact."""
        index, key = (1, "q") if side == "left" else (2, "e")
        if closed:
            self.grips.add(key)
            self.carrying.add(index)
        else:
            self.grips.discard(key)
            self.carrying.discard(index)
        self.menu_aimed = False

    @staticmethod
    def wrist_motion(values, index, poses, heading, speed_m_s=.65):
        x, y, z, yaw, pitch, roll = values
        a = math.radians(heading)
        position = (math.cos(a)*x-math.sin(a)*z, y, math.sin(a)*x+math.cos(a)*z)
        chest = np.asarray(poses[6][:3])
        torso = Rotation.from_quat(poses[6][3:], scalar_first=True)
        shoulder = shoulder_position(chest, torso, -1 if index == 1 else 1)
        if math.dist(position, shoulder) > .85:
            raise ValueError("Wrist target is beyond reach; approach or change stance")
        start = poses[index]
        end = (*position, *rotation(heading+yaw, pitch, roll))
        duration = max(.35, math.dist(start[:3], end[:3]) / speed_m_s,
                       2*math.acos(min(1, abs(np.dot(start[3:], end[3:]))))/3)
        return start, end, duration

    def move_wrist(self, index, start, end, duration, elapsed):
        t = min(1, elapsed / duration)
        f = t*t*(3-2*t)
        xyz = tuple(a+(b-a)*f for a,b in zip(start[:3], end[:3]))
        self.hands[index] = (*xyz, *quaternion_mix(start[3:], end[3:], f))
        return t == 1

    def begin_route_step(self, body, now):
        self.keys.clear()
        step = self.route_steps[self.route_index]
        if step["kind"] == "turn":
            self.heading = (body.yaw + step["yaw"], max(-80, min(80, body.pitch + step["pitch"])))
        else:
            self.keys.add(self.walk_keys[step["direction"]])
            self.pulse_end = now + step["duration"]

    def turn_step(self, body, dt, speed_deg_s=100):
        yaw = (self.heading[0] - body.yaw + 180) % 360 - 180
        pitch = self.heading[1] - body.pitch
        step = min(.05, dt) * speed_deg_s
        body.yaw += max(-step, min(step, yaw))
        body.pitch += max(-step, min(step, pitch))
        return max(abs(yaw), abs(pitch)) < .5

    def start(self, target, image, body, poses, now):
        self.keys.clear()
        self.started = now
        self.last_image = -1
        kind = target["kind"]
        v = target["values"]
        if kind not in ("menu_aim", "menu_click", "menu_double_click"):
            self.menu_aimed = False
        if kind == "rest":
            body.rest.change(target["gesture"], v[0] if v else .45)
        elif kind in ("pickup","place","throw"):
            hand = 1 if target["side"]=="left" else 2
            if (kind=="pickup") == (hand in self.carrying):
                raise ValueError("Requested action conflicts with the current grip state")
            self.hand_action = HandAction(kind,target["side"],v[:3],poses[hand],body.yaw)
            self.hand_action.orientation = rotation(body.yaw+v[3],v[4],v[5])
            self.original_orientation = poses[hand][3:]
        elif kind == "leap":
            if body.rest.name!="standing" or not body.rest.settled:
                raise ValueError("Stand and settle before leaping")
        elif kind == "walk":
            self.keys.add(self.walk_keys[target.get("direction", "forward")])
            self.pulse_end = now + v[0]
        elif kind == "route":
            self.route_steps = v
            self.route_index = 0
            self.begin_route_step(body, now)
        elif kind == "menu_aim":
            x, y = v
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("Menu coordinates must be between zero and one")
            side = target.get("side", "right")
            self.hand_index = 1 if side == "left" else 2
            self.start_pose = poses[self.hand_index]
            self.end_pose = self.pointer_pose(poses[0], x, y, side)
            self.duration = .28
            self.menu_aimed = False
        elif kind == "pointer_pair":
            self.wrist_pair = [(poses[index], self.pointer_pose(poses[0], *xy, side), .28)
                               for index, (side, xy) in enumerate(zip(("left", "right"), v), 1)]
        elif kind == "select_by_name":
            self.selection = Selection(target["label"],target.get("expected",""),target.get("tooltip",""),poses,now)
            self.previous_hands = self.hands.copy()
            self.hands.clear()
        elif kind in ("menu_click", "menu_double_click"):
            if not self.menu_aimed:
                raise ValueError("Aim at a menu item and observe its hover before clicking")
            self.menu_aimed = False
            self.keys.add("x")
            self.pulse_end = now + target.get("hold_seconds", .12)
            if kind == "menu_double_click":
                self.second_click_after = None
                self.second_click_started = False
        elif kind == "menu_toggle":
            self.keys.add("v")  # left Index B; mapped by the user's SteamVR bindings
            self.pulse_end = now + target.get("hold_seconds", .12)
        elif kind == "view":
            self.heading = (body.yaw + v[0], max(-80, min(80, body.pitch + v[1])))
        elif kind == "approach":
            gray = self.gray(image)
            h, w = gray.shape
            x, y, width, height = v[:4]
            if x + width > 1 or y + height > 1:
                raise ValueError("Target rectangle extends beyond the image")
            self.patch = gray[int(y*h):int((y+height)*h), int(x*w):int((x+width)*w)].copy()
            if min(self.patch.shape) < 6 or self.patch.std() < 5:
                raise ValueError("Target has too little visible detail to track")
            self.center = ((x + width/2)*w, (y + height/2)*h)
            self.previous_approach = None
            self.approach_stalls = 0
        elif kind == "wrist":
            side = target["side"]
            self.hand_index = 1 if side == "left" else 2
            self.start_pose, self.end_pose, self.duration = self.wrist_motion(v, self.hand_index, poses, body.yaw)
        elif kind == "wrist_pair":
            self.wrist_pair = [self.wrist_motion(values, index, poses, body.yaw, target.get("speed_m_s", .65))
                               for index, values in enumerate(v, 1)]
        elif kind == "confirm_fbt":
            self.hands.clear()
            self.grips.clear()
            self.keys.add("t")
        elif kind == "gesture":
            self.hands.clear()
            self.duration = body.request_gesture(target["gesture"], now, target.get("repetitions"))
        elif kind == "motion":
            self.job = MotionJob(target["gesture"], v[0], target["keyframes"])
            self.initial = poses
        elif kind == "camera":
            if self.dolly is None:
                raise ValueError("Enable Camera in the pilot to steer the lens")
            self.dolly.start(v, now)
        self.target = target

    @staticmethod
    def pointer_pose(head, x, y, side="right"):
        head = np.asarray(head)
        orient = Rotation.from_quat(head[[4,5,6,3]])
        mirror = -1 if side == "left" else 1
        point = head[:3] + orient.apply([-.032*mirror+.7*(2*x-1), .7*(1-2*y), -.7])
        origin = head[:3] + orient.apply([.07*mirror,-.07,-.12])
        dx,dy,dz = point-origin
        return (*origin,*rotation(math.degrees(math.atan2(dx,-dz)),
                                  math.degrees(math.atan2(dy,math.hypot(dx,dz)))))

    @staticmethod
    def gray(image):
        return np.asarray(image.convert("L").resize((320, 240)))

    def locate(self, image):
        gray = self.gray(image)
        h, w = gray.shape
        cx, cy = self.center
        x, y = max(0,int(cx)-105), max(0,int(cy)-100)
        area = gray[y:min(h,int(cy)+100), x:min(w,int(cx)+105)]
        best = None
        for scale in (.80, .85, 1., 1.18, 1.25):
            template = cv2.resize(self.patch, None, fx=scale, fy=scale)
            th, tw = template.shape
            if th > area.shape[0] or tw > area.shape[1]:
                continue
            scores = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = score, x+loc[0], y+loc[1], tw, th
        if best is None or best[0] < .72:
            return None
        _, x, y, tw, th = best
        self.center = (x+tw/2, y+th/2)
        self.patch = gray[y:y+th, x:x+tw].copy()
        return self.center[0]/w, th/h

    def tick(self, body, image, stamp, now, dt):
        """Return an outcome when the target ends; movement keys expire every pulse."""
        if now >= self.pulse_end:
            self.keys.clear()
        kind = self.target["kind"]
        elapsed = now-self.started
        if kind == "confirm_fbt":
            if elapsed >= 2:
                return "Calibration pose and both trigger pulses sent; verify avatar movement"
            self.keys.add("t")
            if 1 <= elapsed < 1.5:
                self.keys.update(("z","x"))
            return
        if kind == "select_by_name":
            if image is None or now-stamp > 2:
                outcome = "Clicked once; headset stale, destination unverified" if self.selection.clicked else "No click: stale headset"
            else:
                outcome = self.selection.tick(self,image,stamp,now)
            if outcome:
                if self.selection.pending:
                    self.selection.pending.cancel()
                self.selection = None
                self.hands = self.previous_hands
                self.keys.clear()
            return outcome
        if image is None or now-stamp > 2:
            return "stale camera"
        if kind == "walk":
            return "Released the virtual thumbstick; verify displacement" if now >= self.pulse_end else None
        if kind == "route":
            step = self.route_steps[self.route_index]
            complete = (self.turn_step(body, dt, step.get("speed_deg_s", 100))
                        if step["kind"] == "turn" else now >= self.pulse_end)
            if complete:
                self.route_index += 1
                if self.route_index == len(self.route_steps):
                    return "Movement sequence executed; verify arrival in a fresh frame"
                self.begin_route_step(body, now)
            if elapsed >= 10:
                return "Movement sequence timed out; arrival unconfirmed"
            return
        if kind == "rest":
            if body.rest.settled:
                return body.rest.name+" tracker pose reached; inspect support in world"
            return
        if kind in ("pickup","place","throw"):
            action = self.hand_action
            t = min(1., elapsed/action.duration)
            action.step(t,body,self)
            p = self.hands[action.index]
            self.hands[action.index] = (*p[:3],*quaternion_mix(self.original_orientation,p[3:],min(1,t/.35)))
            if t>.8:
                returning = min(1,(t-.8)/.2)
                self.hands[action.index] = (*p[:3],*quaternion_mix(self.hands[action.index][3:],rotation(body.yaw),returning))
            if t==1:
                if kind=="pickup":
                    self.carrying.add(action.index)
                else:
                    self.carrying.discard(action.index)
                self.hands.pop(action.index)
                return kind+" controller sequence finished; verify object in fresh view"
            return
        if kind=="leap":
            if elapsed<.25:
                self.keys.add("c")
            elif elapsed<.85:
                self.keys.add("w")
                if elapsed<.4:
                    self.keys.add("space")
            elif elapsed<1.08:
                self.keys.add("c")
            elif elapsed>1.5:
                return "Jump and landing inputs finished; verify support and displacement"
            return
        if kind == "camera":
            return self.dolly.tick(now)
        if kind == "motion":
            try:
                if self.job:
                    if elapsed > 180:
                        raise ValueError("Kimodo generation timed out")
                    path = self.job.poll()
                    if path is None:
                        return
                    self.playback = Playback(path, self.initial, body.yaw)
                    self.job.close()
                    self.job = None
                    self.started = now
                self.clip_time = now-self.started
                if self.clip_time >= self.playback.duration+2*self.playback.blend:
                    self.playback = None
                    return "Kimodo tracker playback finished"
            except (OSError, ValueError, KeyError) as error:
                if self.job:
                    self.job.close()
                self.job = self.playback = None
                return str(error)
            return
        if elapsed > 9:
            return "timeout; position unconfirmed"
        if kind == "view":
            if self.turn_step(body, dt):
                return "head orientation reached"
        elif kind == "approach":
            if stamp <= self.last_image or stamp < self.pulse_end:
                return
            self.last_image = stamp
            found = self.locate(image)
            if found is None:
                return "visual target lost"
            center, size = found
            if self.previous_approach is not None and np.linalg.norm(np.subtract(found, self.previous_approach)) < .006:
                self.approach_stalls += 1
                if self.approach_stalls >= 2:
                    return "No visible progress toward the target; inspect the path"
            else:
                self.approach_stalls = 0
            self.previous_approach = found
            desired_x, desired_height = self.target["values"][4:]
            if abs(center-desired_x) > .06:
                self.keys = {"d" if center > desired_x else "a"}
            elif size < desired_height*.90:
                self.keys = {"w"}
            elif size > desired_height*1.15:
                self.keys = {"s"}
            else:
                return "visual position reached"
            self.pulse_end = now+.12
        elif kind == "menu_double_click":
            if now >= self.pulse_end:
                if self.second_click_started:
                    return "Two trigger pulses released; inspect the next headset frame"
                if self.second_click_after is None:
                    self.second_click_after = now + .07
                elif now >= self.second_click_after:
                    self.keys.add("x")
                    self.pulse_end = now + self.target.get("hold_seconds", .12)
                    self.second_click_started = True
        elif kind in ("menu_toggle","menu_click"):
            if now >= self.pulse_end:
                return "button released; inspect the next headset frame"
        elif kind in ("wrist_pair", "pointer_pair"):
            finished = [self.move_wrist(index, *motion, elapsed)
                        for index, motion in enumerate(self.wrist_pair, 1)]
            if all(finished):
                return "both controller poses reached; grips unchanged"
        elif kind in ("wrist","menu_aim"):
            if self.move_wrist(self.hand_index, self.start_pose, self.end_pose, self.duration, elapsed):
                if kind == "menu_aim":
                    self.menu_aimed = self.hand_index == 2
                    return "pointer positioned; inspect hover before clicking"
                grip = self.target["grip"]
                key = "q" if self.hand_index == 1 else "e"
                if grip == "close":
                    self.grips.add(key)
                elif grip == "open":
                    self.grips.discard(key)
                elif grip == "click":
                    if self.trigger is None:
                        self.trigger = "z" if self.hand_index == 1 else "x"
                        self.keys.add(self.trigger)
                        self.pulse_end = now + .12
                    if now < self.pulse_end:
                        return
                    self.trigger = None
                return "controller pose reached; grip " + grip
        elif kind == "gesture" and elapsed >= self.duration:
            return "gesture stream finished"

    def overlay(self, frame):
        poses, left, right = frame
        poses = list(self.selection.poses if self.selection else poses)
        for index, pose in self.hands.items():
            poses[index] = pose
        if self.hands:
            poses = procedural_trackers(poses)
        if self.playback:
            poses = self.playback.sample(self.clip_time, poses)
        return poses, left, right
