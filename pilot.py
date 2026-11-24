"""Focused desktop controls and a local preview of the six tracking points."""

import socket
import time
import tkinter as tk
import argparse
import json
from activity_view import ActivityColumn
from concurrent.futures import ThreadPoolExecutor
from vr_agent import decide, observe, pilot_views, LIMITS
from window_capture import close_capture_sessions
from targets import TargetSolver
from camera_dolly import CameraDolly
from PIL import ImageTk
from world_map import MappingWorker
from map_view import MapView

from body import Body, DEVICE_NAMES, PORT, pack_frame


class Pilot:
    def __init__(self, root, output, game_mirror=False):
        self.root = root
        self.output = output
        self.game_mirror = game_mirror
        self.body = Body()
        self.keys = set()
        self.pressed_since_frame = set()
        self.mouse = [0, 0]
        self.drag = None
        self.active = False
        self.objective = ""
        self.bridge = None
        self.remote_ticket = 0
        self.external = False
        self.trial_lease = None
        self.trial_deadline = 0.
        self.remote_outcome = ""
        self.scroll_until = 0.
        self.scroll_axis = 0.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.camera = ThreadPoolExecutor(max_workers=2)
        self.stream_worker = ThreadPoolExecutor(max_workers=1)
        self.stream_capture = self.stream_frame = None
        self.stream_time = 0
        self.action_view = None
        self.capture = None
        self.stereo_frame = None
        self.frame = None
        self.frame_time = 0
        self.frame_id = 0
        self.frame_poses = None
        self.frame_grips = ()
        self.frame_source = None
        self.solver = TargetSolver()
        self.last_poses = None
        self.last_grips = ()
        self.future = None
        self.history = []
        self.observe_after = 0
        self.last = time.monotonic()

        root.title("Desktop Body")
        root.geometry("1460x800")
        root.minsize(880, 620)
        bar = tk.Frame(root)
        bar.pack(fill="x")
        self.prompt = tk.Entry(bar)
        self.prompt.insert(0, "Find the stage floodlight in the room and do a backflip next to it.")
        self.prompt.pack(side="left", expand=True, fill="x")
        tk.Button(bar, text="Run", command=self.start_agent).pack(side="left")
        tk.Button(bar, text="Stop", command=self.pause).pack(side="left")
        self.camera_enabled = tk.BooleanVar()
        self.map_worker = MappingWorker()
        self.map_executor = ThreadPoolExecutor(max_workers=1)
        self.map_future = None
        self.map_capture = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Map capture",variable=self.map_capture).pack(side="left")
        tk.Checkbutton(bar, text="VRChat camera", variable=self.camera_enabled,
                       command=self.toggle_camera).pack(side="left")
        panes = tk.Frame(root)
        panes.pack(fill="both", expand=True, padx=8, pady=4)
        panels = []
        view_name = "VRChat game mirror" if game_mirror else "Headset · left and right eyes"
        for index, title in enumerate((view_name, "VRChat camera", "Body trackers", "World memory")):
            row, column = divmod(index, 2)
            panes.rowconfigure(row, weight=1, uniform="rows")
            panes.columnconfigure(column, weight=1, uniform="columns")
            panel = tk.LabelFrame(panes, text=title)
            panel.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
            panel.pack_propagate(False)
            panels.append(panel)
        self.activity = ActivityColumn(panes)
        self.activity.grid(row=0,column=2,rowspan=2,sticky="nsew",padx=2,pady=2)
        self.eye = tk.Label(panels[0], bg="#10151b", fg="white", text="Connecting to headset...")
        self.eye.pack(fill="both", expand=True)
        self.stream_label = tk.Label(panels[1], bg="#10151b", fg="white", text="Camera off", wraplength=230)
        self.stream_label.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(panels[2], bg="#111923", highlightthickness=0, takefocus=True)
        self.canvas.pack(fill="both", expand=True)
        self.map_view = MapView(panels[3])
        self.status = tk.StringVar(value="Paused. Click the preview to take control.")
        tk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=12)
        tk.Label(
            root, justify="left", anchor="w",
            text="WASD move  ·  Arrows or right-drag look  ·  C crouch  ·  T calibrate\n"
                 "Q/E grip  ·  Z/X trigger  ·  1–5 right fingers  ·  6–0 left fingers\n"
                 "G/V left A/B  ·  Space/B right A/B  ·  F wave  ·  N nod  ·  Esc pause",
        ).pack(fill="x", padx=12, pady=(4, 12))
        self.canvas.bind("<Button-1>", self.arm)
        self.canvas.bind("<Button-3>", self.start_drag)
        self.canvas.bind("<B3-Motion>", self.move_mouse)
        self.canvas.bind("<ButtonRelease-3>", self.end_drag)
        self.canvas.bind("<FocusOut>", self.leave_manual_controls)
        self.canvas.bind("<KeyPress>", self.press)
        self.canvas.bind("<KeyRelease>", self.release_key)
        root.bind("<Escape>", self.pause)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()
        self.refresh_view()
        self.refresh_map()

    def open_map(self):
        self.map_view.dismiss_photo()
        self.root.deiconify()
        self.root.lift()
        self.map_view.canvas.focus_set()

    def refresh_map(self):
        if self.map_future and self.map_future.done():
            try:
                self.map_view.update(self.map_future.result())
            except Exception as error:
                self.map_view.info.set(f"Mapping paused: {error}")
            self.map_future=None
        if self.map_future is None and self.map_capture.get():
            self.map_future=self.map_executor.submit(self.map_worker.step,True)
        if not self.map_capture.get():
            self.map_view.info.set("World memory paused")
        self.root.after(450,self.refresh_map)

    def toggle_camera(self, *, pause_controls=True):
        if pause_controls:
            self.pause()
        self.stream_frame = self.stream_capture = None
        self.stereo_frame = self.frame = self.capture = None
        self.frame_source = None
        if self.solver.dolly:
            self.solver.dolly.close()
            self.stream_worker.submit(self.solver.dolly.close_receiver)
            self.solver.dolly = None
            self.stream_label.configure(image="", text="Camera off")
        if self.camera_enabled.get():
            try:
                self.solver.dolly = CameraDolly(spout=self.game_mirror)
            except (OSError, ValueError) as error:
                self.camera_enabled.set(False)
                self.status.set(str(error))
                return
            self.stream_label.configure(image="", text="Connecting to VRChat camera…")

    def refresh_stream(self):
        if not self.solver.dolly:
            return
        if self.stream_capture is None:
            self.stream_started = time.monotonic()
            self.stream_capture = self.stream_worker.submit(self.solver.dolly.observe)
        elif self.stream_capture.done():
            try:
                self.stream_frame = self.stream_capture.result()
                self.stream_time = self.stream_started
                display = self.stream_frame.copy()
                display.thumbnail((max(1,self.stream_label.winfo_width()), max(1,self.stream_label.winfo_height())))
                self.stream_preview = ImageTk.PhotoImage(display)
                self.stream_label.configure(image=self.stream_preview, text="")
            except Exception as error:
                self.stream_frame = None
                self.stream_label.configure(image="", text=str(error))
            self.stream_capture = None

    def refresh_view(self):
        self.refresh_stream()
        if self.capture is None:
            self.capture_started = time.monotonic()
            self.capture_poses = self.last_poses
            self.capture_grips = self.last_grips
            require_eye = self.solver.dolly is not None and not self.solver.dolly.spout
            self.capture = self.camera.submit(self.capture_views, require_eye,
                                              bool(self.solver.selection))
        elif self.capture.done():
            try:
                self.stereo_frame, self.frame, self.frame_source = self.capture.result()
                self.frame_time = self.capture_started
                self.frame_id += 1
                self.frame_poses = self.capture_poses
                self.frame_grips = self.capture_grips
                display = (self.stereo_frame or self.frame).copy()
                display.thumbnail((max(1, self.eye.winfo_width()), max(1, self.eye.winfo_height())))
                self.eye_image = ImageTk.PhotoImage(display)
                label = "" if self.stereo_frame or self.frame_source == "game_mirror" else "Stereo unavailable · single view"
                self.eye.configure(image=self.eye_image, compound="top", text=label)
            except Exception as error:
                self.stereo_frame = self.frame = self.frame_source = None
                self.eye.configure(image="", text=str(error), wraplength=250)
                if self.objective:
                    self.pause()
                    self.status.set(f"Capture unavailable: {error}")
            self.capture = None
        self.root.after(100, self.refresh_view)

    def capture_views(self, require_eye, menu_detail):
        # With Spout on, the game window keeps its first-person view.
        if self.game_mirror and not require_eye:
            return None, observe(stream=True, menu_detail=menu_detail), "game_mirror"
        pair, eye = pilot_views(require_eye, menu_detail)
        return pair, eye, "headset"

    def arm(self, _event):
        self.pause()
        self.canvas.focus_set()
        self.active = True
        self.status.set("Controls active. Esc or focus loss pauses input.")

    def start_drag(self, event):
        if self.active:
            self.drag = event.x, event.y

    def move_mouse(self, event):
        if self.drag is not None:
            self.mouse[0] += event.x - self.drag[0]
            self.mouse[1] += event.y - self.drag[1]
            self.drag = event.x, event.y

    def end_drag(self, _event):
        self.drag = None

    def press(self, event):
        key = event.keysym.lower()
        if key == "escape":
            self.pause()
            return "break"
        if self.trial_lease or self.objective or self.external or self.solver.hands or self.solver.grips:
            self.pause()
            self.active = True
        if not self.active:
            return
        if key not in self.keys and key in ("f", "n"):
            self.body.request_gesture({"f": "wave", "n": "nod"}[key], time.monotonic())
        self.keys.add(key)
        self.pressed_since_frame.add(key)

    def release_key(self, event):
        self.keys.discard(event.keysym.lower())

    def leave_manual_controls(self, _event=None):
        if not self.objective:
            self.pause()

    def pause(self, _event=None):
        self.trial_lease = None
        self.scroll_until = 0.
        self.remote_ticket += 1
        self.external = False
        self.solver.reset()
        self.action_view = None
        self.objective = ""
        self.active = False
        self.keys.clear()
        self.pressed_since_frame.clear()
        self.mouse[:] = 0, 0
        self.drag = None
        self.body.release()
        self.status.set("Paused. Click the preview to take control.")

    def remote(self, command, value, lease=None):
        now = time.monotonic()
        if command == "trial_begin":
            if self.trial_lease or self.external or self.objective or self.keys or self.drag is not None:
                raise ValueError("Desktop Body is busy. Stop or finish the current activity first.")
            self.trial_lease = value
            self.trial_deadline = now + 90
            self.status.set("Benchmark running. Stop, Esc, or take the controls to cancel.")
            return value
        if lease is not None and self.trial_lease != lease:
            raise ValueError("Trial stopped or controls reclaimed")
        if lease is not None:
            self.trial_deadline = now + 90
        if self.trial_lease and lease is None and command not in (
            "status", "observe", "headset_snapshot", "map_status", "stop",
        ):
            raise ValueError("A benchmark owns the controls. Stop it to take over.")
        if command == "trial_end":
            if not lease:
                raise ValueError("A trial lease is required")
            self.pause()
            return "Trial released the controls"
        if command == "menu_scroll":
            if self.external or self.objective or self.keys or self.drag is not None:
                raise ValueError("Pilot busy")
            self.scroll_axis, duration = value
            self.scroll_until = now + duration
            self.active = True
            return "Right thumbstick scroll scheduled"
        if command=="map_status":
            state=self.map_view.state
            pose=state.get("pose")
            hud=state.get("hud")
            return dict(world=state.get("world"),world_id=state.get("world_id"),region=state.get("region"),
                        state=state.get("status"),keyframes=len(state.get("frames",[])),stats=state.get("stats",{}),
                        map_xyz=pose[:3,3].tolist() if pose is not None else None,
                        last_discontinuity=state.get("jump"),
                        screen_world_xyz_xyzw=hud["screen"].tolist() if hud else None,
                        photo_world_xyz_xyzw=hud["photo"].tolist() if hud and hud["photo_available"] else None,
                        frame="world-aligned stereo (Y down)" if state.get("region")==-1 else "region-local stereo")
        if command=="open_map":
            self.open_map()
            return self.remote("map_status",None)
        if command == "status":
            return f"{self.status.get()}  Objective: {self.objective or '(none)'}"
        if command == "headset_snapshot":
            image = self.remote("observe", value)
            return image, dict(frame_id=self.frame_id, capture_started=self.frame_time,
                               age_ms=round((now-self.frame_time)*1000), view=value,
                               source=self.frame_source,
                               closed_grips=self.frame_grips,
                               commanded_poses=self.frame_poses[:3] if self.frame_poses is not None else None)
        if command == "observe":
            if value=="headset":
                image,stamp=self.frame,self.frame_time
            elif value=="camera":
                image,stamp=self.stream_frame,self.stream_time
            elif value in ("left","right","stereo"):
                image,stamp=self.stereo_frame,self.frame_time
                if image is not None and value!="stereo":
                    half=image.width//2
                    start=half if value=="right" else 0
                    image=image.crop((start,0,start+half,image.height))
            else:
                raise ValueError("Unknown observation view")
            if image is None or now-stamp > 2:
                raise ValueError(f"No current {value} frame")
            return image
        if command == "cancel_selection":
            if self.external and value==self.remote_ticket and self.solver.selection:
                self.pause()
            return self.status.get()
        if command == "stop":
            self.pause()
        elif command == "menu_result":
            if value != self.remote_ticket:
                raise ValueError("Menu action superseded or paused")
            if self.external or self.frame is None or self.frame_time < self.observe_after:
                return None
            return self.remote_outcome
        elif self.objective or self.external or self.keys or self.drag is not None:
            raise ValueError("Controls are busy. Read status or stop the current action.")
        elif command == "grip":
            side, closed = value
            self.solver.set_grip(side, closed)
            return {"side": side, "grip_closed": closed, "object_contact": "unverified"}
        elif command == "camera":
            if value != self.camera_enabled.get():
                self.camera_enabled.set(value)
                self.toggle_camera(pause_controls=lease is None)
            camera = self.solver.dolly
            if value != bool(camera):
                raise ValueError(self.status.get())
            if camera:
                mode, streaming = camera.mode()
                if mode == 0 and not camera.stream_ready(mode, streaming):
                    raise ValueError("Open Photo Camera in VRChat before reconnecting the stream")
                if not camera.stream_ready(mode, streaming):
                    camera.enable_stream()
            return "Camera connected" if camera else "Camera disconnected"
        elif command == "camera_zoom":
            camera = self.solver.dolly
            if camera is None:
                raise ValueError("Connect the stream camera before adjusting its zoom")
            if value is None:
                return camera.zoom()
            camera.send("/usercamera/Zoom", float(value))
        elif command == "take_photo":
            camera = self.solver.dolly
            if camera is None or camera.mode()[0] not in (1, 2):
                raise ValueError("Open the connected camera in Photo or Stream mode before taking a picture")
            camera.send("/usercamera/Capture", True)
            return "Photo requested"
            return "Camera connected" if value else "Camera disconnected"
        elif command == "objective":
            if not value.strip():
                raise ValueError("Enter an objective")
            self.prompt.delete(0,"end")
            self.prompt.insert(0,value)
            self.start_agent()
        elif command in ("menu","target","select_by_name","gesture","confirm_fbt","body_action"):
            self.remote("observe","headset")
            if self.frame_time < self.observe_after:
                raise ValueError("Waiting for a new headset frame")
            if command=="body_action":
                action, values, side, posture, direction = value
                limits = [(.1, 2.)] if action == "walk" else LIMITS.get(action, [])
                if len(values)!=len(limits) or any(not lo<=x<=hi for x,(lo,hi) in zip(values,limits)):
                    raise ValueError("Action outside supported range")
                target=dict(kind=action,values=values,side=side,gesture=posture,direction=direction)
            elif command=="confirm_fbt":
                action="confirm_fbt"
                target=dict(kind=action,values=[])
            elif command=="gesture":
                action="gesture"
                name, repetitions = value
                target=dict(kind=action, gesture=name, repetitions=repetitions, values=[])
            elif command == "select_by_name":
                label,expected,tooltip=value
                action="select_by_name"
                target=dict(kind=action,values=[],label=label,expected=expected,tooltip=tooltip)
            elif command == "menu":
                action,x,y,hold_seconds = value
                target = dict(kind="menu_"+action, values=[x,y] if action=="aim" else [],
                              hold_seconds=hold_seconds)
            else:
                action,values,side = value
                limits = LIMITS[action]
                if len(values)!=len(limits) or any(not lo<=v<=hi for v,(lo,hi) in zip(values,limits)):
                    raise ValueError("Target outside supported coordinates")
                if action=="camera":
                    self.remote("observe","camera")
                target = dict(kind=action, values=values, side=side, grip="keep")
            self.solver.start(target,self.frame,self.body,self.last_poses,now)
            self.external = self.active = True
            self.remote_ticket += 1
            self.status.set("Menu " + action)
            return self.remote_ticket
        return self.status.get()

    def close(self):
        self.pause()
        self.map_executor.submit(self.map_worker.map.close)
        self.map_executor.shutdown(wait=False)
        if self.solver.dolly:
            self.solver.dolly.close()
            self.stream_worker.submit(self.solver.dolly.close_receiver)
            self.solver.dolly = None
        self.send(time.monotonic(), 0)
        self.solver.reader.shutdown(wait=False,cancel_futures=True)
        self.camera.shutdown(wait=False, cancel_futures=True)
        self.stream_worker.shutdown(wait=False)
        self.executor.shutdown(wait=False, cancel_futures=True)
        close_capture_sessions()
        self.root.destroy()

    def send(self, now, dt):
        self.advance_agent(now)
        steady = self.bridge is not None and self.bridge.control_owner is not None
        keys = self.solver.keys | self.solver.grips
        if not steady:
            keys.update(self.keys | self.pressed_since_frame)
        mouse = (0, 0) if steady else self.mouse
        frame = self.solver.overlay(self.body.update(keys, mouse, now, dt, steady=steady))
        if not steady and now < self.scroll_until:
            poses, left, right = frame
            frame = poses, left, (0., self.scroll_axis, *right[2:])
        self.last_poses = frame[0]
        self.last_grips = tuple(side for side, controls in zip(("left", "right"), frame[1:]) if controls[3])
        self.pressed_since_frame.clear()
        self.mouse[:] = 0, 0
        try:
            self.output.sendto(pack_frame(*frame), ("127.0.0.1", PORT))
        except OSError as error:
            self.pause()
            self.status.set(f"Local pose stream failed. {error}")
        return frame

    def start_agent(self):
        # A follow-up objective may start with a prop from the completed one.
        keep = self.solver.carrying.copy() if not (self.objective or self.external) else set()
        self.pause()
        self.solver.carrying = keep
        self.solver.grips.update("q" if hand==1 else "e" for hand in keep)
        if self.future and not self.future.done():
            self.status.set("Waiting for the previous model call to finish. Then press Run.")
            return
        self.future = None
        self.canvas.focus_set()
        self.active = True
        self.history = []
        self.objective = self.prompt.get().strip()
        self.observe_after = 0
        self.status.set("Observing VRChat…")

    def advance_agent(self, now):
        if self.solver.dolly and (not self.solver.target or self.solver.target["kind"] != "camera"):
            self.solver.dolly.idle()
        if self.external:
            outcome = self.solver.tick(self.body,self.frame,self.frame_time,now,now-self.last)
            if outcome:
                self.external = False
                self.solver.target = None
                self.solver.keys.clear()
                if outcome == "stale camera":
                    self.pause()
                self.observe_after = now+.2
                self.remote_outcome = outcome
                self.status.set(outcome)
            return
        if not self.objective:
            return
        if self.solver.target:
            target = self.solver.target
            if target["kind"] in ("motion","gesture") and self.stream_frame is not None:
                if target["kind"] == "motion":
                    mid = (self.solver.playback.duration/2 + self.solver.playback.blend
                           if self.solver.playback else float("inf"))
                else:
                    mid = self.solver.duration / 2
                if self.action_view is None and self.stream_time >= self.solver.started + mid:
                    self.action_view = self.stream_frame
            if self.solver.target["kind"] == "camera" and (self.stream_frame is None or now-self.stream_time > 2):
                self.pause()
                self.status.set("Stopped camera movement because its video feed is stale.")
                return
            outcome = self.solver.tick(self.body, self.frame, self.frame_time, now, now-self.last)
            if outcome:
                self.activity.append("Solver result · "+outcome)
                self.history[-1]["outcome"] = outcome
                self.status.set(outcome)
                self.solver.target = None
                self.solver.keys.clear()
                self.solver.trigger = None
                if outcome == "stale camera":
                    self.pause()
                    self.status.set("Stopped because the headset image is stale.")
                    return
                self.observe_after = now + .3
            return
        if now < self.observe_after:
            return
        if not self.future:
            if self.frame is None or self.frame_time < self.observe_after or now - self.frame_time > 2:
                return
            if len(self.history) >= 60:
                self.pause()
                self.status.set("Stopped after 60 decisions.")
                return
            self.decision_image = self.frame
            stream = self.stream_frame if now-self.stream_time < 2 and self.stream_time >= self.observe_after else None
            if (self.solver.dolly and self.history and self.history[-1]["kind"] == "camera"
                    and stream is None and now-self.observe_after < 3):
                return
            right=None
            if self.stereo_frame is not None and self.history and self.history[-1]["kind"]=="inspect_right":
                right=self.remote("observe","right")
            self.activity.append("Observing headset · waiting for local model")
            self.future = self.executor.submit(decide, self.objective, self.history.copy(), self.frame,
                                               stream, self.action_view,right,
                                               {"posture":self.body.rest.name,
                                                "closed_carry_grips":["left" if i==1 else "right" for i in sorted(self.solver.carrying)]})
            self.action_view = None
            return
        if not self.future.done():
            return
        try:
            decision = self.future.result()
            kind = decision["kind"]
            self.activity.append("Attempt · "+decision["evidence"])
            self.activity.append("Planner request · "+json.dumps({k:v for k,v in decision.items() if k!="evidence"},ensure_ascii=False))
            if self.frame is None or now - self.frame_time > 2:
                raise ValueError("Stopped because the headset image is stale.")
            flipped = any(item["kind"] == "gesture" and item["gesture"] == "backflip"
                          and item.get("outcome") == "gesture stream finished" for item in self.history)
            if kind == "done" and "backflip" in self.objective.lower() and not flipped:
                raise ValueError("Completion claimed before backflip playback")
            if kind == "gesture" and decision["gesture"] == "backflip" and flipped:
                raise ValueError("Repeat backflip requested")
            if kind == "done":
                self.objective = ""
                self.solver.keys.clear()
            elif kind == "blocked":
                self.pause()
            elif kind == "inspect_right":
                decision["outcome"]="Requested right eye" if self.stereo_frame is not None else "Stereo unavailable"
            else:
                try:
                    self.solver.start(decision, self.decision_image, self.body, self.last_poses, now)
                except ValueError as error:
                    self.activity.append("Solver error · "+str(error))
                    decision["outcome"] = str(error)
                    self.observe_after = now + .3
            self.history.append(decision)
            message = f"{kind} - {decision.get('outcome',decision['evidence'])}"
            print(message, flush=True)
            self.status.set(message)
        except Exception as error:
            self.pause()
            self.status.set(str(error))
            print(f"Agent stopped: {error}", flush=True)
        finally:
            self.future = None

    def tick(self):
        now = time.monotonic()
        if self.trial_lease and now >= self.trial_deadline:
            self.pause()
            self.status.set("Benchmark disconnected. Controls released.")
        if self.bridge:
            self.bridge.drain()
        self.draw(*self.send(now, now - self.last))
        self.last = now
        self.root.after(11, self.tick)

    def draw(self, poses, left, right):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        scale = min(width * 0.32, height * 0.43)
        center, floor = width * 0.47, height * 0.83
        points = [(center + (x - 0.35 * z) * scale,
                   floor - (y + 0.23 * z) * scale) for x, y, z, *_ in poses]
        canvas.create_text(14, 14, anchor="nw", text=self.body.motion.clip.name,
                           fill="#9ab3c9", font=("Segoe UI", 10, "bold"))
        canvas.create_line(16, floor, width - 16, floor, fill="#334457")
        for start, end in ((0, 6), (6, 3), (6, 7), (7, 1), (6, 8), (8, 2), (3, 9), (9, 4), (3, 10), (10, 5)):
            canvas.create_line(*points[start], *points[end], fill="#7893a7", width=2)
        for (x, y), label in zip(points, DEVICE_NAMES):
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#6fd3ee", outline="")
            canvas.create_text(x + 8, y - 6, text=label, anchor="sw", fill="#dcebf6")
        canvas.create_text(14, height - 12, anchor="sw", fill="#dcebf6",
                           text=f"Yaw {self.body.yaw:+.0f}°  Pitch {self.body.pitch:+.0f}°"
                                f"   Fingers L {sum(left[6:]):.0f}/5  R {sum(right[6:]):.0f}/5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="Connect the VRChat stream camera on startup")
    parser.add_argument("--mcp", action="store_true", help="Expose pilot tools at 127.0.0.1:29746/mcp")
    parser.add_argument("--game-mirror", action="store_true", help="Use VRChat's live game mirror as a single view")
    parser.add_argument("--objective", help="Start a visual objective")
    args = parser.parse_args()
    root = tk.Tk()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
        pilot = Pilot(root, output, game_mirror=args.game_mirror)
        if args.mcp:
            from pilot_mcp import PilotBridge
            pilot.bridge = PilotBridge(pilot)
            pilot.bridge.start()
        if args.camera:
            pilot.camera_enabled.set(True)
            pilot.toggle_camera()
        if args.objective and (not args.camera or pilot.solver.dolly):
            pilot.prompt.delete(0, "end")
            pilot.prompt.insert(0, args.objective)
            root.after(500, pilot.start_agent)
        root.mainloop()


if __name__ == "__main__":
    main()
