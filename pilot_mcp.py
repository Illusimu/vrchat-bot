"""MCP tools for the currently running desktop pilot."""
import asyncio
from concurrent.futures import Future
from contextvars import ContextVar
import io
import math
from queue import Empty, SimpleQueue
import threading
from time import monotonic
from typing import Literal
from uuid import uuid4

from mcp.server.fastmcp import FastMCP, Image
from PIL import ImageDraw

from body import gesture_count
from menu_selection import GAME_MIRROR_AIM_OFFSET, PointerCalibration, laser_tip
from menu_workflows import AUDIO, LAUNCH, VolumeWorkflow, locate, stereo_menu_geometry
from spatial_geometry import measure_surface, palm_step
from world_menu_route import WorldRoute


class PilotBridge:
    def __init__(self, pilot):
        self.pilot = pilot
        self.queue = SimpleQueue()
        self.aim_lock = asyncio.Lock()
        self.control_owner = None
        self.trial = ContextVar("trial", default=None)
        self.trial_lock = asyncio.Lock()
        self.server = FastMCP("VRChat Pilot", host="127.0.0.1", port=29746,
                              stateless_http=True, json_response=True)

        @self.server.tool()
        async def observe(view: Literal["headset","left","right","stereo","camera"]="headset") -> Image:
            """Read the menu-compatible headset view, either native eye, their simultaneous stereo pair, or the handheld camera."""
            frame = await self.call("observe",view)
            data = io.BytesIO()
            frame.save(data,"JPEG",quality=90)
            return Image(data=data.getvalue(),format="jpeg")

        @self.server.tool()
        async def menu(action: Literal["toggle","aim","click","double_click"], x: float=.5, y: float=.5,
                       hold_seconds: float=.12) -> str:
            """Present the menu, aim at headset x/y, or click at the existing aim. Double-click uses two short taps."""
            if not .12 <= hold_seconds <= 3:
                raise ValueError("Trigger duration must be between 0.12 and 3 seconds")
            ticket = await self.call("menu",(action,x,y,.12 if action=="double_click" else hold_seconds))
            for _ in range(60):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result",ticket)
                if outcome:
                    return outcome
            raise ValueError("No fresh headset frame after menu action; inspect status before retrying")

        @self.server.tool()
        async def headset_state(view: Literal["headset","left","right","stereo"]="headset") -> dict:
            """Report frame identity, age, commanded poses and closed grip inputs at capture start."""
            _, metadata = await self.call("headset_snapshot",view)
            return metadata

        @self.server.tool()
        async def pointer_preview(x: float, y: float) -> Image:
            """Show the current headset frame, target and estimated right laser tip."""
            image, metadata = await self.call("headset_snapshot","headset")
            preview = image.convert("RGB")
            width,height = preview.size
            draw = ImageDraw.Draw(preview)
            for point,color in (((x,y),"yellow"),(laser_tip(preview,target=(x,y)),"magenta")):
                if point is not None:
                    px,py = point[0]*width,point[1]*height
                    draw.ellipse((px-6,py-6,px+6,py+6),outline=color,width=2)
            draw.text((8,8),f"Frame {metadata['frame_id']}  {metadata['age_ms']} ms",fill="white",stroke_width=2,stroke_fill="black")
            data = io.BytesIO()
            preview.save(data,"PNG")
            return Image(data=data.getvalue(),format="png")

        @self.server.tool()
        async def aim_at(x: float, y: float, seed: list[float] | None = None,
                         tolerance: float = .015) -> dict:
            """Aim at an image coordinate using fresh laser observations. Returns evidence and leaves the trigger released."""
            if not (0 <= x <= 1 and 0 <= y <= 1 and .005 <= tolerance <= .06):
                raise ValueError("Use normalized screen coordinates and a tolerance between 0.005 and 0.06")
            if seed is not None and (len(seed) != 2 or any(not 0 <= coordinate <= 1 for coordinate in seed)):
                raise ValueError("Seed must contain two normalized aim coordinates")
            if self.aim_lock.locked():
                raise ValueError("Pointer calibration is already running")
            async with self.aim_lock:
                calibration = PointerCalibration((x,y))
                previous, metadata = await self.call("headset_snapshot","headset")
                if seed is None and metadata.get("source") == "game_mirror":
                    dx, dy = GAME_MIRROR_AIM_OFFSET
                    seed = [max(0., x - dx), max(0., y - dy)]
                if seed is not None:
                    calibration.initial[:] = seed
                aim = calibration.probe()
                tip = None
                for attempt in range(1,9):
                    await menu("aim",*aim)
                    image, fresh = await self.call("headset_snapshot","headset")
                    if fresh["frame_id"] <= metadata["frame_id"]:
                        raise ValueError("No new headset frame after moving the pointer")
                    tip = await asyncio.to_thread(laser_tip,image,previous,(x,y))
                    if tip and max(abs(tip[0]-x),abs(tip[1]-y)) <= tolerance:
                        return dict(aligned=True,frame_id=fresh["frame_id"],tip=tip,
                                    aim=aim.tolist(),attempts=attempt)
                    previous, metadata = image,fresh
                    aim = calibration.correct(aim,tip) if tip else calibration.probe()
                    if aim is None:
                        break
                return dict(aligned=False,frame_id=metadata["frame_id"],tip=tip,
                            attempts=attempt,reason="Laser did not converge on the target")

        @self.server.tool()
        async def menu_scroll(direction: Literal["up","down"], seconds: float=.35) -> str:
            """Scroll the menu panel under the right pointer with a bounded right-stick pulse."""
            if not .1 <= seconds <= 1.5:
                raise ValueError("Duration must be between 0.1 and 1.5 seconds")
            await self.call("menu_scroll",(1. if direction=="up" else -1.,seconds))
            await asyncio.sleep(seconds+.2)
            return "Scroll pulse finished; observe the panel"

        @self.server.tool()
        async def menu_geometry() -> dict:
            """Estimate a known menu's tilt and target position in headset axes from one stereo capture."""
            pair, metadata = await self.call("headset_snapshot", "stereo")
            geometry = await asyncio.to_thread(stereo_menu_geometry, pair)
            return dict(frame_id=metadata["frame_id"], **geometry)

        @self.server.tool()
        async def locate_surface(x: float, y: float, radius_px: int = 14) -> dict:
            """Measure a point chosen in the left eye. Returns VRChat world XYZ with Y up."""
            pair, metadata = await self.call("headset_snapshot", "stereo")
            result = await asyncio.to_thread(measure_surface, pair, x, y, radius_px)
            return dict(frame_id=metadata["frame_id"], **result)

        @self.server.tool()
        async def hover_palm(x: float, y: float, side: Literal["left", "right"] = "right",
                             height_m: float = .10, radius_px: int = 14) -> dict:
            """Move an open palm above a horizontal surface selected in the left eye."""
            if not .04 <= height_m <= .35:
                raise ValueError("Use a hover height between 0.04 and 0.35 metres")
            if self.control_owner is not None or self.aim_lock.locked():
                raise ValueError("Another controller operation is running")
            self.control_owner = asyncio.current_task()
            try:
                pair, metadata = await self.call("headset_snapshot", "stereo")
                point = await asyncio.to_thread(measure_surface, pair, x, y, radius_px)
                normal = point["surface_normal"]
                if normal is None or normal[1] < .82:
                    raise ValueError("Select a clear point on an upward-facing surface")
                goal = point["world_xyz_m"].copy()
                goal[1] += height_m
                previous = None
                blocked = None
                for step in range(6):
                    if metadata["closed_grips"]:
                        raise ValueError("Release the held object before moving to a hover point")
                    error, palm, command = palm_step(pair, metadata["commanded_poses"], side, goal)
                    if blocked is None and previous is not None and previous-error < .005:
                        blocked = "The measured palm stopped getting closer."
                    if error <= .025 or step == 5 or blocked:
                        return dict(reached=error <= .025, steps=step, error_m=error,
                                    palm_world_xyz_m=palm, goal_world_xyz_m=goal, surface=point,
                                    reason=None if error <= .025 else blocked or "Correction limit reached.")
                    if previous is None and error > .65:
                        raise ValueError("The surface is too far from the palm. Approach it first.")
                    previous = error
                    try:
                        outcome = await target("wrist", command, side)
                    except ValueError as error:
                        if "beyond reach" not in str(error) and "outside supported coordinates" not in str(error):
                            raise
                        return dict(reached=False, steps=step, error_m=previous,
                                    palm_world_xyz_m=palm, goal_world_xyz_m=goal, surface=point,
                                    reason="The controller target is outside its reach. Move closer or change stance.")
                    if not outcome.startswith("controller pose reached"):
                        raise ValueError(outcome)
                    pair, metadata = await self.call("headset_snapshot", "stereo")
            finally:
                self.control_owner = None

        @self.server.tool()
        async def present_quick_menu() -> dict:
            """Present the closed Quick Menu at a readable angle and verify the Launch Pad."""
            if self.control_owner is not None or self.aim_lock.locked():
                raise ValueError("Another menu operation is running")
            self.control_owner = asyncio.current_task()
            try:
                image, metadata = await self.call("headset_snapshot", "headset")
                if known := await asyncio.to_thread(locate, image, (*LAUNCH, *AUDIO)):
                    return {"verified": True, "panel": known.panel.name, "frame_id": metadata["frame_id"]}
                if not metadata.get("commanded_poses"):
                    return {"verified": False, "panel": None, "frame_id": metadata["frame_id"],
                            "reason": "No head pose available to present the menu"}
                w, x, y, z = metadata["commanded_poses"][0][3:]
                pitch = math.degrees(math.asin(max(-1., min(1., 2 * (w*x - y*z)))))
                await target("view", [0, max(-80., min(80., -5 - pitch))])
                await target("wrist", [-.17, 1.48, -.40, 0, 100, -12])
                image, metadata = await self.call("headset_snapshot", "headset")
                known = await asyncio.to_thread(locate, image, (*LAUNCH, *AUDIO))
                if known is None:
                    await menu("toggle")
                    image, metadata = await self.call("headset_snapshot", "headset")
                    known = await asyncio.to_thread(locate, image, (*LAUNCH, *AUDIO))
                return {"verified": known is not None, "panel": known.panel.name if known else None,
                        "frame_id": metadata["frame_id"]}
            finally:
                self.control_owner = None

        @self.server.tool()
        async def set_ui_volume(percent: int) -> dict:
            """Open the in-game Audio panel and set UI/Menu volume. Checks the slider fill and returns an estimated percentage."""
            if not 5 <= percent <= 95:
                raise ValueError("Use a percentage from 5 to 95 for visual slider control")
            if self.control_owner is not None or self.aim_lock.locked():
                raise ValueError("Another menu operation is running")
            self.control_owner = asyncio.current_task()
            try:
                workflow = VolumeWorkflow(
                    lambda: self.call("headset_snapshot", "headset"),
                    menu,
                    lambda pitch: target("view", [0, pitch]),
                    lambda: target("wrist", [-.20, 1.43, -.48, 0, 68, 0]),
                )
                return await workflow.run(percent)
            except asyncio.CancelledError:
                await self.call("stop", None)
                raise
            finally:
                self.control_owner = None

        async def execute_world_route(world: str, *, details_only: bool) -> dict:
            if self.control_owner is not None or self.aim_lock.locked():
                raise ValueError("Another menu operation is running")
            self.control_owner = asyncio.current_task()
            try:
                route = WorldRoute(
                    lambda: self.call("headset_snapshot", "headset"),
                    menu,
                    lambda pitch, yaw=0: target("view", [yaw, pitch]),
                    world_context,
                )
                return await route.run(world, details_only=details_only)
            except asyncio.CancelledError:
                await self.call("stop", None)
                raise
            finally:
                self.control_owner = None

        @self.server.tool()
        async def join_invite_world(world: str) -> dict:
            """Join a mapped world through VRChat's menus. Start on Audio, Launch Pad, or Worlds."""
            return await execute_world_route(world, details_only=False)

        @self.server.tool()
        async def find_world_details(world: str) -> dict:
            """Find a mapped world and verify its details and creator while staying in the current instance."""
            return await execute_world_route(world, details_only=True)

        @self.server.tool()
        async def select_by_name(label: str, expected: str="", tooltip: str="") -> str:
            """Select one visible English VRChat menu control. New Instance has defaults; otherwise supply the new screen heading and hover tooltip."""
            ticket = await self.call("select_by_name",(label,expected,tooltip))
            try:
                for _ in range(520):
                    await asyncio.sleep(.1)
                    outcome=await self.call("menu_result",ticket)
                    if outcome:
                        if not outcome.startswith("Verified "):
                            raise ValueError(outcome)
                        return outcome
                raise ValueError("Selection timed out; inspect the current screen before retrying")
            finally:
                await self.call("cancel_selection",ticket)

        @self.server.tool()
        async def target(kind: Literal["view","wrist","camera","approach"], values: list[float],
                         side: Literal["left","right"]="left") -> str:
            """Set a view, wrist, or camera target, or approach a visible landmark.

            Approach takes [x,y,width,height,desired_center_x,desired_height], all fractions of the headset image.
            View takes [relative yaw,pitch]. Wrist takes [x,y,z,yaw,pitch,roll].
            Camera takes player-local [x,y,z,pitch,yaw,roll]. Positions use metres and angles use degrees.
            """
            ticket = await self.call("target",(kind,values,side))
            for _ in range(150):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result",ticket)
                if outcome:
                    return outcome
            raise ValueError("Target result unavailable")

        @self.server.tool()
        async def grip(side: Literal["left","right"], closed: bool) -> dict:
            """Close or open one controller grip without moving the wrist. Inspect the object to verify contact."""
            result = await self.call("grip", (side, closed))
            fresh_after = monotonic() + .08
            for _ in range(8):
                await asyncio.sleep(.10)
                try:
                    _, view = await self.call("headset_snapshot", "headset")
                except ValueError as error:
                    if "No current headset frame" not in str(error):
                        raise
                    continue
                if view["capture_started"] >= fresh_after:
                    return result
            return {**result, "observation": "No fresh frame yet; recheck the object before moving."}

        @self.server.tool()
        async def body_action(action: Literal["rest","pickup","place","throw","leap","walk"], values: list[float],
                              side: Literal["left","right"]="right", posture: str="standing",
                              direction: Literal["forward","backward","left","right"]="forward") -> str:
            """Walk takes [seconds] (.1–2) and a direction. Rest takes seat height. Manipulation takes xyz and yaw/pitch/roll."""
            ticket = await self.call("body_action",(action,values,side,posture,direction))
            for _ in range(150):
                await asyncio.sleep(.1)
                result = await self.call("menu_result",ticket)
                if result:
                    return result
            raise ValueError("Action result unavailable")

        @self.server.tool()
        async def confirm_fbt() -> str:
            """After opening VRChat Calibrate FBT, hold a T pose and confirm with both triggers."""
            ticket = await self.call("confirm_fbt", None)
            for _ in range(60):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result", ticket)
                if outcome:
                    return outcome
            raise ValueError("Calibration result unavailable")

        @self.server.tool()
        async def gesture(name: Literal["wave","nod","backflip","jumping-jacks","push-ups","leg-lift"],
                          repetitions: int | None = None) -> str:
            """Play one gesture command for 1–5 repetitions. Defaults to one, three push-ups, or two jumping jacks."""
            count = gesture_count(name, repetitions)
            ticket = await self.call("gesture", (name, count))
            for _ in range(110):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result", ticket)
                if outcome == "gesture stream finished":
                    return f"Gesture stream finished. Commanded repetitions: {count}."
                if outcome:
                    return outcome
            raise ValueError("Gesture timed out")

        @self.server.tool()
        async def camera(enabled: bool) -> str:
            """Connect or disconnect the VRChat Stream camera."""
            return await self.call("camera",enabled)

        @self.server.tool()
        async def camera_zoom(value: float | None = None) -> dict:
            """Read or set zoom (20–150). Smaller values give a wider shot."""
            if value is None:
                return {"zoom": await self.call("camera_zoom", None)}
            if not 20 <= value <= 150:
                raise ValueError("Camera zoom must be between 20 and 150")
            await self.call("camera_zoom", value)
            for _ in range(10):
                await asyncio.sleep(.10)
                current = await self.call("camera_zoom", None)
                if abs(current - value) <= .5:
                    await asyncio.sleep(.65)
                    return {"zoom": current}
            raise ValueError("VRChat did not confirm the requested zoom")

        @self.server.tool()
        async def take_photo() -> str:
            """Press the shutter on the connected VRChat camera."""
            return await self.call("take_photo", None)

        @self.server.tool()
        async def run_objective(objective: str) -> str:
            """Have the local visual model execute a text objective in the pilot."""
            return await self.call("objective",objective)

        @self.server.tool()
        async def map_status() -> dict:
            """Get the active world's visual region, estimated location and latest region change."""
            return await self.call("map_status",None)

        @self.server.tool()
        async def open_map() -> dict:
            """Show Desktop Body's current world fly-through viewer."""
            return await self.call("open_map",None)

        @self.server.tool()
        async def world_context() -> dict:
            """Read the confirmed VRChat world ID, name and opaque instance key."""
            from world_context import world_context as read_context
            return await asyncio.to_thread(read_context)

        @self.server.tool()
        async def status() -> str:
            """Read the current pilot status and objective."""
            return await self.call("status",None)

        @self.server.tool()
        async def stop() -> str:
            """Cancel pilot activity and release held inputs."""
            if self.control_owner is not None:
                self.control_owner.cancel()
            return await self.call("stop",None)

        trial_tools = {tool.__name__: tool for tool in (
            aim_at, menu, menu_scroll, select_by_name, target, grip, body_action,
            confirm_fbt, gesture, camera, camera_zoom, take_photo, join_invite_world, find_world_details, set_ui_volume,
            present_quick_menu,
        )}

        @self.server.tool()
        async def trial_control(action: Literal["begin", "check", "step", "end"],
                                lease: str = "", tool: str = "",
                                arguments: dict | None = None) -> dict:
            """Run one benchmark action under an exclusive, locally revocable lease."""
            if action in ("check", "end"):
                if not lease:
                    raise ValueError("A trial lease is required")
                token = self.trial.set(lease)
                try:
                    command = "status" if action == "check" else "trial_end"
                    return {"result": await self.call(command, None)}
                finally:
                    self.trial.reset(token)
            if self.trial_lock.locked():
                raise ValueError("A benchmark action is still running")
            async with self.trial_lock:
                if action == "begin":
                    return {"lease": await self.call("trial_begin", uuid4().hex)}
                if not lease:
                    raise ValueError("A trial lease is required")
                token = self.trial.set(lease)
                try:
                    if tool not in trial_tools:
                        raise ValueError("Tool is outside the benchmark controls")
                    await self.call("status", None)
                    return {"result": await trial_tools[tool](**(arguments or {}))}
                finally:
                    self.trial.reset(token)

    async def call(self, command, value):
        if self.control_owner is not None and asyncio.current_task() is not self.control_owner:
            if command not in ("observe", "headset_snapshot", "status", "stop"):
                raise ValueError("A visual workflow owns the controls")
        reply = Future()
        self.queue.put((command,value,reply,self.trial.get()))
        return await asyncio.wait_for(asyncio.wrap_future(reply),5)

    def drain(self):
        for _ in range(16):
            try:
                command,value,reply,lease = self.queue.get_nowait()
            except Empty:
                break
            if not reply.set_running_or_notify_cancel():
                continue
            try:
                reply.set_result(self.pilot.remote(command,value,lease))
            except Exception as error:
                reply.set_exception(error)

    def start(self):
        threading.Thread(target=self.server.run,kwargs={"transport":"streamable-http"},
                         daemon=True,name="pilot-mcp").start()
