"""Run successive planner-supplied VRChat objectives under one controller lease."""

import argparse
import asyncio
import base64
from io import BytesIO
import json
from math import dist, hypot
from pathlib import Path
from time import monotonic, sleep
from typing import Annotated, Literal

import cv2
import httpx
import numpy as np
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from camera_pose import read_eye
from pilot_mcp import RouteTurn, RouteWalk
from pickup_vision import AmbiguousColorMatch, hue_mask
from reactive_vr import CONTROLS, PARKED_WRIST, PILOT, Check, ColorAim, Plan, Session, measure_color_surface, release_lease
from spatial_geometry import measure_surface_in_context, stereo_context
from vr_benchmark import Trace, TrialStopped, call
from vr_presets import PresetName


class PositionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    xz: tuple[float, float]
    radius_m: float = Field(default=.35, ge=.1, le=1)
    nudge_limit_m: float = Field(default=0, ge=0, le=.6)
    max_nudges: int = Field(default=1, ge=0, le=2)
    clear_empty_wrists: bool = False


class NeighborView(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    left: tuple[float, float, float, float, float, float]
    right: tuple[float, float, float, float, float, float]


class HeldCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_height_m: float
    value_floor: int = Field(default=65, ge=40, le=65)
    colors: dict[Literal["left", "right"], ColorAim] = Field(min_length=1, max_length=2)


RouteName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
RouteSteps = Annotated[list[RouteTurn | RouteWalk], Field(min_length=1, max_length=6)]


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: int
    objective: str = Field(min_length=1, max_length=1400)
    controls: list[str] = Field(min_length=1)
    preset: PresetName | None = None
    seconds: float = Field(default=15, ge=1, le=20)
    plans: int = Field(default=4, ge=1, le=8)
    before_grip: Check | None = None
    before_release: Check | None = None
    verify: Check | None = None
    stop_on_grip_change: bool = False
    queued_plan: Plan | None = None
    require_grips: set[Literal["left", "right"]] | None = None
    start_check: Check | None = None
    finish_check: Check | None = None
    require_stereo: bool = False
    arrival: PositionCheck | None = None
    routes: dict[RouteName, RouteSteps] = Field(default_factory=dict, max_length=6)
    aim_references: dict[Literal["left", "right"], str] = Field(default_factory=dict, max_length=2)
    settled_checks: list[Check] = Field(default_factory=list, max_length=6)
    settled_colors: dict[str, ColorAim] = Field(default_factory=dict, max_length=6)
    unchanged_colors: dict[str, ColorAim] = Field(default_factory=dict, max_length=6)
    held_check: HeldCheck | None = None
    neighbor_view: NeighborView | None = None
    settled_support_height_m: float | None = Field(default=None, allow_inf_nan=False)
    settle_seconds: float = Field(default=.75, ge=.25, le=3)
    settle_retry_seconds: float = Field(default=.9, ge=0, le=3)


def publish(file, data):
    staging = file.with_suffix(".tmp")
    staging.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(7):
        try:
            staging.replace(file)
            return
        except PermissionError:
            if attempt == 6:
                raise
            sleep(.015)


def disarm(inbox):
    for pending in inbox.glob("next-*.json"):
        pending.unlink(missing_ok=True)


def settled_color_evidence(colors, before, after):
    evidence = {}
    for name, color in colors.items():
        try:
            first = color.resolve_center(before)
        except ValueError as error:
            evidence[name] = {"stationary": False, "missing_in": "before", "reason": str(error)}
            continue
        try:
            last = color.resolve_center(after)
            delta = dist(first, last)
            evidence[name] = {"stationary": delta <= .022, "normalized_shift": delta,
                              "before_point": first, "point": last}
        except ValueError as error:
            evidence[name] = {"stationary": False, "before_point": first, "missing_in": "after", "reason": str(error)}
    return evidence


def stationary_color_surface(eye, context, color, anchor):
    hsv = cv2.cvtColor(np.asarray(eye.convert("RGB")), cv2.COLOR_RGB2HSV)
    mask = hue_mask(hsv, color.hue) & (hsv[:, :, 2] >= color.min_value)
    height, width = mask.shape
    left, top, right, bottom = color.roi
    samples = []
    for dx, dy in ((0., 0.), (.032, 0.), (-.032, 0.), (0., .027), (0., -.027),
                   (.032, .025), (-.032, .025)):
        x, y = anchor[0] + dx, anchor[1] + dy
        if not (left <= x < right and top <= y < bottom and mask[round(y * (height - 1)), round(x * (width - 1))]):
            continue
        try:
            point = measure_surface_in_context(context, x, y, color.radius_px)["world_xyz_m"]
        except ValueError:
            continue
        for prior in samples:
            if dist(prior, point) <= .08:
                return [(a + b) * .5 for a, b in zip(prior, point)]
        samples.append(point)
    raise ValueError("No consistent stereo depth around the stationary color region")


def settled_support_evidence(colors, frame, height_m, stationary=None):
    with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
        context = stereo_context(pair)
        eye = pair.crop((0, 0, pair.width // 2, pair.height))
        evidence = {}
        for name, color in colors.items():
            try:
                point = measure_color_surface(eye, context, color)["world_xyz_m"]
            except ValueError as error:
                anchor = stationary.get(name, {}).get("point") if stationary else None
                if not anchor or not isinstance(error, AmbiguousColorMatch):
                    evidence[name] = {"supported": False, "reason": str(error)}
                    continue
                try:
                    point = stationary_color_surface(eye, context, color, anchor)
                except ValueError as fallback_error:
                    evidence[name] = {"supported": False, "reason": str(fallback_error)}
                    continue
            evidence[name] = {"supported": -.045 <= point[1] - height_m <= .32,
                              "world_xyz_m": point}
        return evidence


def held_color_evidence(check, frame):
    with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
        context = stereo_context(pair)
        eye = pair.crop((0, 0, pair.width // 2, pair.height))
        evidence = {}
        grips = set(frame.state.get("closed_grips", []))
        for side, color in check.colors.items():
            palm = context[1]["palms"].get(side)
            if side not in grips or palm is None:
                evidence[side] = {"held": False, "reason": "A closed grip and measured palm are required"}
                continue
            try:
                point = measure_color_surface(eye, context, color, value_floor=check.value_floor)["world_xyz_m"]
            except ValueError as error:
                evidence[side] = {"held": False, "reason": str(error)}
                continue
            height = point[1] - check.source_height_m
            distance = dist(point, palm[0])
            evidence[side] = {"held": height >= .16 and distance <= .30, "world_xyz_m": point,
                              "height_above_source_m": height, "distance_from_palm_m": distance}
        return evidence


async def verify_arrival(session, target, trace, interrupted):
    def position(frame):
        with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
            if pair.size != (1280, 720):
                raise ValueError("A calibrated eye capture is required")
            left, right = np.hsplit(np.asarray(pair.convert("RGB")), 2)
        for eye, image, uncertainty_m in (("left", left, 0.), ("right", right, .15)):
            if hud := read_eye(image):
                pose = hud["pose"]
                return pose[[0, 2], 3].tolist(), pose[[0, 2], 2].tolist(), eye, uncertainty_m
        raise ValueError("Neither eye has readable position telemetry")

    after = -1
    parked = False
    nudges = 0
    previous_error = None
    while True:
        frame = await session.observe(view="stereo", after=after)
        try:
            xz, forward, eye, uncertainty = await asyncio.to_thread(position, frame)
        except ValueError as error:
            trace.record("arrival_unavailable", frame=frame.frame_id, reason=str(error))
            if (parked or not target.clear_empty_wrists or frame.state.get("closed_grips") != []
                    or str(error) != "Neither eye has readable position telemetry"):
                return "arrival_unavailable"
            if interrupted():
                return "interrupted"
            trace.record("arrival_tracking_recovery_started", frame=frame.frame_id)
            try:
                arguments = {**PARKED_WRIST}
                if session.fast_wrist_pair:
                    arguments["speed_m_s"] = 1.25
                event = await call(session.pilot, "trial_control", action="step", lease=session.lease,
                                   tool="wrist_pair", arguments=arguments)
            except TrialStopped:
                raise
            except RuntimeError as park_error:
                trace.record("arrival_tracking_recovery_finished", error=str(park_error))
                return "arrival_unavailable"
            trace.record("arrival_tracking_recovery_finished", result=event["result"])
            after, parked = frame.frame_id, True
            continue
        distance = dist(xz, target.xz)
        arrived = distance + uncertainty <= target.radius_m
        trace.record("arrival_evidence", frame=frame.frame_id, xz=xz, expected_xz=target.xz,
                     error_m=distance, eye=eye, uncertainty_m=uncertainty, arrived=arrived)
        if arrived:
            return "verified"
        heading_length = hypot(*forward)
        stalled = previous_error is not None and distance >= previous_error - .025
        if nudges >= target.max_nudges or stalled or uncertainty or distance > target.nudge_limit_m or heading_length < .5:
            return "arrival_mismatch"
        if interrupted():
            return "interrupted"

        dx, dz = (want - actual for want, actual in zip(target.xz, xz))
        fx, fz = (value / heading_length for value in forward)
        front = dx * fx + dz * fz
        right = dx * fz - dz * fx
        axes = [(front, "forward", "backward"), (right, "right", "left")]
        offset, positive, negative = max(axes, key=lambda axis: abs(axis[0]))
        step = {"kind": "walk", "direction": positive if offset > 0 else negative,
                "duration": max(.1, abs(offset) / 3.75)}
        trace.record("arrival_adjustment_started", frame=frame.frame_id, step=step)
        try:
            event = await call(session.pilot, "trial_control", action="step", lease=session.lease,
                               tool="route_motion", arguments={"steps": [step]})
        except TrialStopped:
            raise
        except RuntimeError as error:
            trace.record("arrival_adjustment_finished", error=str(error))
            return "arrival_mismatch"
        result = event["result"]
        trace.record("arrival_adjustment_finished", result=result)
        if any(result.get(key) is False for key in ("executed", "reached")):
            return "arrival_mismatch"
        after = frame.frame_id
        previous_error = distance
        nudges += 1


async def verify_settled(session, task, trace, interrupted):
    before = await session.observe()
    deadline = monotonic() + task.settle_seconds + task.settle_retry_seconds
    delay = task.settle_seconds
    while True:
        await asyncio.sleep(delay)
        if interrupted():
            return "interrupted"
        after = await session.observe(after=before.frame_id)
        if before.state.get("closed_grips") or after.state.get("closed_grips"):
            return "settle_grips_closed"
        evidence = await asyncio.to_thread(settled_color_evidence, task.settled_colors, before, after)
        trace.record("settled_color_evidence", before=before.frame_id, after=after.frame_id, colors=evidence)
        if all(item["stationary"] for item in evidence.values()):
            support = {}
            if task.settled_support_height_m is not None:
                stereo = await session.observe(view="stereo", after=after.frame_id)
                if stereo.state.get("closed_grips") != []:
                    return "settle_grips_closed"
                try:
                    support = await asyncio.to_thread(settled_support_evidence, task.settled_colors, stereo,
                                                      task.settled_support_height_m, evidence)
                except ValueError as error:
                    trace.record("settled_support_unavailable", frame=stereo.frame_id, reason=str(error))
                    return "settle_support_failed"
                trace.record("settled_support_evidence", frame=stereo.frame_id, colors=support)
                if not support or not all(item["supported"] for item in support.values()):
                    return "settle_support_failed"
            if task.settled_checks and not await session.check_frame(task.settled_checks, after):
                return "settle_check_failed"
            if support:
                session.scene_references.update({
                    name: {"hue": task.settled_colors[name].hue, "world_xyz_m": item["world_xyz_m"],
                           "support_height_m": task.settled_support_height_m, "frame": stereo.frame_id}
                    for name, item in support.items()
                })
            return "verified"
        delay = min(.5, deadline - monotonic())
        if delay < .25:
            return "settle_color_failed"
        before = after


async def await_objective(inbox, checkpoint, pilot, lease, deadline):
    keepalive = monotonic() + 20
    while monotonic() < deadline:
        if (inbox / "stop").exists():
            return None
        mailbox = inbox / "next.json"
        if not mailbox.exists():
            mailbox = inbox / f"next-{checkpoint}.json"
        if mailbox.exists():
            packet = mailbox.read_text(encoding="utf-8-sig")
            mailbox.unlink()
            try:
                objective = Objective.model_validate_json(packet)
                if objective.checkpoint != checkpoint:
                    raise ValueError("Objective refers to an old checkpoint")
                if any(tool not in CONTROLS for tool in objective.controls):
                    raise ValueError("Objective includes an unavailable control")
                if objective.queued_plan:
                    objective.queued_plan.check_shape(objective.controls)
                return objective
            except ValueError as error:
                publish(inbox / "rejected.json", {"reason": str(error)})
        if monotonic() >= keepalive:
            await call(pilot, "trial_control", action="check", lease=lease)
            keepalive = monotonic() + 20
        await asyncio.sleep(.12)


async def run(inbox, warm_task=None):
    async with streamable_http_client(PILOT) as (read, write, _):
        async with ClientSession(read, write) as pilot, httpx.AsyncClient(timeout=15) as http:
            await pilot.initialize()
            world = await call(pilot, "world_context")
            if world.get("access") != "invite" or not world.get("scene_ready"):
                raise ValueError("Stage the task in a scene-ready Invite instance")
            schemas = {tool.name: tool.inputSchema for tool in (await pilot.list_tools()).tools}
            trace = Trace("supervised-freeform")
            trace.record("start", world=world)
            lease = (await call(pilot, "trial_control", action="begin"))["lease"]
            idle = Session(pilot, http, trace, {"target": CONTROLS["target"]}, schemas, monotonic() + 600)
            status = "ready"
            neighbors = None
            scene_references = {}

            def make_session(task):
                routes = {name: [step.model_dump(mode="json") for step in steps]
                          for name, steps in task.routes.items()}
                batch_size = 1 if routes and task.controls == ["route_motion"] else 4 if task.preset else 2
                session = Session(pilot, http, trace, {tool: CONTROLS[tool] for tool in task.controls},
                                  schemas, idle.deadline, batch_size=batch_size,
                                  managed_pickups=bool(task.before_grip), preset=task.preset, routes=routes,
                                  aim_references=task.aim_references, scene_references=scene_references)
                session.lease = lease
                return session

            try:
                if warm_task is not None and not (inbox / "stop").exists():
                    deadline = monotonic() + 4
                    trace.record("cache_warm_started", preset=warm_task.preset, checkpoint=warm_task.checkpoint)
                    try:
                        frame = await idle.observe(deadline=deadline)
                        await make_session(warm_task).plan(
                            "Observation-only cache warm-up. Return done without any actions.",
                            frame, [], deadline=deadline,
                        )
                    except (RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as error:
                        trace.record("cache_warm_finished", outcome="unavailable", reason=str(error))
                    else:
                        trace.record("cache_warm_finished", outcome="ready")
                for checkpoint in range(64):
                    if monotonic() >= idle.deadline:
                        break
                    frame = await idle.observe()
                    publish(inbox / "checkpoint.json", {
                        "checkpoint": checkpoint, "status": status, "world": world,
                        "frame": str(trace.folder / f"frame-{frame.frame_id}.jpg"),
                        "state": frame.state, "trace": str(trace.folder / "events.jsonl"),
                        "neighbor_motion": neighbors,
                        "scene_references": scene_references,
                    })
                    task = await await_objective(inbox, checkpoint, pilot, lease, idle.deadline)
                    if task is None:
                        break
                    neighbors = None
                    trace.record("supervisor_objective", task=task.model_dump(mode="json"))
                    session = make_session(task)
                    if task.require_grips is not None or task.start_check or task.unchanged_colors:
                        frame = await session.observe()
                        grips = set(frame.state.get("closed_grips", []))
                        if task.require_grips is not None and grips != task.require_grips:
                            status = "grip_precondition_failed"
                            trace.record("supervisor_result", checkpoint=checkpoint, status=status, grips=sorted(grips))
                            disarm(inbox)
                            continue
                        if task.start_check and await session.wait_for(task.start_check, frame) is None:
                            status = "start_check_expired"
                            trace.record("supervisor_result", checkpoint=checkpoint, status=status)
                            disarm(inbox)
                            continue
                    def interrupted():
                        return (inbox / "pause").exists() or (inbox / "stop").exists()

                    async def record_pickup_stage(tool, observed):
                        evidence = await asyncio.to_thread(settled_color_evidence, task.unchanged_colors, frame, observed)
                        trace.record("neighbor_motion_stage", tool=tool, before=frame.frame_id,
                                     after=observed.frame_id, colors=evidence)

                    window_end = min(idle.deadline, monotonic() + task.seconds)
                    status = await session.run(
                        task.objective, before_grip=task.before_grip, before_release=task.before_release,
                        verify=task.verify, recover=True, max_plans=task.plans,
                        window_end=window_end, stop_on_grip_change=task.stop_on_grip_change,
                        queued_plan=task.queued_plan,
                        interrupted=interrupted,
                        require_stereo=task.require_stereo,
                        pickup_stage=record_pickup_stage if task.unchanged_colors else None,
                        external_verification=bool(task.arrival or task.finish_check or task.settled_checks
                                                   or task.settled_colors or task.unchanged_colors or task.held_check),
                    )
                    pickup_completed = status in ("plan_limit", "done", "grip_changed", "verified")
                    pickup_uncertain = (
                        status == "preset_unverified" and (task.held_check is not None
                                                           or task.neighbor_view is not None and task.verify is not None)
                    )
                    if task.unchanged_colors and (pickup_completed or pickup_uncertain):
                        after = await session.observe(after=frame.frame_id)
                        evidence = await asyncio.to_thread(settled_color_evidence, task.unchanged_colors, frame, after)
                        neighbors = {"before": frame.frame_id, "after": after.frame_id, "colors": evidence}
                        trace.record("neighbor_motion_evidence", **neighbors)
                        if (task.neighbor_view and set(after.state.get("closed_grips", [])) == {"left", "right"}
                                and (pickup_uncertain or not all(item["stationary"] for item in evidence.values()))
                                and not interrupted() and monotonic() < window_end):
                            trace.record("neighbor_view_recovery_started", frame=after.frame_id)
                            if task.verify:
                                status = "preset_unverified"
                            try:
                                result = await call(pilot, "trial_control", action="step", lease=lease, tool="wrist_pair",
                                                    arguments=task.neighbor_view.model_dump(mode="json"))
                                after = await session.observe(after=after.frame_id, deadline=window_end)
                            except TrialStopped:
                                raise
                            except (RuntimeError, TimeoutError) as error:
                                trace.record("neighbor_view_recovery_finished", reason=str(error) or type(error).__name__)
                            else:
                                trace.record("neighbor_view_recovery_finished", result=result["result"])
                                evidence = await asyncio.to_thread(settled_color_evidence, task.unchanged_colors, frame, after)
                                neighbors = {"before": frame.frame_id, "after": after.frame_id, "colors": evidence}
                                trace.record("neighbor_motion_evidence", **neighbors)
                                remaining = window_end - monotonic()
                                if (task.verify and all(item["stationary"] for item in evidence.values())
                                        and set(after.state.get("closed_grips", [])) == {"left", "right"}
                                        and not interrupted() and remaining > 0):
                                    check = task.verify.model_copy(update={"seconds": min(task.verify.seconds, remaining)})
                                    if await session.wait_for(check, after):
                                        status = "verified"
                        if not all(item["stationary"] for item in evidence.values()):
                            status = "neighbor_moved_or_occluded"
                    if task.held_check and status in ("plan_limit", "done", "grip_changed", "verified", "preset_unverified"):
                        status = "held_check_failed"
                        try:
                            held = await session.observe(view="stereo", deadline=window_end)
                            evidence = await asyncio.to_thread(held_color_evidence, task.held_check, held)
                        except (ValueError, TimeoutError) as error:
                            trace.record("held_color_unavailable", reason=str(error) or type(error).__name__)
                        else:
                            trace.record("held_color_evidence", frame=held.frame_id, colors=evidence)
                            if all(item["held"] for item in evidence.values()):
                                status = "verified"
                    if task.arrival and status in ("plan_limit", "done", "grip_changed", "verified"):
                        status = await verify_arrival(session, task.arrival, trace, interrupted)
                    if task.finish_check and status in ("plan_limit", "done", "grip_changed", "verified"):
                        frame = await session.observe()
                        status = "verified" if await session.wait_for(task.finish_check, frame) else "finish_check_expired"
                    if (task.settled_checks or task.settled_colors) and status in (
                        "plan_limit", "done", "grip_changed", "verified"
                    ):
                        status = await verify_settled(session, task, trace, interrupted)
                    (inbox / "pause").unlink(missing_ok=True)
                    trace.record("supervisor_result", checkpoint=checkpoint, status=status)
                    if status not in ("plan_limit", "done", "grip_changed", "verified"):
                        # Disarm future work until the supervisor explicitly submits fresh packets.
                        disarm(inbox)
            except Exception as error:
                status = "supervisor_error"
                trace.record("supervisor_error", reason=str(error))
                raise
            finally:
                await release_lease(pilot, lease, trace)
                trace.record("finished", outcome=status)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inbox", type=Path, help="New directory for checkpoints and objective packets")
    cli.add_argument("--queue", type=Path, help="Optional JSON list of objectives to arm before the first action")
    cli.add_argument("--warm-cache", action="store_true",
                     help="Prime Omni with the first model-planned preset in the queue before any motion")
    args = cli.parse_args()
    tasks = []
    if args.queue:
        tasks = TypeAdapter(list[Objective]).validate_json(args.queue.read_text(encoding="utf-8-sig"))
        if not 1 <= len(tasks) <= 64 or [task.checkpoint for task in tasks] != list(range(len(tasks))):
            cli.error("Queue checkpoints must start at zero and run consecutively")
    warm_task = next((task for task in tasks if task.preset and task.queued_plan is None), None) if args.warm_cache else None
    if args.warm_cache and (warm_task is None or any(tool not in CONTROLS for tool in warm_task.controls)):
        cli.error("Cache warming needs a queued model-planned preset with supported controls")
    args.inbox.mkdir(parents=True, exist_ok=False)
    for task in tasks:
        publish(args.inbox / f"next-{task.checkpoint}.json", task.model_dump(mode="json"))
    asyncio.run(run(args.inbox, warm_task))
