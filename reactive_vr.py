"""Run bounded VRChat microtasks with Omni actions and d1 verification."""

import argparse
import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass
import json
from io import BytesIO
from math import atan2, cos, pi, sin
from time import monotonic
from typing import Literal

import httpx
import numpy as np
from PIL import Image
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation

from vr_benchmark import Trace, TrialStopped, call, headset_yaw, observed_wrists
from menu_workflows import AUDIO, CAMERA, LAUNCH, locate
from pickup_vision import ColorTarget, color_aim_point, color_center, colored_pickup_outline
from vr_presets import PRESETS
from spatial_geometry import K, measure_surface_in_context, palm_step, stereo_context

PILOT = "http://127.0.0.1:29746/mcp"
OBSERVER = "http://127.0.0.1:8098/v1/chat/completions"
CHECKER = "http://127.0.0.1:8099/v1/systemone"
PARKED_WRIST = {"left": [-.34, 1.03, -.12, 0, 0, 0], "right": [.34, 1.03, -.12, 0, 0, 0]}
CONTROLS = {
    "menu": 'arguments {"action":"toggle"} or {"action":"click"} after verified hover. For raw aim, {"action":"aim","x":0.5,"y":0.5}.',
    "present_quick_menu": "Present and verify Launch Pad.",
    "aim_menu_beam": 'Aim at Audio with {} or name a mapped menu control with {"control":"camera_tab"} or {"control":"photo_camera"}.',
    "aim_at": "Align the beam at a normalized visual point and return visual verification.",
    "aim_surface": 'Use stereo depth at an image point in 0–1000 integer coordinates. Select side left or right, then verify the highlight before gripping.',
    "aim_surface_pair": 'Aim two empty hands simultaneously at distinct image points. Supply left and right [x,y] lists scaled from 0.0 to 1.0.',
    "aim_color": 'Aim at a unique color inside roi [left,top,right,bottom] on the current frame. Supply hue 0–179 and side. Inspect the highlight before gripping.',
    "aim_color_pair": 'Aim two empty hands at distinctly colored regions from one observation. Give left and right objects with hue and normalized roi, then verify both highlights.',
    "target": 'For a head turn use {"kind":"view","values":[yaw_degrees,pitch_degrees]}. Positive yaw turns right. For a wrist use {"kind":"wrist","side":"right","values":[x,y,z,yaw,pitch,roll]}.',
    "grip": 'Close only after aiming at an outlined pickup. Use {"side":"right","closed":true} to close or {"side":"right","closed":false} to release above a support.',
    "grip_pair": 'Set both grips with {"left":true,"right":true} for two verified highlighted pickups or {"left":false,"right":false} above the two landing spots.',
    "wrist_pair": 'Move both wrists simultaneously, supplying left and right lists of [x,y,z,yaw,pitch,roll] in metres and degrees. Grip states stay unchanged.',
    "lift_pair": "Lift two held objects vertically without changing wrist angles or grip. Supply up_m in metres.",
    "translate_wrist": 'Move the current holding wrist without changing grip. Example {"side":"right","up_m":0.12}.',
    "body_action": 'For a walk use {"action":"walk","direction":"forward","values":[seconds]}.',
    "look": "Turn the view in place by the supplied relative yaw and pitch in degrees. Positive yaw is right and negative pitch is down. The grips stay unchanged.",
    "scan_view": "Look left, right, up, or down in place by a small angle. Both grips stay unchanged.",
    "park_free_wrist": 'Retract one empty hand to clear the headset view. Supply side "left" or "right".',
    "align_held_surface": "Make one bounded wrist correction to align a held colored object above a support point. The grip stays closed.",
    "upright_held_object": "Rotate a held object upright without changing grip. Supply left-eye coordinates for the top and base of the same object.",
    "route_motion": 'Queue verified short route segments. Steps are {"kind":"turn","yaw":90,"pitch":0} or {"kind":"walk","direction":"forward","duration":1.2}. Verify the endpoint in the next frame.',
    "camera": 'Connect to the already open VRChat camera stream with {"enabled":true}.',
}


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=8, max_length=240)
    expected: bool = True
    confidence: float = Field(default=.8, ge=.65, le=1.)
    seconds: float = Field(default=2., gt=0, le=12)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict = Field(default_factory=dict)
    when: Check | None = None
    checks: list[Check] = Field(default_factory=list, max_length=6)


class Look(BaseModel):
    model_config = ConfigDict(extra="forbid")
    yaw: float = Field(ge=-40, le=40)
    pitch: float = Field(default=0, ge=-35, le=35)


class ScanView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["left", "right", "up", "down"]
    degrees: float = Field(default=15, ge=5, le=35)

    def turn(self):
        yaw_sign, pitch_sign = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}[self.direction]
        return {"yaw": yaw_sign * self.degrees, "pitch": pitch_sign * self.degrees}


class PairLift(BaseModel):
    model_config = ConfigDict(extra="forbid")
    up_m: float = Field(gt=0, le=.25)


class ColorAim(ColorTarget):
    side: Literal["left", "right"] = "right"

    def resolve_center(self, frame):
        with Image.open(BytesIO(base64.b64decode(frame.data))) as image:
            return color_center(image, self.roi, self.hue, self.min_value)

    def resolve_stereo(self, frame):
        points, radius = stereo_color_points(frame, self.radius_px, self)
        return {**points[0], "radius_px": radius}

    def point(self, image, validate=None, *, near_base=False, value_floor=65):
        x, y = color_aim_point(image, self.roi, self.hue, validate, min_value=self.min_value,
                               near_base=near_base, value_floor=value_floor)
        return dict(x=x, y=y, side=self.side, radius_px=self.radius_px)


class ColorAimPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: ColorAim
    right: ColorAim

    def resolve(self, frame):
        radius = min(self.left.radius_px, self.right.radius_px)
        (left, right), radius = stereo_color_points(frame, radius, self.left, self.right)
        if (left["x"] - right["x"]) ** 2 + (left["y"] - right["y"]) ** 2 < .0025:
            raise ValueError("Paired targets must resolve to separate image regions")
        return {"left": [left["x"], left["y"]], "right": [right["x"], right["y"]],
                "radius_px": radius}


def project_color_region(context, point, roi):
    images, _, pose, _ = context
    height, width = images[0].shape
    camera_point = np.linalg.solve(pose[:3, :3], np.asarray(point) - pose[:3, 3])
    if camera_point[2] <= .05:
        raise ValueError("The last verified prop position is behind the current view")
    x, y = (K @ camera_point)[:2] / camera_point[2] / [width - 1, height - 1]
    if not (0 < x < 1 and 0 < y < 1):
        raise ValueError("The last verified prop position is outside the current view")
    half_width = max(.12, min(.20, (roi[2] - roi[0]) * .5))
    half_height = max(.14, min(.20, (roi[3] - roi[1]) * .5))
    return [max(0., x - half_width), max(0., y - half_height),
            min(1., x + half_width), min(1., y + half_height)]


class HeldAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    color: ColorAim
    support: tuple[float, float] | None = None
    support_reference: str | None = Field(default=None, min_length=1, max_length=80)
    height_m: float = Field(default=.14, ge=.06, le=.30)
    support_radius_px: int = Field(default=25, ge=10, le=32)
    support_height_m: float | None = None
    support_witness: tuple[float, float] | None = None
    support_clearance_m: float = Field(default=0., ge=0., le=.20)
    tolerance_m: float | None = Field(default=None, ge=.04, le=.14)
    near_base: bool = False
    premeasure_support: bool = False
    allow_reach_sidestep: bool = False

    @model_validator(mode="after")
    def check_support_target(self):
        if (self.support is None) == (self.support_reference is None):
            raise ValueError("Supply one image support point or one verified resting location")
        return self

    def has_support_clearance(self, context, point):
        if not self.support_clearance_m:
            return True
        images, _, pose, _ = context
        height, width = images[0].shape
        point = np.asarray(point)
        ring = []
        for sector in (0, 2, 4, 6, 1, 3, 5, 7):
            angle = sector * pi / 4
            direction = np.array([cos(angle), 0., sin(angle)])
            expected = point + 1.8 * self.support_clearance_m * direction
            camera = np.linalg.solve(pose[:3, :3], expected - pose[:3, 3])
            if camera[2] <= 0.:
                continue
            pixel = K @ camera / camera[2]
            try:
                probe = measure_surface_in_context(context, pixel[0] / (width - 1), pixel[1] / (height - 1), 22)
            except ValueError:
                continue
            normal = probe["surface_normal"]
            measured = np.asarray(probe["world_xyz_m"])
            if normal is not None and normal[1] >= .82 and np.linalg.norm(measured - expected) <= .05:
                ring.append((measured - point)[[0, 2]])
            if sector in (6, 7) and len(ring) >= 3:
                ring.sort(key=lambda offset: atan2(offset[1], offset[0]))
                edges = zip(ring, ring[1:] + ring[:1])
                # The landing point must clear every edge of the measured support polygon.
                if all(a[0] * b[1] - a[1] * b[0] >= self.support_clearance_m * np.linalg.norm(b - a)
                       for a, b in edges):
                    return True
        return False

    def measure_support(self, context):
        if self.support is None:
            raise ValueError("Supply an image point or a verified resting location for the landing")
        x, y = self.support
        if not (0 <= x < 1 and 0 <= y < 1):
            raise ValueError("Select a normalized support point in the visible image")
        for dx, dy in ((0, 0), (-.04, 0), (.04, 0), (0, .04), (0, -.04), (-.05, -.01), (-.06, .05)):
            try:
                surface = measure_surface_in_context(context, x + dx, y + dy, self.support_radius_px)
            except ValueError:
                continue
            normal = surface["surface_normal"]
            if normal is None or normal[1] < .82:
                continue
            if self.support_height_m is not None and abs(surface["world_xyz_m"][1] - self.support_height_m) > .08:
                continue
            if self.has_support_clearance(context, surface["world_xyz_m"]):
                return surface
        if self.support_witness is None or self.support_height_m is None:
            raise ValueError("No flat stereo-supported landing patch with the requested clearance")
        witness = measure_surface_in_context(context, *self.support_witness, self.support_radius_px)
        normal = witness["surface_normal"]
        reference = np.asarray(witness["world_xyz_m"])
        if normal is None or normal[1] < .82 or abs(reference[1] - self.support_height_m) > .08:
            raise ValueError("The support witness does not match the expected tabletop or shelf")
        images, _, pose, _ = context
        height, width = images[0].shape
        ray = pose[:3, :3] @ np.linalg.solve(K, [x * (width - 1), y * (height - 1), 1])
        if ray[1] >= -.12:
            raise ValueError("The landing ray does not point down toward the support")
        distance = (reference[1] - pose[1, 3]) / ray[1]
        point = pose[:3, 3] + ray * distance
        if not .2 <= distance <= 2 or np.linalg.norm((point - reference)[[0, 2]]) > .35:
            raise ValueError("The landing point is too far from the verified support witness")
        if not self.has_support_clearance(context, point):
            raise ValueError("The projected landing point lacks measured support clearance")
        return {"world_xyz_m": point.tolist(), "source": "witness_plane",
                "witness_world_xyz_m": reference.tolist()}

    def resolve(self, frame, *, measured_support=None):
        with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
            context = stereo_context(pair)
            eye = pair.crop((0, 0, pair.width // 2, pair.height))
            support = measured_support if measured_support is not None else self.measure_support(context)
            visible = measure_color_surface(eye, context, self.color, near_base=self.near_base)
            side = self.color.side
            palm = context[1]["palms"].get(side)
            if palm is None:
                raise ValueError("The held wrist has no visible pose marker")
            goal = palm[0] + support["world_xyz_m"] - np.asarray(visible["world_xyz_m"])
            goal[1] += self.height_m
            error, _, wrist = palm_step(pair, frame.state.get("commanded_poses"), side, goal)
            if error > .48:
                raise ValueError("The proposed landing is too far for a fine wrist correction")
            evidence = {"error_m": error, "support_world_xyz_m": support["world_xyz_m"],
                        "support_source": support.get("source", "stereo_patch"),
                        "support_clearance_m": self.support_clearance_m,
                        "object_world_xyz_m": visible["world_xyz_m"]}
            if "witness_world_xyz_m" in support:
                evidence["witness_world_xyz_m"] = support["witness_world_xyz_m"]
            if "reference_frame" in support:
                evidence["reference_frame"] = support["reference_frame"]
            return {"kind": "wrist", "side": side, "values": wrist}, evidence

    def support_from_frame(self, frame):
        with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
            return self.measure_support(stereo_context(pair))


def measure_color_surface(eye, context, color, *, near_base=False, value_floor=65):
    surface = None

    def measure(point):
        nonlocal surface
        try:
            surface = measure_surface_in_context(context, *point, color.radius_px)
        except ValueError:
            return False
        return True

    color.point(eye, measure, near_base=near_base, value_floor=value_floor)
    return surface


class HeldUpright(BaseModel):
    model_config = ConfigDict(extra="forbid")
    side: Literal["left", "right"]
    tip: tuple[float, float]
    base: tuple[float, float]
    radius_px: int = Field(default=17, ge=10, le=25)

    def resolve(self, frame):
        with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
            context = stereo_context(pair)
        tip = measure_surface_in_context(context, *self.tip, self.radius_px)["world_xyz_m"]
        base = measure_surface_in_context(context, *self.base, self.radius_px)["world_xyz_m"]
        axis = np.asarray(tip) - base
        if not .06 <= np.linalg.norm(axis) <= .45:
            raise ValueError("Select two distinct ends of the held object")
        forward = context[2][[0, 2], 2]
        if np.linalg.norm(forward) < .25:
            raise ValueError("Look closer to level before aligning the held object")
        forward = forward / np.linalg.norm(forward)
        right = np.array([forward[1], -forward[0]])
        horizontal = axis[[0, 2]]
        local_axis = [right @ horizontal, axis[1], -forward @ horizontal]
        correction, _ = Rotation.align_vectors([[0., 1., 0.]], [local_axis])
        tilt = float(np.degrees(correction.magnitude()))
        if tilt > 120:
            raise ValueError("The object is too far inverted for a single correction, or the selected ends were reversed")
        wrist = (observed_wrists(frame.state) or {}).get(self.side)
        if wrist is None:
            raise ValueError("The current wrist pose is unavailable")
        yaw, pitch, roll = (correction * Rotation.from_euler(
            "YXZ", [-wrist[3], wrist[4], wrist[5]], degrees=True)).as_euler("YXZ", degrees=True)
        return {"kind": "wrist", "side": self.side, "values": [*wrist[:3], -yaw, pitch, roll]}, {
            "tip_world_xyz_m": tip, "base_world_xyz_m": base, "tilt_deg": tilt,
        }


def stereo_color_points(frame, radius, *colors):
    with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
        context = stereo_context(pair)
        eye = pair.crop((0, 0, pair.width // 2, pair.height))

        def has_depth(point):
            try:
                measure_surface_in_context(context, *point, candidate_radius)
            except ValueError:
                return False
            return True

        for candidate_radius in (radius, 10) if radius > 10 else (radius,):
            points = []
            for color in colors:
                try:
                    points.append(color.point(eye, has_depth))
                except ValueError as error:
                    failed_hue, failure = color.hue, error
                    break
            else:
                return points, candidate_radius
        raise ValueError(f"Color hue {failed_hue}: {failure}") from failure


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(default_factory=list, max_length=6)
    when: Check | None = None
    child: str | None = Field(default=None, min_length=8, max_length=260)
    verify: Check | None = None
    status: Literal["continue", "done", "blocked"] = "continue"

    def check_shape(self, allowed):
        if any(action.tool not in allowed for action in self.actions):
            raise ValueError("Omni selected a control outside this microtask's allowance")
        if self.status != "continue" and (self.actions or self.when or self.child or self.verify):
            raise ValueError("A terminal plan must have no pending work")
        if self.status == "continue" and not (self.actions or self.child):
            raise ValueError("A continuing plan needs an action or a child microtask")
        if self.child and (not self.when or self.actions or self.verify):
            raise ValueError("A child needs a trigger and starts with its own empty action queue")


@dataclass
class Frame:
    data: str
    state: dict

    @property
    def frame_id(self):
        return self.state["frame_id"]

    @property
    def url(self):
        return "data:image/jpeg;base64," + self.data

    def menu_panel(self):
        image = Image.open(BytesIO(base64.b64decode(self.data)))
        match = locate(image, (*LAUNCH, *AUDIO, *CAMERA))
        return match.panel.name if match else None


def inline_tool_schema(schema):
    """Resolve tool-local definitions before nesting them in the planner's schema."""
    definitions = schema.get("$defs", {})

    def resolve(value):
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            ref = value["$ref"]
            if not ref.startswith("#/$defs/") or ref[8:] not in definitions:
                raise ValueError(f"Tool schema contains an unsupported reference: {ref}")
            return resolve({**definitions[ref[8:]], **{key: item for key, item in value.items() if key != "$ref"}})
        return {key: resolve(item) for key, item in value.items() if key != "$defs"}

    return resolve(schema)


class Session:
    def __init__(self, pilot, http, trace, controls, schemas, deadline, batch_size=6, managed_pickups=False,
                 preset=None, routes=None, aim_references=None, scene_references=None):
        self.pilot = pilot
        self.http = http
        self.trace = trace
        self.controls = controls
        self.deadline = deadline
        self.lease = ""
        self.managed_pickups = managed_pickups
        self.preset = preset
        self.routes = routes or {}
        self.aim_references = aim_references or {}
        self.scene_references = scene_references if scene_references is not None else {}
        if self.routes and "route_motion" in self.controls:
            self.controls = {**controls, "route_motion": "Use a supplied named route as {\"route\": \"name\"}."}
        self.native_color_aim = "aim_colored_surfaces" in schemas
        self.fast_wrist_pair = "speed_m_s" in schemas.get("wrist_pair", {}).get("properties", {})
        plan_schema = Plan.model_json_schema()
        plan_schema["$defs"]["Check"]["properties"]["confidence"] = {"enum": [.75, .8, .85, .9]}
        plan_schema["$defs"]["Check"]["properties"]["seconds"] = {"enum": [.5, 1, 2, 4, 8, 12]}
        argument_schemas = {
            **schemas,
            "look": Look.model_json_schema(),
            "scan_view": ScanView.model_json_schema(),
            "park_free_wrist": {"type": "object", "required": ["side"], "additionalProperties": False,
                                "properties": {"side": {"enum": ["left", "right"]}}},
            "lift_pair": PairLift.model_json_schema(),
            "align_held_surface": HeldAlignment.model_json_schema(),
            "upright_held_object": HeldUpright.model_json_schema(),
            "aim_color": ColorAim.model_json_schema(),
            "aim_color_pair": ColorAimPair.model_json_schema(),
            "aim_menu_beam": {"type": "object", "additionalProperties": False,
                              "properties": {"control": {"enum": ["audio", "camera_tab", "photo_camera"]}}},
            "aim_surface": {
                "type": "object", "additionalProperties": False, "required": ["x", "y"],
                "properties": {
                    **{axis: {"type": "integer", "minimum": 0, "maximum": 999} for axis in ("x", "y")},
                    "side": {"enum": ["left", "right"]},
                    "radius_px": {"type": "integer", "minimum": 6, "maximum": 32},
                },
            },
        }
        if self.routes:
            argument_schemas["route_motion"] = {
                "type": "object", "required": ["route"], "additionalProperties": False,
                "properties": {"route": {"enum": list(self.routes)}},
            }
        properties = plan_schema["properties"]
        properties["actions"]["minItems"] = 1
        properties["actions"]["maxItems"] = batch_size
        properties["actions"]["items"] = {"anyOf": [
            {"type": "object", "required": ["tool", "arguments"], "additionalProperties": False,
             "properties": {
                 "tool": {"const": tool}, "arguments": inline_tool_schema(argument_schemas[tool]),
                 **({"checks": {"type": "array", "minItems": 1, "maxItems": 3,
                                "items": {"$ref": "#/$defs/Check"}}} if preset else
                    {} if managed_pickups and tool in ("aim_color", "aim_color_pair", "aim_surface", "aim_surface_pair", "aim_at", "grip") else
                    {"when": {"$ref": "#/$defs/Check"}}),
             }}
            for tool in controls
        ]}
        self.plan_schema = {"$defs": plan_schema["$defs"], "anyOf": [
            {"type": "object", "additionalProperties": False, "required": ["actions"],
             "properties": {key: properties[key] for key in ("actions", "when", "verify")}},
            {"type": "object", "additionalProperties": False, "required": ["when", "child"],
             "properties": {"when": {"$ref": "#/$defs/Check"}, "child": {"type": "string"}}},
            {"type": "object", "additionalProperties": False, "required": ["status"],
             "properties": {"status": {"enum": ["done", "blocked"]}}},
        ]}
        if preset:
            del self.plan_schema["anyOf"][1]

    async def observe(self, after=-1, deadline=None, view="headset"):
        deadline = min(self.deadline, deadline or self.deadline)
        while monotonic() < deadline:
            async with asyncio.timeout(max(0, deadline - monotonic())):
                response = await self.pilot.call_tool("observation_frame", {"view": view})
            if not response.isError:
                state, image = response.content
                frame = Frame(image.data, json.loads(state.text))
                if frame.frame_id > after:
                    suffix = "" if view == "headset" else f"-{view}"
                    file = f"frame-{frame.frame_id}{suffix}.jpg"
                    (self.trace.folder / file).write_bytes(base64.b64decode(frame.data))
                    self.trace.record("frame", frame=frame.frame_id, image=file, view=view, yaw_deg=headset_yaw(frame.state),
                                      commanded_poses=frame.state.get("commanded_poses"),
                                      closed_grips=frame.state.get("closed_grips"))
                    return frame
            elif "No current headset frame" not in response.content[0].text:
                raise RuntimeError(response.content[0].text)
            await asyncio.sleep(.06)
        raise TimeoutError("No new observation before the deadline")

    async def request(self, url, payload, deadline=None):
        remaining = min(self.deadline, deadline or self.deadline) - monotonic()
        async with asyncio.timeout(max(0, remaining)):
            try:
                result = await self.http.post(url, json=payload)
            except httpx.TransportError as error:
                self.trace.record("model_transport_retry", url=url, reason=str(error))
                result = await self.http.post(url, json=payload)
            result.raise_for_status()
            return result.json()

    async def plan(self, objective, frame, history, *, exclude=(), deadline=None):
        controls = {tool: info for tool, info in self.controls.items() if tool not in exclude}
        available = "\n".join(f"{tool}: {info}" for tool, info in controls.items())
        schema = self.plan_schema
        if exclude:
            schema = deepcopy(schema)
            items = schema["anyOf"][0]["properties"]["actions"]["items"]
            items["anyOf"] = [item for item in items["anyOf"] if item["properties"]["tool"]["const"] in controls]
        if self.preset:
            instruction = (
                f"Control a VRChat avatar. Preset {self.preset}. {PRESETS[self.preset][0]} "
                "Return consecutive tool calls in actions with explicit arguments. Put observable yes/no facts in "
                "checks on the action they gate. Each check names a visible object or event. Use verify for the "
                "final visible result. Use done only when already complete or blocked when there is no viable action."
            )
        else:
            instruction = (
                "Control a VRChat avatar. Select a short batch of allowed actions with explicit arguments. "
                "Actions run in listed order. Use a yes/no question about one visible fact to wait before "
                "running them, or put the fresh session's objective in child alongside that when check. "
                "Put when on a single action when its check depends on an earlier action's result. "
                "For example, attach when to grip to ask whether the intended object is outlined. "
                "Omit confidence to use 0.8 unless the task gives a different threshold. "
                "Use verify only for a fact visible in the frame after the actions. Never ask what to do. "
                "When actively searching for a missing object, use a nonzero turn or short move to inspect a new "
                "area, then check the fresh view. After a sighting check expires, change the view before "
                "asking it again. A view change used for searching must not depend on already seeing the object. "
                "Check earlier action results before choosing another movement. When actions and their order "
                "are known, batch them. The harness halts a batch on failed alignment. "
                "Use done after the objective is visibly met, or blocked when you cannot make progress."
            )
            if self.managed_pickups:
                instruction += " The runner already gates gripping. Aim immediately and let the runner check the outline before closing."
        route_context = f"\nAvailable named routes: {', '.join(self.routes)}" if self.routes else ""
        response = await self.request(OBSERVER, {
            "model": "vision", "temperature": 0, "max_tokens": 340 if self.preset == "place_and_verify" else 260,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
            "messages": [
                {"role": "system", "content": f"{instruction}\nAvailable controls:\n{available}"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": frame.url}},
                    {"type": "text", "text": (
                        f"Objective: {objective}\nClosed grips: {frame.state.get('closed_grips', [])}"
                        f"{route_context}"
                        f"\nEarlier results in this session: {history[-4:]}"
                    )},
                ]},
            ],
        }, deadline)
        content = response["choices"][0]["message"]["content"]
        try:
            plan = Plan.model_validate_json(content)
            plan.check_shape(controls)
        except ValueError:
            self.trace.record("invalid_plan", raw=content, frame=frame.frame_id)
            raise
        self.trace.record("omni_plan", plan=plan.model_dump(), frame=frame.frame_id, preset=self.preset,
                          usage=response.get("usage"), timings=response.get("timings"))
        return plan

    async def wait_for(self, check, frame):
        expires = min(self.deadline, monotonic() + check.seconds)
        try:
            while monotonic() < expires:
                started = monotonic()
                response = await self.request(CHECKER, {
                    "state": PRESETS[self.preset][1] if self.preset else "Answer this visible fact about the current VRChat frame.",
                    "files": [frame.url],
                    "questions": {"trigger": {"type": "noul", "instructions": check.question}},
                }, expires)
                confidence = response["answers"]["trigger"]["noul"]
                confidence = confidence if check.expected else 1-confidence
                met = confidence >= check.confidence and monotonic() < expires
                self.trace.record("d1_check", question=check.question, frame=frame.frame_id,
                                  confidence=confidence, fired=met, request_seconds=monotonic()-started,
                                  usage=response.get("usage"), timings=response.get("timings"))
                if met:
                    return frame
                if expires - monotonic() < .15:
                    break
                frame = await self.observe(after=frame.frame_id)
        except TimeoutError:
            pass
        self.trace.record("trigger_expired", question=check.question)
        return None

    async def check_frame(self, checks, frame, deadline=None):
        """Evaluate independent facts in one d1 request against a single frame."""
        self.trace.record("verification_started", frame=frame.frame_id, questions=len(checks))
        started = monotonic()
        response = await self.request(CHECKER, {
            "state": PRESETS[self.preset][1] if self.preset else "Answer each visible fact independently about this VRChat frame.",
            "files": [frame.url],
            "questions": {str(i): {"type": "noul", "instructions": check.question}
                          for i, check in enumerate(checks)},
        }, deadline)
        passed = True
        for i, check in enumerate(checks):
            confidence = response["answers"][str(i)]["noul"]
            confidence = confidence if check.expected else 1 - confidence
            fired = confidence >= check.confidence
            self.trace.record("d1_check", frame=frame.frame_id, question=check.question,
                              confidence=confidence, fired=fired)
            passed &= fired
        self.trace.record("verification_finished", frame=frame.frame_id, passed=passed,
                          request_seconds=monotonic()-started, usage=response.get("usage"),
                          timings=response.get("timings"))
        return passed

    async def camera_frame(self):
        while monotonic() < self.deadline:
            async with asyncio.timeout(max(0, self.deadline - monotonic())):
                reply = await self.pilot.call_tool("observe", {"view": "camera"})
            if not reply.isError:
                data = reply.content[0].data
                (self.trace.folder / "camera-live.jpg").write_bytes(base64.b64decode(data))
                self.trace.record("camera_observation", image="camera-live.jpg")
                return True
            await asyncio.sleep(.06)
        return False

    async def hold_frames(self, frame, seconds):
        samples = [frame]
        async with asyncio.timeout(max(0, self.deadline - monotonic())):
            for _ in range(2):
                await asyncio.sleep(seconds / 2)
                frame = await self.observe(after=frame.frame_id)
                samples.append(frame)
        grips = samples[0].state["closed_grips"]
        return samples if all(sample.state.get("closed_grips") == grips for sample in samples) else []

    async def align_for_release(self, target, action, after, window_end, interrupted, *, support=None):
        deadline = min(self.deadline, window_end) if window_end is not None else self.deadline
        corrections = 0
        sidestepped = False
        for attempt in range(4 if target.allow_reach_sidestep else 3):
            source = await self.observe(view="stereo", after=after, deadline=deadline)
            if target.color.side not in source.state.get("closed_grips", []):
                raise ValueError("The grip opened during placement alignment")
            arguments, evidence = await asyncio.to_thread(target.resolve, source, measured_support=support)
            if support is None:
                support = {"world_xyz_m": evidence["support_world_xyz_m"], "source": evidence["support_source"]}
                if "witness_world_xyz_m" in evidence:
                    support["witness_world_xyz_m"] = evidence["witness_world_xyz_m"]
            self.trace.record("held_alignment", frame=source.frame_id, attempt=attempt,
                              goal=target.model_dump(), wrist=arguments, **evidence)
            if evidence["error_m"] <= target.tolerance_m:
                return
            if corrections == 2 or (interrupted and interrupted()):
                break
            self.trace.record("action_started", tool=action.tool, arguments=arguments)
            try:
                async with asyncio.timeout(max(0, deadline - monotonic())):
                    event = await call(self.pilot, "trial_control", action="step", lease=self.lease,
                                       tool="target", arguments=arguments)
            except TrialStopped:
                raise
            except RuntimeError as error:
                # A cross-body reach can be shortened without moving toward the fixture.
                x = arguments["values"][0]
                crosses_body = x < -.12 if target.color.side == "right" else x > .12
                if (sidestepped or not target.allow_reach_sidestep or not crosses_body
                        or "Wrist target is beyond reach" not in str(error) or (interrupted and interrupted())):
                    raise
                self.trace.record("action_result", action=action.model_dump(), executed_arguments=arguments,
                                  result={"executed": False, "reason": str(error)})
                step = {"kind": "walk", "direction": "left" if x < 0 else "right", "duration": .10}
                self.trace.record("placement_reach_recovery_started", frame=source.frame_id, step=step)
                async with asyncio.timeout(max(0, deadline - monotonic())):
                    move = await call(self.pilot, "trial_control", action="step", lease=self.lease,
                                      tool="route_motion", arguments={"steps": [step]})
                self.trace.record("placement_reach_recovery_finished", result=move["result"])
                if not move["result"].get("executed"):
                    raise ValueError("Could not make the short stance correction") from error
                sidestepped = True
                after = source.frame_id
                continue
            corrections += 1
            self.trace.record("action_result", action=action.model_dump(),
                              executed_arguments=arguments, result=event["result"])
            result = event["result"]
            if isinstance(result, dict) and any(result.get(flag) is False for flag in ("reached", "executed")):
                raise ValueError("The wrist did not reach its placement correction")
            after = source.frame_id
        raise ValueError("Repeated wrist corrections could not verify alignment before release")

    async def preview_landing_support(self, plan, frame, deadline):
        if not any(action.tool == "align_held_surface" and action.arguments.get("premeasure_support") is True
                   and not action.arguments.get("support_reference")
                   for action in plan.actions):
            return None
        try:
            return await self.observe(view="stereo", after=frame.frame_id, deadline=deadline)
        except TimeoutError:
            return None

    def referenced_landing(self, alignment):
        reference = self.scene_references.get(alignment.support_reference)
        if reference is None:
            raise ValueError(f"No verified resting location for {alignment.support_reference}")
        if alignment.support_clearance_m:
            raise ValueError("Measuring additional edge clearance requires an image support point")
        height = reference["support_height_m"]
        if alignment.support_height_m is not None and abs(alignment.support_height_m - height) > .08:
            raise ValueError("The resting location belongs to a different support height")
        x, _, z = reference["world_xyz_m"]
        return {"world_xyz_m": [x, height, z], "source": "verified_rest_location",
                "reference_frame": reference["frame"]}

    async def reference_color_aim(self, tool, arguments, deadline, *, after=-1):
        if tool == "aim_color":
            sides = {arguments.get("side", "right"): arguments}
        else:
            sides = {side: dict(arguments[side]) for side in ("left", "right")}
        names = {side: self.aim_references[side] for side in sides if side in self.aim_references}
        if not names:
            return arguments
        frame = await self.observe(view="stereo", after=after, deadline=deadline)

        def project():
            with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
                context = stereo_context(pair)
                for side, name in names.items():
                    recorded = self.scene_references.get(name)
                    if recorded is None or sides[side]["hue"] != recorded["hue"]:
                        raise ValueError(f"No matching verified location for {name}")
                    sides[side]["roi"] = project_color_region(context, recorded["world_xyz_m"], sides[side]["roi"])

        await asyncio.to_thread(project)
        self.trace.record("aim_reference_projection", frame=frame.frame_id, names=names,
                          regions={side: sides[side]["roi"] for side in names})
        return arguments if tool == "aim_color" else {**arguments, **sides}

    async def run(self, objective, *, trigger=None, verify=None, before_grip=None, before_release=None, carry_objective=None, menu_sequence=(),
                  require_camera=False, hold_seconds=0., recover=False, max_plans=6,
                  window_end=None, stop_on_grip_change=False, interrupted=None, queued_plan=None,
                  require_stereo=False, external_verification=False, pickup_stage=None):
        history = []
        frame = await self.observe()
        menu_evidence = []
        carrying = False
        carry_history = []
        attempted_routes = set()
        initial_grips = frame.state.get("closed_grips", [])
        if trigger:
            frame = await self.wait_for(trigger, frame)
            if frame is None:
                return "expired"
            self.trace.record("child_started", objective=objective, frame=frame.frame_id)
        for step in range(max_plans):
            if interrupted and interrupted():
                return "interrupted"
            if window_end is not None and monotonic() >= window_end:
                return "window_elapsed"
            holding = bool(carry_objective and "right" in frame.state.get("closed_grips", []))
            if holding and not carrying:
                self.trace.record("child_started", objective=carry_objective, frame=frame.frame_id)
                carry_history = []
            carrying = holding
            child_history = carry_history if carrying else history
            if step == 0 and queued_plan is not None:
                queued_plan.check_shape(self.controls)
                plan = queued_plan
                self.trace.record("queued_plan", frame=frame.frame_id, plan=plan.model_dump())
            else:
                self.trace.record("inference_started", frame=frame.frame_id)
                try:
                    plan = await self.plan(carry_objective if carrying else objective, frame, child_history,
                                           deadline=window_end if self.preset else None,
                                           exclude=("aim_color", "aim_color_pair", "aim_surface", "aim_surface_pair", "aim_at") if carrying else ())
                except TimeoutError:
                    if self.preset:
                        return "window_elapsed"
                    raise
                except ValueError:
                    if not recover:
                        raise
                    if step + 1 >= max_plans:
                        return "invalid_plan"
                    child_history.append(("planner", {"executed": False, "error": "Invalid action JSON. Return a shorter plan."}))
                    self.trace.record("local_recovery", preset=self.preset, frame=frame.frame_id,
                                      plans_remaining=max_plans - step - 1)
                    continue
            if plan.status != "continue":
                if plan.status == "done" and require_stereo:
                    return "verified" if await self.verify_stereo() else "stereo_unavailable"
                if plan.status == "done" and verify and recover:
                    if await self.wait_for(verify, frame) is not None:
                        return "verified"
                    history.append(("scene_check", {"verified": False, "question": verify.question}))
                    continue
                if plan.status == "done" and (verify or menu_sequence or require_camera or hold_seconds):
                    return "unverified"
                return plan.status
            if plan.when:
                checked = await self.wait_for(plan.when, frame)
                if checked is None:
                    if not recover:
                        return "expired"
                    history.append(("wait", {"check_expired": plan.when.question, "executed": False}))
                    frame = await self.observe(after=frame.frame_id)
                    continue
                frame = checked
            if plan.child:
                self.trace.record("child_started", objective=plan.child, frame=frame.frame_id)
                objective, history = plan.child, []
                continue
            action_failed = False
            previous_color_aim = None
            initial_frame = frame.frame_id
            action_applied = False
            grips = set(frame.state.get("closed_grips", []))
            support_preview = await self.preview_landing_support(plan, frame, window_end)
            for action in plan.actions:
                if interrupted and interrupted():
                    return "interrupted"
                if window_end is not None and monotonic() >= window_end:
                    return "window_elapsed"
                arguments = action.arguments.copy()
                command = action.tool
                if (command in ("route_motion", "body_action", "look", "scan_view")
                        or command == "target" and arguments.get("kind") != "wrist"):
                    support_preview = None
                state_error = None
                route_name = None
                if action.tool == "route_motion" and self.routes:
                    route_name = arguments.get("route")
                    if arguments.keys() != {"route"} or not isinstance(route_name, str) or route_name not in self.routes:
                        state_error = "Select one of the supplied named routes."
                    elif route_name in attempted_routes:
                        state_error = "This named route was already attempted. Check the live position."
                    else:
                        arguments = {"steps": self.routes[route_name]}
                if action.tool in ("look", "scan_view"):
                    try:
                        if action.tool == "scan_view":
                            turn = ScanView.model_validate(arguments).turn()
                        else:
                            turn = Look.model_validate(arguments).model_dump()
                    except ValueError as error:
                        state_error = str(error)
                    else:
                        command = "route_motion"
                        arguments = {"steps": [{"kind": "turn", **turn}]}
                if action.tool == "park_free_wrist":
                    side = arguments.get("side")
                    if side not in ("left", "right") or side in grips:
                        state_error = "Select an open hand to retract."
                    else:
                        command = "target"
                        arguments = {"kind": "wrist", "side": side, "values": PARKED_WRIST[side]}
                if action.tool == "aim_surface":
                    for axis in ("x", "y"):
                        value = arguments.get(axis)
                        if type(value) is not int or not 0 <= value < 1000:
                            state_error = "Surface aim needs x and y as integers from 0 to 999."
                            break
                        arguments[axis] = value / 1000
                condition = action.when
                aiming_side = arguments.get("side", "right")
                if action.tool in ("aim_color", "aim_surface") and aiming_side in grips:
                    state_error = f"The {aiming_side} hand is occupied. Use a free hand or place the held item first."
                if action.tool in ("aim_color_pair", "aim_surface_pair") and grips:
                    state_error = "Paired aiming requires two free hands."
                if action.tool == "lift_pair" and grips != {"left", "right"}:
                    state_error = "Paired lifting requires both grips to be closed."
                if action.tool == "grip":
                    side = arguments["side"]
                    closed = arguments["closed"]
                    guard = before_grip if closed else before_release
                    if guard:
                        condition = guard
                        if closed == (side in grips):
                            state_error = f"The {side} grip is already {'closed' if closed else 'open'}."
                if self.preset == "pick_up_and_confirm" and action.tool in ("grip", "grip_pair"):
                    requested = ({arguments.get("side"): arguments.get("closed")} if action.tool == "grip" else arguments)
                    closing = {side for side, closed in requested.items() if closed is True} - grips
                    opening = {side for side, closed in requested.items() if closed is False} & grips
                    if opening:
                        state_error = "Keep occupied grips closed during this pickup objective."
                    elif len(action.checks) + bool(condition) < len(closing):
                        state_error = "Provide a visible outline check for each grip that will close."
                if state_error:
                    self.trace.record("action_skipped", action=action.model_dump(), reason=state_error)
                    child_history.append((action.tool, {"executed": False, "error": state_error}))
                    action_failed = True
                    break
                if condition:
                    if action_applied:
                        frame = await self.observe(after=frame.frame_id)
                        action_applied = False
                    checked = await self.wait_for(condition, frame)
                    if checked is None:
                        if not recover:
                            return "expired"
                        child_history.append((action.tool, {"executed": False, "check_expired": condition.question}))
                        frame = await self.observe(after=frame.frame_id)
                        action_failed = True
                        break
                    frame = checked
                if action.checks:
                    if action_applied:
                        frame = await self.observe(after=frame.frame_id)
                        action_applied = False
                    passed = await self.check_frame(action.checks, frame)
                    if (not passed and previous_color_aim and self.preset == "pick_up_and_confirm"
                            and action.tool in ("grip", "grip_pair") and closing
                            and len(action.checks) == len(closing)
                            and all(c.expected and any(cue in c.question.lower() for cue in ("outline", "highlight", "rim"))
                                    for c in action.checks)):
                        passed = await self.check_aim_outlines(previous_color_aim, frame, closing)
                    if not passed:
                        child_history.append((action.tool, {"executed": False, "verification": "failed"}))
                        action_failed = True
                        break
                if interrupted and interrupted():
                    return "interrupted"
                if window_end is not None and monotonic() >= window_end:
                    return "window_elapsed"
                if action.tool == "lift_pair":
                    if action_applied:
                        frame = await self.observe(after=frame.frame_id)
                        action_applied = False
                    try:
                        height = PairLift.model_validate(arguments).up_m
                        wrists = observed_wrists(frame.state)
                        if not wrists or len(wrists) != 2:
                            raise ValueError("Current wrist poses unavailable")
                    except ValueError as error:
                        self.trace.record("action_skipped", action=action.model_dump(), reason=str(error))
                        child_history.append((action.tool, {"executed": False, "error": str(error)}))
                        action_failed = True
                        break
                    command = "wrist_pair"
                    arguments = {side: [pose[0], pose[1] + height, *pose[2:]] for side, pose in wrists.items()}
                if action.tool in ("align_held_surface", "upright_held_object"):
                    try:
                        if action.tool == "align_held_surface":
                            target = HeldAlignment.model_validate(arguments)
                            side = target.color.side
                        else:
                            target = HeldUpright.model_validate(arguments)
                            side = target.side
                        if side not in grips:
                            raise ValueError("The selected wrist must hold the object being aligned")
                        support = None
                        if isinstance(target, HeldAlignment) and target.support_reference is not None:
                            support = self.referenced_landing(target)
                        elif isinstance(target, HeldAlignment) and target.premeasure_support and support_preview:
                            try:
                                support = await asyncio.to_thread(target.support_from_frame, support_preview)
                            except ValueError as error:
                                self.trace.record("landing_support_preview_unavailable",
                                                  frame=support_preview.frame_id, reason=str(error))
                            else:
                                self.trace.record("landing_support_preview", frame=support_preview.frame_id,
                                                  world_xyz_m=support["world_xyz_m"])
                        if isinstance(target, HeldAlignment) and target.tolerance_m is not None:
                            after = max(frame.frame_id, support_preview.frame_id) if support_preview else frame.frame_id
                            await self.align_for_release(target, action, after, window_end, interrupted, support=support)
                            action_applied = True
                            continue
                        source = await self.observe(view="stereo", deadline=window_end)
                        if isinstance(target, HeldAlignment):
                            arguments, evidence = await asyncio.to_thread(target.resolve, source, measured_support=support)
                        else:
                            arguments, evidence = await asyncio.to_thread(target.resolve, source)
                    except TrialStopped:
                        raise
                    except (ValueError, RuntimeError, TimeoutError) as error:
                        self.trace.record("action_skipped", action=action.model_dump(), reason=str(error))
                        child_history.append((action.tool, {"executed": False, "error": str(error),
                                                            "failed_arguments": action.arguments}))
                        action_failed = True
                        break
                    command = "target"
                    self.trace.record("held_alignment", frame=source.frame_id, method=action.tool,
                                      goal=target.model_dump(), wrist=arguments, **evidence)
                if action.tool in ("aim_color", "aim_color_pair") and self.aim_references:
                    try:
                        arguments = await self.reference_color_aim(
                            action.tool, arguments, window_end, after=frame.frame_id if action_applied else -1,
                        )
                    except (KeyError, ValueError, TimeoutError) as error:
                        reason = str(error) or type(error).__name__
                        self.trace.record("action_skipped", action=action.model_dump(), reason=reason)
                        child_history.append((action.tool, {"executed": False, "error": reason}))
                        action_failed = True
                        break
                if action.tool in ("aim_color", "aim_color_pair") and self.native_color_aim:
                    try:
                        if action.tool == "aim_color":
                            color = ColorAim.model_validate(arguments)
                            targets = {color.side: color.model_dump(exclude={"side"})}
                        else:
                            colors = ColorAimPair.model_validate(arguments)
                            targets = {side: getattr(colors, side).model_dump(exclude={"side"})
                                       for side in ("left", "right")}
                    except ValueError as error:
                        self.trace.record("action_skipped", action=action.model_dump(), reason=str(error))
                        child_history.append((action.tool, {"executed": False, "error": str(error)}))
                        action_failed = True
                        break
                    command = "aim_colored_surfaces"
                    arguments = {"targets": targets}
                    if action_applied:
                        frame = await self.observe(after=frame.frame_id)
                        action_applied = False
                elif action.tool in ("aim_color", "aim_color_pair"):
                    if action_applied:
                        frame = await self.observe(after=frame.frame_id)
                        action_applied = False
                    try:
                        if action.tool == "aim_color":
                            resolve = ColorAim.model_validate(arguments).resolve_stereo
                        else:
                            resolve = ColorAimPair.model_validate(arguments).resolve
                        source = await self.observe(view="stereo")
                        arguments = await asyncio.to_thread(resolve, source)
                    except ValueError as error:
                        self.trace.record("action_skipped", action=action.model_dump(), reason=str(error))
                        child_history.append((action.tool, {"executed": False, "error": str(error)}))
                        action_failed = True
                        break
                    self.trace.record("aim_resolved", frame=source.frame_id, point=arguments)
                    command = "aim_surface" if action.tool == "aim_color" else "aim_surface_pair"
                self.trace.record("action_started", tool=action.tool, arguments=arguments)
                if route_name is not None:
                    attempted_routes.add(route_name)
                try:
                    async with asyncio.timeout(max(0, self.deadline-monotonic())):
                        event = await call(self.pilot, "trial_control", action="step", lease=self.lease,
                                           tool=command, arguments=arguments)
                except TrialStopped:
                    raise
                except RuntimeError as error:
                    if not recover:
                        raise
                    result = {"verified": False, "error": str(error)}
                    event = {"result": result}
                    if (command == "aim_colored_surfaces" and self.preset == "pick_up_and_confirm"
                            and "park_free_wrist" in self.controls and "both unobstructed eyes" in str(error)):
                        self.trace.record("stereo_recovery_started", error=str(error), frame=frame.frame_id)
                        try:
                            if (await self.recover_stereo_aim(frame, window_end, interrupted)
                                    and (not interrupted or not interrupted())
                                    and (window_end is None or monotonic() < window_end)):
                                event = await call(self.pilot, "trial_control", action="step", lease=self.lease,
                                                   tool=command, arguments=arguments)
                        except TrialStopped:
                            raise
                        except (RuntimeError, ValueError, TimeoutError) as retry_error:
                            event = {"result": {"verified": False, "error": str(retry_error)}}
                        finally:
                            self.trace.record("stereo_recovery_finished", result=event["result"])
                    elif action.tool == "aim_surface" and aiming_side == "right":
                        self.trace.record("aim_fallback", reason=str(error), point=arguments)
                        try:
                            async with asyncio.timeout(max(0, self.deadline-monotonic())):
                                event = await call(self.pilot, "trial_control", action="step", lease=self.lease,
                                                   tool="aim_at", arguments={axis: arguments[axis] for axis in ("x", "y")})
                        except TrialStopped:
                            raise
                        except RuntimeError as fallback_error:
                            event = {"result": {"verified": False, "error": str(fallback_error)}}
                self.trace.record("action_result", action=action.model_dump(),
                                  executed_arguments=arguments, result=event["result"])
                previous_color_aim = ((frame.frame_id, arguments["targets"])
                                      if command == "aim_colored_surfaces" and event["result"].get("plans") else None)
                action_applied = True
                history.append((action.tool, event["result"]))
                if carrying:
                    carry_history.append((action.tool, event["result"]))
                if action.tool == "grip_pair" and "closed_grips" in event["result"]:
                    grips = set(event["result"]["closed_grips"])
                elif action.tool == "grip" and "grip_closed" in event["result"]:
                    if event["result"]["grip_closed"]:
                        grips.add(event["result"]["side"])
                    else:
                        grips.discard(event["result"]["side"])
                if pickup_stage and grips and action.tool in (
                    "grip", "grip_pair", "translate_wrist", "lift_pair", "target", "wrist_pair"
                ):
                    frame = await self.observe(after=frame.frame_id, deadline=window_end)
                    action_applied = False
                    await pickup_stage(action.tool, frame)
                if stop_on_grip_change and grips != set(initial_grips):
                    await self.observe(after=frame.frame_id)
                    return "grip_changed"
                if require_camera and action.tool == "camera" and action.arguments.get("enabled"):
                    return "camera_observed" if await self.camera_frame() else "camera_unavailable"
                if isinstance(event["result"], dict) and any(
                    event["result"].get(flag) is False for flag in ("aligned", "verified", "reached", "executed")
                ):
                    action_failed = True
                    break
                if action.tool == "menu" and action.arguments.get("action") in ("click", "toggle"):
                    frame = await self.observe(after=frame.frame_id)
                    action_applied = False
                    panel = await asyncio.to_thread(frame.menu_panel)
                    self.trace.record("menu_observation", frame=frame.frame_id, panel=panel)
                    history.append(("visible_menu_panel", panel))
                    if menu_sequence and len(menu_evidence) < len(menu_sequence):
                        expected = menu_sequence[len(menu_evidence)]
                        if panel == expected or (expected == "closed" and panel is None and
                                                 action.arguments["action"] == "toggle" and menu_evidence):
                            menu_evidence.append(frame.frame_id)
                            if len(menu_evidence) == len(menu_sequence):
                                self.trace.record("menu_sequence_observed", frames=menu_evidence)
                                return "verified"
            if frame.frame_id == initial_frame or action_applied:
                frame = await self.observe(after=frame.frame_id)
            if action_failed and queued_plan is not None and max_plans == 1:
                return "queued_action_failed"
            if action_failed and self.preset:
                self.trace.record("local_recovery", preset=self.preset, frame=frame.frame_id,
                                  plans_remaining=max_plans - step - 1)
            if require_stereo and not action_failed:
                if await self.verify_stereo():
                    return "verified"
                history.append(("stereo_check", {"verified": False, "error": "Tracking HUD remains obscured."}))
                continue
            if hold_seconds and not action_failed and frame.state.get("closed_grips"):
                if samples := await self.hold_frames(frame, hold_seconds):
                    self.trace.record("hold_observation", frames=[sample.frame_id for sample in samples])
                    return "hold_observed"
            finish = (verify if self.preset or not recover else None) if verify else plan.verify
            if finish and not action_failed:
                if await self.wait_for(finish, frame) is not None:
                    return "verified"
            if external_verification and plan.actions and not finish and not action_failed:
                return "done"
        return "preset_unverified" if self.preset else "plan_limit"

    async def verify_stereo(self, after=-1, deadline=None):
        frame = await self.observe(view="stereo", after=after, deadline=deadline)
        def check():
            with Image.open(BytesIO(base64.b64decode(frame.data))) as pair:
                stereo_context(pair)
        try:
            await asyncio.to_thread(check)
        except ValueError as error:
            self.trace.record("stereo_verification", frame=frame.frame_id, passed=False, error=str(error))
            return False
        self.trace.record("stereo_verification", frame=frame.frame_id, passed=True)
        return True

    async def check_aim_outlines(self, prior, frame, closing):
        before, targets = prior
        if not closing <= targets.keys() or before >= frame.frame_id:
            return False
        before_path = self.trace.folder / f"frame-{before}.jpg"
        after_path = self.trace.folder / f"frame-{frame.frame_id}.jpg"
        def check():
            return {side: colored_pickup_outline(before_path, after_path, ColorTarget.model_validate(targets[side]))
                    for side in sorted(closing)}
        try:
            cues = await asyncio.to_thread(check)
        except (ValueError, OSError):
            return False
        self.trace.record("pickup_outline_fallback", before=before, after=frame.frame_id, cues=cues)
        return all(cues.values())

    async def recover_stereo_aim(self, previous, deadline, interrupted):
        """Let Omni choose free wrists to retract without changing the aim ROIs."""
        frame = await self.observe(after=previous.frame_id, deadline=deadline)
        free = set(PARKED_WRIST) - set(frame.state.get("closed_grips", []))
        if not free or (interrupted and interrupted()):
            return False
        self.trace.record("inference_started", frame=frame.frame_id, phase="stereo_recovery")
        plan = await self.plan(
            f"Both eye markers must be readable before retrying the pickup. Select which free wrists "
            f"to retract using park_free_wrist. Free hands: {', '.join(sorted(free))}. "
            "State explicit actions. The runner verifies stereo after the motion.",
            frame, [("stereo", {"error": "An eye tracking marker is obscured"})],
            exclude=tuple(tool for tool in self.controls if tool != "park_free_wrist"), deadline=deadline,
        )
        selected = {action.arguments.get("side") for action in plan.actions
                    if action.tool == "park_free_wrist" and not action.checks and not action.when}
        if (not selected or not selected <= free or plan.when or (interrupted and interrupted())
                or (deadline is not None and monotonic() >= deadline)):
            return False
        if len(selected) == 2:
            tool, arguments = "wrist_pair", PARKED_WRIST
        else:
            side = next(iter(selected))
            tool = "target"
            arguments = {"kind": "wrist", "side": side, "values": PARKED_WRIST[side]}
        self.trace.record("stereo_recovery_motion_started", frame=frame.frame_id, sides=sorted(selected))
        result = await call(self.pilot, "trial_control", action="step", lease=self.lease,
                            tool=tool, arguments=arguments)
        self.trace.record("stereo_recovery_motion_finished", result=result["result"])
        return await self.verify_stereo(after=frame.frame_id, deadline=deadline)


async def run_trial(args):
    async with streamable_http_client(PILOT) as (read, write, _):
        async with ClientSession(read, write) as pilot, httpx.AsyncClient(timeout=15) as http:
            await pilot.initialize()
            world = await call(pilot, "world_context")
            if world.get("access") != "invite" or not world.get("scene_ready"):
                raise ValueError("Stage the task in a scene-ready Invite instance")
            schemas = {tool.name: tool.inputSchema for tool in (await pilot.list_tools()).tools}
            lease = (await call(pilot, "trial_control", action="begin"))["lease"]
            trace = Trace("reactive-" + args.label)
            trace.record("start", world=world, objective=args.objective, seconds_limit=args.seconds)
            session = Session(pilot, http, trace, args.controls, schemas, trace.start + args.seconds, args.batch_size,
                              managed_pickups=bool(args.before_grip))
            session.lease = lease
            trigger = Check(question=args.when, seconds=args.trigger_seconds) if args.when else None
            verify = Check(question=args.verify, seconds=args.trigger_seconds) if args.verify else None
            before_grip = (Check(question=args.before_grip, seconds=args.trigger_seconds, confidence=.72)
                           if args.before_grip else None)
            before_release = (Check(question=args.before_release, seconds=args.trigger_seconds, confidence=.78)
                              if args.before_release else None)
            outcome = "aborted"
            try:
                outcome = await session.run(args.objective, trigger=trigger, verify=verify, before_grip=before_grip,
                                            before_release=before_release,
                                            carry_objective=args.carry_objective,
                                            menu_sequence=args.menu_sequence, require_camera=args.camera_evidence,
                                            hold_seconds=args.hold_evidence, recover=args.recover, max_plans=args.plans)
            except TimeoutError:
                outcome = "deadline"
            except (ValueError, RuntimeError, httpx.HTTPError, McpError) as error:
                outcome = "error"
                trace.record("error", reason=str(error))
            finally:
                trace.record("finished", outcome=outcome)
                await release_lease(pilot, lease, trace)
            print(outcome, flush=True)


async def release_lease(pilot, lease, trace):
    try:
        await call(pilot, "trial_control", action="end", lease=lease)
    except TrialStopped:
        return
    except (McpError, httpx.HTTPError):
        # A deadline can interrupt an observation while the MCP transport is sending it.
        async with streamable_http_client(PILOT) as (read, write, _):
            async with ClientSession(read, write) as recovery:
                await recovery.initialize()
                try:
                    await call(recovery, "trial_control", action="end", lease=lease)
                except TrialStopped:
                    pass
                trace.record("lease_reconnected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective")
    parser.add_argument("--label", choices=("menu-05", "object-01", "media-01", "smoke", "freeform"), default="smoke")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--when", help="Visible condition that starts a fresh planning session")
    parser.add_argument("--verify", help="Visible postcondition that ends the session after an action")
    parser.add_argument("--before-grip", help="Required visible contact check before closing a grip")
    parser.add_argument("--before-release", help="Required visible support check before opening a grip")
    parser.add_argument("--carry-objective", help="Fresh microtask to run while the right grip stays closed")
    parser.add_argument("--menu-sequence", nargs="+", default=(), choices=("launch_pad", "audio", "camera", "closed"),
                        help="Stop on a sequence of panels. Review the saved frames for closure.")
    parser.add_argument("--camera-evidence", action="store_true",
                        help="Save the live stream after connection for independent framing review.")
    parser.add_argument("--hold-evidence", type=float, default=0.,
                        help="Capture three frames over this duration with the same grip closed.")
    parser.add_argument("--trigger-seconds", type=float, default=2)
    parser.add_argument("--plans", type=int, default=6)
    parser.add_argument("--batch-size", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--recover", action="store_true",
                        help="Replan from a fresh frame when an individual action or its check fails.")
    parser.add_argument("--controls", nargs="+", choices=CONTROLS, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 180 or not 1 <= args.plans <= 60:
        parser.error("Use up to 180 seconds and 1 to 60 plans")
    if not 0 <= args.hold_evidence < args.seconds:
        parser.error("Hold evidence duration must fit within the trial")
    args.controls = {tool: CONTROLS[tool] for tool in args.controls}
    asyncio.run(run_trial(args))


if __name__ == "__main__":
    main()
