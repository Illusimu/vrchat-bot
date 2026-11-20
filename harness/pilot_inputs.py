"""Locomotion and handheld photo controls for the local VRChat harness."""
import asyncio
import base64
import io
import json
import sys
import time
from typing import Literal
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP, Image
from pythonosc.udp_client import SimpleUDPClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_pose import read_eye

server = FastMCP("Pilot inputs")
osc = SimpleUDPClient("127.0.0.1", 9000)
grip_samples = {}


@server.tool()
async def hold_menu_upright() -> str:
    """Pose the left controller to present SV1's wrist Quick Menu toward the eyes. Observe headset to verify centering before aiming."""
    async with streamable_http_client("http://127.0.0.1:29746/mcp") as (reader, writer, _):
        async with ClientSession(reader, writer) as client:
            await client.initialize()
            result = await client.call_tool("target", {"kind": "wrist", "side": "left",
                "values": [-.17, 1.48, -.40, 0, 100, -12]})
    if result.isError:
        raise ValueError(str(result.content))
    return "Left wrist command completed. Verify the panel in the headset; adjust for seated height if needed."


@server.tool()
async def walk(forward: float, right: float, seconds: float) -> str:
    """Walk relative to player heading. Axes -1..1, duration 0.1..2 seconds. Observe headset to check displacement."""
    if not (-1 <= forward <= 1 and -1 <= right <= 1 and .1 <= seconds <= 2):
        raise ValueError("Axes must be within [-1,1] and duration within [0.1,2].")
    try:
        osc.send_message("/input/Vertical", float(forward))
        osc.send_message("/input/Horizontal", float(right))
        await asyncio.sleep(seconds)
    finally:
        osc.send_message("/input/Vertical", 0.0)
        osc.send_message("/input/Horizontal", 0.0)
    return "Released movement axes. Verify displacement with a fresh observation."


async def frame(view: str) -> bytes:
    async with streamable_http_client("http://127.0.0.1:29746/mcp") as (reader, writer, _):
        async with ClientSession(reader, writer) as client:
            await client.initialize()
            result = await client.call_tool("observe", {"view": view})
    if result.isError or not result.content or result.content[0].type != "image":
        raise ValueError(f"No current {view} frame available")
    return base64.b64decode(result.content[0].data, validate=True)


@server.tool()
async def camera_state() -> dict:
    """Measure head, handheld lens and decoded palm markers. Head-local axes are right/up/forward. An empty palms object means missing telemetry."""
    rgb = np.asarray(PILImage.open(io.BytesIO(await frame("left"))).convert("RGB"))
    hud = read_eye(rgb)
    if hud is None or not hud["photo_available"]:
        return {"available": False, "reason": "No valid photo-camera pose in the current eye HUD"}
    head, lens = hud["screen"], hud["photo"]
    measured = hud["palms"]
    if len(measured) == 2 and all(np.array_equal(a, b) for a, b in zip(measured["left"], measured["right"])):
        return {"available": False, "reason": "Both palm packets contain the identical transform; grip measurement unavailable"}
    delta = lens[:3] - head[:3]
    distance = float(np.linalg.norm(delta))
    relative = Rotation.from_quat(head[3:]).inv().apply(delta)
    lens_forward = Rotation.from_quat(lens[3:]).apply([0, 0, 1])
    alignment = None if distance < .001 else float(np.degrees(np.arccos(
        np.clip(np.dot(lens_forward, -delta / distance), -1, 1))))
    lens_rotation = Rotation.from_quat(lens[3:])
    palms = {}
    for side, (pos, axes) in hud["palms"].items():
        palms[side] = {"world_xyz": pos.tolist(),
                       "world_xyzw": Rotation.from_matrix(axes).as_quat().tolist(),
                       "lens_to_palm_lens_local_m": lens_rotation.inv().apply(pos - lens[:3]).tolist(),
                       "distance_to_lens_m": float(np.linalg.norm(pos - lens[:3]))}
    return {"available": True, "palms": palms, "head_world_xyz_xyzw": head.tolist(),
            "lens_world_xyz_xyzw": lens.tolist(), "lens_from_head_right_up_forward_m": relative.tolist(),
            "distance_m": distance, "lens_angle_from_head_deg": alignment,
            "angle_meaning": "0 points toward the head, 180 faces away; verify framing in the stream preview"}


@server.tool()
async def calibrate_grip(side: Literal["left", "right"], stage: Literal["start", "sample", "finish"]) -> dict:
    """After gripping, start at rest, move ONLY the held wrist 10-20 cm, sample, then move it in another direction and finish. Keep head still and camera dolly off."""
    state = await camera_state()
    hand = state.get("palms", {}).get(side)
    if not hand:
        grip_samples.pop(side, None)
        raise ValueError("Missing live palm or lens. Equip the SV1 palm test build.")
    pos = np.array(hand["world_xyz"])
    head = np.array(state["head_world_xyz_xyzw"][:3])
    lens = np.array(state["lens_world_xyz_xyzw"][:3])
    offset = np.array(hand["lens_to_palm_lens_local_m"])
    relative_rot = Rotation.from_quat(state["lens_world_xyz_xyzw"][3:]).inv() * Rotation.from_quat(hand["world_xyzw"])
    now = time.monotonic()
    if stage == "start":
        grip_samples[side] = []
        if np.linalg.norm(offset) > .45:
            raise ValueError("Palm is over 45 cm from the lens; bring it to the camera handle first.")
    samples = grip_samples.get(side)
    expected = {"start": 0, "sample": 1, "finish": 2}[stage]
    if samples is None or len(samples) != expected or (samples and now-samples[0][0]>240):
        raise ValueError("Start a new three-sample grip check.")
    if samples:
        _, prev_pos, _, prev_lens, _, _ = samples[-1]
        _, _, first_head, _, first_offset, first_rot = samples[0]
        if (np.linalg.norm(head-first_head)>.05 or np.linalg.norm(pos-prev_pos)<.08 or
            np.linalg.norm(lens-prev_lens)<.06 or np.linalg.norm(offset-first_offset)>.035 or
            (first_rot.inv()*relative_rot).magnitude()>np.radians(12)):
            raise ValueError("Grip unverified: check head stability, wrist/lens displacement and relative pose. Retry with dolly off.")
    if stage == "finish":
        d1 = samples[1][1]-samples[0][1]
        d2 = pos-samples[1][1]
        if abs(np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))) > .85:
            raise ValueError("Move the wrist along a different axis for independent verification.")
    samples.append((now,pos,head,lens,offset,relative_rot))
    if stage != "finish":
        return {"samples": len(samples), "verified": False}
    result = {"verified": True, "side": side, "method": "two independent wrist-only displacements",
              "lens_to_palm_local_m": np.mean([s[4] for s in samples], axis=0).tolist(),
              "palm_rotation_in_lens_xyzw": relative_rot.as_quat().tolist(),
              "recorded": datetime.now().astimezone().isoformat()}
    folder = Path(__file__).resolve().parents[1] / "build" / "grip-calibration"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{side}-{datetime.now():%Y%m%d-%H%M%S-%f}.json").write_text(json.dumps(result, indent=2))
    del grip_samples[side]
    return result


@server.tool()
async def take_selfie() -> Image:
    """Save the current stream preview into chat. Check observe(camera) for the avatar's face first. This captures whatever the stream shows."""
    data = await frame("camera")
    folder = Path(__file__).resolve().parents[1] / "build" / "selfies"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"selfie-{datetime.now():%Y%m%d-%H%M%S-%f}.jpg").write_bytes(data)
    return Image(data=data, format="jpeg")


if __name__ == "__main__":
    server.run()
