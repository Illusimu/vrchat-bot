"""Run the mapped in-game world route and stop at the first unverified edge."""
import asyncio
from contextlib import contextmanager
import ctypes
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import pythoncom
import pywintypes
import win32clipboard

from menu_selection import GAME_MIRROR_AIM_OFFSET, PointerCalibration, laser_tip
from menu_workflows import AUDIO, LAUNCH, locate


TEMPLATES = Path(__file__).with_name("menu_templates")
SIZE = np.array([640, 720])
MAP_PATH = TEMPLATES / "world_join.json"
MAP = json.loads(MAP_PATH.read_text(encoding="utf-8"))


class RouteStop(Exception):
    def __init__(self, reason, expected=None):
        super().__init__(reason)
        self.expected = expected


@lru_cache(maxsize=64)
def anchor_image(name):
    image = cv2.imread(str(TEMPLATES / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Missing world menu reference {name}")
    return image


def anchor(gray, name, mirror=False):
    box, threshold = MAP["anchors"][name]
    variants = (name, f"{name}_hover") if name in MAP["hover_anchors"] else (name,)
    if mirror:
        x, y = 0, 35  # Exclude the Windows title bar.
        bottom, right = gray.shape
        if name in MAP.get("mirror_anchors", ()):
            variants = (f"{name}_mirror",)
            if name in MAP.get("mirror_tilt_anchors", ()):
                variants += (f"{name}_mirror_tilt",)
            if name in MAP.get("mirror_hover_anchors", ()):
                variants += (f"{name}_mirror_hover",)
            variants += tuple(MAP.get("mirror_extra_anchors", {}).get(name, ()))
            scales = (.80, .90, 1, 1.10, 1.20)
            threshold = max(threshold, .78)
        else:
            scales = (.8, 1, 1.15, 1.3, 1.5, 1.65, 1.8)
    else:
        x, y, right, bottom = box
        scales = (1, .94, 1.06, .88, 1.12)
    crop = gray[y:bottom, x:right]
    best = None
    for variant in variants:
        reference = anchor_image(variant)
        for scale in scales:
            patch = cv2.resize(reference, None, fx=scale, fy=scale)
            h, w = patch.shape
            if h > crop.shape[0] or w > crop.shape[1]:
                continue
            scores = cv2.matchTemplate(crop, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, (u, v) = cv2.minMaxLoc(scores)
            if score >= threshold and (best is None or score > best[0]):
                best = score, np.array([x + u + w / 2, y + v + h / 2])
    return None if best is None else best[1]


def track_card(gray, reference, point):
    """Confirm the same card beneath a moving laser at nearly the same position."""
    x, y = np.rint(point).astype(int)
    height, width = gray.shape
    if x < 30 or x + 30 > width or y < 34 or y + 22 > height:
        return None
    region = gray[y-34:y+22, x-30:x+30]
    scores = cv2.matchTemplate(region, reference, cv2.TM_CCOEFF_NORMED)
    _, score, _, (u, v) = cv2.minMaxLoc(scores)
    return point + [u - 3, v - 3] if score >= .76 else None


@contextmanager
def clipboard_text(text):
    """Stage plain text for the game's Paste control, then restore prior text."""
    clipboard = win32clipboard
    sequence = ctypes.windll.user32.GetClipboardSequenceNumber
    clipboard_sequence = sequence()
    pythoncom.CoInitialize()
    try:
        data = pythoncom.OleGetClipboard()
        ole_formats = set()
        if data:
            entries = data.EnumFormatEtc(pythoncom.DATADIR_GET)
            while batch := entries.Next(16):
                ole_formats.update(entry[0] for entry in batch)
        del data
    finally:
        pythoncom.CoUninitialize()
    clipboard.OpenClipboard()
    try:
        if sequence() != clipboard_sequence:
            raise RouteStop("clipboard_changed_before_paste")
        formats = set()
        bookkeeping = set()
        item = clipboard.EnumClipboardFormats(0)
        while item:
            if item >= 0xc000 and clipboard.GetClipboardFormatName(item) in ("DataObject", "Ole Private Data"):
                bookkeeping.add(item)
            else:
                formats.add(item)
            item = clipboard.EnumClipboardFormats(item)
        formats.update(ole_formats - bookkeeping)
        if formats and (13 not in formats or formats - {1, 7, 13, 16}):
            raise RouteStop("clipboard_contains_non_text_data", "plain text or empty clipboard")
        original = clipboard.GetClipboardData(13) if formats else None
        clipboard.EmptyClipboard()
        clipboard.SetClipboardText(text, 13)
    finally:
        clipboard.CloseClipboard()
    saved_sequence = sequence()
    try:
        yield
    finally:
        clipboard.OpenClipboard()
        try:
            if sequence() == saved_sequence:
                clipboard.EmptyClipboard()
                if original is not None:
                    clipboard.SetClipboardText(original, 13)
        finally:
            clipboard.CloseClipboard()


class WorldRoute:
    def __init__(self, snapshot, menu, view, world_context):
        self.snapshot = snapshot
        self.menu = menu
        self.view = view
        self.world_context = world_context
        self.frame_id = 0
        self.source = None
        self.image = self.gray = None
        self.anchors = {}
        self.state = None
        self.step = "recognize_start"
        self.click_sent = False
        self.probes = 0
        self.trace = []
        self.last_aim = None
        self.search_submitted = False

    async def refresh(self):
        image, metadata = await self.snapshot()
        if metadata["frame_id"] > self.frame_id:
            self.frame_id = metadata["frame_id"]
            self.source = metadata.get("source")
            self.image = image.resize(tuple(SIZE))
            self.gray = np.asarray(self.image.convert("L"))
            self.anchors.clear()

    async def wait(self, check, timeout=3):
        end = monotonic() + timeout
        while monotonic() < end:
            await self.refresh()
            found = await asyncio.to_thread(check)
            if found is not None:
                return found
            await asyncio.sleep(.12)
        return None

    def find(self, names):
        points = {name: self.anchor(name) for name in names}
        return points if all(point is not None for point in points.values()) else None

    def anchor(self, name):
        if name not in self.anchors:
            mirror = self.source == "game_mirror"
            if name == "world_paste" and mirror:
                # Copy and Paste look alike in isolation. Locate the stacked pair,
                # then select its lower button.
                pair = anchor(self.gray, "world_clipboard_pair", mirror=True)
                self.anchors[name] = None if pair is None else pair + [0, 18]
            else:
                self.anchors[name] = anchor(self.gray, name, mirror=mirror)
        return self.anchors[name]

    def mirror_worlds(self):
        if self.anchor("world_results") is not None:
            return None
        search = self.anchor("worlds_search")
        if search is not None:
            return {"worlds_search": search}
        landmarks = self.find(("worlds_sidebar_title", "worlds_sidebar_my_worlds"))
        if landmarks is None:
            return None
        title = landmarks["worlds_sidebar_title"]
        below = landmarks["worlds_sidebar_my_worlds"]
        dx, dy = below - title
        if abs(dx) < 36 and 82 <= dy <= 132:
            return {"worlds_search": title + .60 * (below - title)}
        return None

    def screen(self, name, world):
        if name == "worlds" and self.source == "game_mirror":
            return self.mirror_worlds()
        if name == "world_details" and any(self.anchor(overlay) is not None
                                           for overlay in ("world_create", "world_created")):
            return None
        names = [key.format(**(world or {})) for key in MAP["screens"][name]]
        if name == "search_results" and self.source == "game_mirror":
            return self.mirror_search_results(world)
        points = self.find(names)
        if name == "search_filled" and points is not None:
            query = points[world["input"]]
            submit = points["world_submit"]
            dx, dy = submit - query
            if dx < 45 or abs(dy) > 32:
                return None
            x, y = submit.round().astype(int)
            if self.source == "game_mirror":
                start = round(query[0] + anchor_image(f'{world["input"]}_mirror').shape[1] * .62)
                end = x - 48  # Leave out the clear-text button beside Submit.
                if start >= end:
                    return None
                rgb = np.asarray(self.image, dtype=np.int16)[max(0, y-11):y+12, start:end]
                extra = (rgb.min(axis=2) > 145) & (np.ptp(rgb, axis=2) < 45)
            else:
                extra = self.gray[y-5:y+5, round(query[0])+34:x-27] > 175
            if extra.sum() >= 12:
                return None
        return points

    def mirror_search_results(self, world):
        results = self.find(("world_results", world["card"]))
        if results is None:
            return None
        heading = results["world_results"]
        card = results[world["card"]]
        if not 45 <= card[1] - heading[1] <= 295:
            return None
        if self.search_submitted:
            return results
        query = self.anchor(world["results_query"])
        if query is not None and query[0] > heading[0] + 50 and abs(query[1] - heading[1]) <= 90:
            return results
        return None

    def initial(self, world=None):
        if world is not None:
            if self.screen("instance_created", world) is not None:
                return "instance_created"
            if self.screen("world_details", world) is not None:
                return "world_details"
            if self.source == "game_mirror":
                results = self.mirror_search_results(world)
            else:
                results = self.find(("world_results", world["results_query"], world["card"]))
            if results is not None:
                return "search_results"
            if self.screen("search_filled", world) is not None:
                return "search_filled"
        prompt = self.find(("world_input", "world_paste", "world_submit"))
        if prompt is not None:
            x, y = prompt["world_submit"].round().astype(int)
            if self.source == "game_mirror":
                left = round(prompt["world_paste"][0]) + 22
                text_end = x - 48  # Leave out the clear-text button beside Submit.
                if left >= text_end:
                    return None
                rgb = np.asarray(self.image, dtype=np.int16)[max(0, y-11):y+12, :]
                letters = (rgb.min(axis=2) > 145) & (np.ptp(rgb, axis=2) < 45)
                return "search_input" if letters[:, left:text_end].sum() < 12 else None
            if (self.gray[y-5:y+5, x-230:x-24] > 175).sum() < 12:
                return "search_input"
            return None
        if self.screen("worlds", world) is not None:
            return "worlds"
        match = locate(self.image, (*AUDIO, *LAUNCH))
        return match.panel.name if match else None

    def launch_control(self):
        match = locate(self.image, LAUNCH)
        if match is None:
            return None
        point = match.project([match.panel.controls["worlds"]])[0]
        if self.source == "game_mirror":
            offset = GAME_MIRROR_AIM_OFFSET
        else:
            offset = (.025, .405) if match.panel.image == "launch_panel" else match.panel.aim_offset
        return point, offset

    def worlds_highlighted(self, point):
        x, y = point.round().astype(int)
        crop = np.asarray(self.image.crop((x-12, y-12, x+12, y+12)), dtype=np.int16)
        red, green, _ = crop.transpose(2, 0, 1)
        return ((green > 60) & (green - red > 25)).mean() > .56

    def paste_highlighted(self, point):
        x, y = point.round().astype(int)
        crop = np.asarray(self.image.crop((x-6, y-6, x+7, y+7)), dtype=np.int16)
        red, green, _ = crop.transpose(2, 0, 1)
        return ((green > 58) & (green - red > 42)).mean() > .65

    async def aim(self, control, target_for_frame, nominal, radius, hover=None, *, paste_hover=False):
        target = await self.wait(target_for_frame, 1)
        if target is None:
            raise RouteStop("control_not_recognized", control)
        point = target.copy()
        command = np.clip(nominal, .003, .997)
        mirror = self.source == "game_mirror"
        if mirror:
            calibration = PointerCalibration(point / SIZE)
            calibration.initial[:] = command
            calibration.probe()
        previous = self.image
        for _ in range(8 if mirror else 6):
            self.probes += 1
            await self.menu("aim", *map(float, command))
            if mirror and control in ("world_submit", "world_new_instance"):
                await asyncio.sleep(.45)
            await self.refresh()
            updated = target_for_frame()
            if updated is None:
                raise RouteStop("control_changed_while_aiming", control)
            point = updated
            if paste_hover and self.paste_highlighted(point):
                first_frame = self.frame_id

                def stable_paste():
                    updated = target_for_frame()
                    if (self.frame_id > first_frame and updated is not None
                            and np.max(np.abs(updated - point)) < 5 and self.paste_highlighted(updated)):
                        return updated

                if await self.wait(stable_paste, timeout=.7) is not None:
                    return
            tip = await asyncio.to_thread(laser_tip, self.image, previous if mirror else None, point / SIZE)
            previous = self.image
            self.last_aim = {"control_px": point.round(1).tolist(), "command": command.tolist(),
                             "tip_px": (np.array(tip) * SIZE).round(1).tolist() if tip else None}
            delta = point - np.asarray(tip) * SIZE if tip else None
            if delta is not None and np.all(np.abs(delta) <= radius):
                if hover is None or hover(point):
                    return
                command = np.clip(command + [0, .022], .003, .997)
                continue
            if mirror:
                calibration.target = point / SIZE
                command = calibration.correct(command, tip) if tip is not None else calibration.probe()
                if command is None:
                    break
            elif delta is None:
                command = np.clip(command + [0, .026], .003, .997)
            else:
                command = np.clip(command + np.clip(delta / SIZE * .65, -.09, .09), .003, .997)
        raise RouteStop("pointer_not_confirmed", control)

    async def click(self):
        self.click_sent = True
        await self.menu("click")

    async def wait_for_worlds_after_click(self, world):
        def worlds_screen():
            found = self.screen("worlds", world)
            if found and self.source == "game_mirror":
                x, y = found["worlds_search"] / SIZE
                if not (.08 <= x <= .92 and .20 <= y <= .92):
                    return None
            return found

        matched = await self.wait(worlds_screen)
        if matched is not None or self.source != "game_mirror":
            return matched
        # Opening Worlds from a low Launch Pad can leave its search row clipped.
        for pitch in (24, 16):
            await self.view(pitch)
            matched = await self.wait(worlds_screen, timeout=2)
            if matched is not None:
                return matched
        await self.view(-40)
        return await self.wait(worlds_screen, timeout=.7)

    async def open_worlds(self, world):
        if self.state == "worlds":
            return
        if self.state == "audio":
            self.step = "exit_audio"
            await self.menu("toggle")
            await self.refresh()
            if self.initial() is not None:
                raise RouteStop("audio_close_unverified", "closed Quick Menu")
            await self.menu("toggle")
        self.step = "open_worlds"
        found = await self.wait(self.launch_control)
        if found is None:
            raise RouteStop("screen_mismatch", "launch_pad")
        if self.state != "launch_pad":
            self.state = "launch_pad"
            self.trace.append(self.state)
        for _ in range(2):
            point, offset = found
            if .57 < point[1] / SIZE[1] < .74:
                break
            await self.view(float(np.clip((.65 - point[1] / SIZE[1]) * 95, -22, 22)))
            found = await self.wait(self.launch_control)
            if found is None:
                raise RouteStop("screen_mismatch", "launch_pad")
        point, offset = found

        def world_button():
            located = self.launch_control()
            return None if located is None else located[0]

        await self.aim("worlds_tile", world_button, point / SIZE - offset, (20, 16), self.worlds_highlighted)
        await self.click()
        if await self.wait_for_worlds_after_click(world) is None:
            raise RouteStop("screen_mismatch", "worlds")
        self.state = "worlds"
        self.trace.append("worlds")

    async def edge(self, step, world, name):
        self.step = control = step["control"].format(**world)
        self.click_sent = False
        mirror = self.source == "game_mirror"
        mirror_paste = mirror and step.get("paste")
        mirror_turn = [0., 0.]
        results_verified = False
        card_reference = None

        def current_control():
            if mirror and step["from"] == "search_input":
                screen = self.find(("world_input", control))
            elif results_verified:
                screen = self.find(("world_results", world["card"]))
                if screen is None and card_reference is not None:
                    texture, original, original_heading = card_reference
                    heading = self.anchor("world_results")
                    if heading is not None and np.max(abs(heading - original_heading)) <= 8:
                        return track_card(self.gray, texture, original + heading - original_heading)
            else:
                screen = self.screen(step["from"], world)
            if screen is None:
                return None
            if "relative_to" in step and self.source != "game_mirror":
                name, x, y = step["relative_to"]
                return screen[name] + [x, y]
            point = screen[control] if control in screen else self.anchor(control)
            if point is None and mirror and control == "world_new_instance":
                # The laser can cover the tile's label. The row heading and
                # adjacent tile still constrain the location of the first tile.
                actions = self.anchor("world_details_actions")
                preload = self.anchor("world_clear_preload")
                if actions is not None and preload is not None:
                    dx, dy = preload - actions
                    if 54 <= dx <= 100 and 44 <= dy <= 73:
                        return preload + [-dx + 10, -4]
            return point

        point = await self.wait(current_control, 1.5)
        if point is None and mirror and control == "world_new_instance":
            details = self.screen(step["from"], world)
            if details is not None and details[world["details"]][1] > SIZE[1] * .46:
                # The first actions row may be below the frame when the
                # verified details heading sits this low in the mirror.
                await self.view(-12)
                point = await self.wait(current_control)
        if point is None:
            raise RouteStop("screen_or_control_mismatch", [step["from"], control])
        results_verified = mirror and step["from"] == "search_results"
        if mirror:
            for _ in range(2):
                x, y = point / SIZE
                # Submit is on the right edge of the input. Keep the title at
                # the left visible instead of centering the arrow horizontally.
                submit = control == "world_submit"
                if .35 <= x <= (.84 if submit else .78) and .61 <= y <= .87:
                    break
                pitch = float(np.clip(((.66 if submit else .73) - y) * 70, -20, 20))
                yaw = 0.0 if submit and .35 <= x <= .84 else float(np.clip((x - .54) * 70, -20, 20))
                await self.view(pitch, yaw)
                mirror_turn[0] += pitch
                mirror_turn[1] += yaw
                point = await self.wait(current_control)
                if point is None:
                    raise RouteStop("screen_or_control_mismatch", [step["from"], control])
        elif step.get("center_view"):
            for _ in range(2):
                error_y = step["center"][1] - point[1]
                if abs(error_y) <= 10:
                    break
                await self.view(float(np.clip(error_y / 6.6, -24, 24)))
                point = await self.wait(current_control)
                if point is None:
                    raise RouteStop("screen_or_control_mismatch", [step["from"], control])
        if results_verified:
            x, y = np.rint(point).astype(int)
            height, width = self.gray.shape
            heading = self.anchor("world_results")
            if heading is not None and 27 <= x <= width - 27 and 31 <= y <= height - 19:
                texture = self.gray[y-31:y+19, x-27:x+27].copy()
                if texture.std() >= 16:
                    card_reference = texture, point.copy(), heading.copy()
        if mirror:
            if "mirror_aim" in step:
                nominal = np.asarray(step["mirror_aim"]) + point / SIZE - step["mirror_center"]
            else:
                nominal = point / SIZE - GAME_MIRROR_AIM_OFFSET
        else:
            nominal = np.asarray(step["aim"]) + (point - step["center"]) / SIZE
        hover = step.get("hover")
        check_hover = (lambda _: self.anchor(hover) is not None) if hover else None
        radius = step.get("mirror_radius", step["radius"]) if mirror else step["radius"]
        if mirror and control == "world_new_instance":
            radius = (23, 14)
        if mirror and control == "world_submit":
            radius = (8, 7)
        if mirror_paste:
            radius = (13, 16)
            check_hover = self.paste_highlighted
        await self.aim(control, current_control, nominal, np.asarray(radius), check_hover,
                       paste_hover=mirror_paste)
        if step.get("paste"):
            with clipboard_text(name):
                await self.click()
                if mirror_paste and any(mirror_turn):
                    await self.view(-mirror_turn[0], -mirror_turn[1])
                if await self.wait(lambda: self.find((world["input"],))) is None:
                    raise RouteStop("search_text_unverified", name)
        else:
            if step["to"] == "joined":
                self.join_started = monotonic()
            await self.click()
            if control == "world_submit":
                self.search_submitted = True
        expected = step["to"]
        if expected != "joined":
            def expected_screen():
                return self.screen(expected, world)

            correction = step.get("mirror_after_view") if self.source == "game_mirror" else None
            matched = await self.wait(expected_screen, timeout=1 if correction else 3)
            if matched is None and correction:
                yaw, pitch = correction
                await self.view(pitch, yaw)
                matched = await self.wait(expected_screen)
            if matched is None:
                raise RouteStop("screen_mismatch", expected)
            self.state = expected
            self.trace.append(expected)

    async def run(self, name, *, details_only=False):
        started = monotonic()
        try:
            saved_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return {"status": "stopped", "step": "load_map", "reason": str(error)}
        MAP.update(saved_map)
        anchor_image.cache_clear()
        world = MAP["worlds"].get(name)
        if world is None:
            return {"status": "world_not_mapped", "mapped_worlds": list(MAP["worlds"])}
        previous = await self.world_context()
        try:
            if source := world.get("capture_source"):
                await self.refresh()
                if self.source != source:
                    raise RouteStop("capture_source_mismatch", source)
            self.state = await self.wait(lambda: self.initial(world))
            if self.state is None:
                raise RouteStop("start_screen_unrecognized", ["audio", "launch_pad", "worlds", "verified search input"])
            if details_only and self.state == "instance_created":
                raise RouteStop("dismiss_instance_confirmation", "world_details")
            self.trace.append(self.state)
            start = {"search_input": 1, "search_filled": 2, "search_results": 3,
                     "world_details": 4, "instance_created": 6}.get(self.state, 0)
            if start == 0:
                await self.open_worlds(world)
            for step in MAP["steps"][start:]:
                if details_only and step["to"] not in ("search_input", "search_filled", "search_results", "world_details"):
                    break
                await self.edge(step, world, name)
            if details_only:
                room = await self.world_context()
                if (self.state != "world_details" or not previous.get("instance_key")
                        or room.get("instance_key") != previous["instance_key"]):
                    raise RouteStop("world_details_unverified", "details in the original instance")
                return {"status": "found", "world": name, "creator": world["creator"],
                        "screens": self.trace, "probes": self.probes,
                        "seconds": round(monotonic() - started, 2)}
            join_start = self.join_started
            while monotonic() - join_start < 35:
                room = await self.world_context()
                if (room.get("world_id") == world["world_id"] and room.get("access") == "invite"
                        and room.get("scene_ready") and room.get("instance_key") != previous.get("instance_key")):
                    return {"status": "joined", "world": name, "access": "invite",
                            "instance": room.get("instance_number"), "screens": self.trace,
                            "probes": self.probes, "seconds": round(monotonic() - started, 2),
                            "load_after_trigger_seconds": round(monotonic() - join_start, 2)}
                await asyncio.sleep(.45)
            raise RouteStop("join_unverified", f"new Invite instance of {name}")
        except (RouteStop, ValueError, OSError, pywintypes.error) as error:
            if self.image is not None:
                folder = Path(__file__).with_name("build")
                folder.mkdir(exist_ok=True)
                path = folder / datetime.now().strftime("world-route-stop-%H%M%S%f.png")
                self.image.save(path)
            else:
                path = None
            result = {"status": "stopped", "step": self.step, "last_screen": self.state,
                      "expected": getattr(error, "expected", None), "reason": str(error),
                      "trigger_sent_at_step": self.click_sent, "frame_id": self.frame_id,
                      "diagnostic": str(path) if path else None, "last_aim": self.last_aim,
                      "screens": self.trace, "probes": self.probes,
                      "seconds": round(monotonic() - started, 2)}
            if isinstance(error, RouteStop) and str(error) == "clipboard_contains_non_text_data":
                result["recovery"] = {
                    "input": name, "method": "in_game_keyboard", "resume_from": "search_filled",
                    "instruction": "Enter the world title exactly, leave the input open, and retry the route.",
                }
            return result
