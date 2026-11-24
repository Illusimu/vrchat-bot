"""Local visual verification for VRChat menu actions."""
import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from menu_selection import GAME_MIRROR_AIM_OFFSET, laser_tip
from spatial_geometry import match_stereo_pixel, stereo_context
from world_map import BASELINE, K


@dataclass(frozen=True)
class Panel:
    name: str
    image: str
    origin: tuple
    controls: dict
    aim_offset: tuple
    track: tuple = ()
    hitboxes: dict | None = None


LAUNCH = (
    Panel("launch_pad", "launch_panel", (105, 134),
          {"audio": (341, 425), "worlds": (290, 320), "camera_tab": (319, 439),
           "social": (246, 394), "calibrate_fbt": (334, 407), "respawn": (292, 423)}, (.04, .45),
          hitboxes={"respawn": ((276, 425), (302, 413), (306, 423), (281, 434))}),
    Panel("launch_pad", "launch_front_panel", (254, 373),
          {"audio": (353, 563), "worlds": (335, 470), "camera_tab": (335, 564),
           "social": (296, 504), "calibrate_fbt": (351, 544), "respawn": (318, 544)}, (.02, .36)),
    Panel("launch_pad", "launch_floor_panel", (191, 394),
          {"audio": (309, 538), "worlds": (273, 458), "camera_tab": (286, 546),
           "social": (232, 499), "calibrate_fbt": (304, 521), "respawn": (262, 530)}, (.12, .37)),
)
AUDIO = (
    Panel("audio", "audio_panel", (124, 318), {}, (.02, .46), ((233, 456), (327, 427))),
    Panel("audio", "audio_floor_panel", (190, 402), {}, (.12, .38), ((239, 443), (306, 431))),
    Panel("audio", "audio_low_panel", (166, 510), {}, (.08, .35), ((230, 566), (307, 549))),
    Panel("audio", "audio_presented_panel", (228, 269), {}, (.035, .41), ((293, 346), (372, 323))),
    Panel("audio", "audio_high_tilt_panel", (238, 100), {}, (.035, .41), ((394, 322), (580, 270))),
)
CAMERA = (
    Panel("camera", "camera_panel_mirror", (311, 320), {"photo_camera": (389, 462)}, (.033, .59),
          hitboxes={"photo_camera": ((342, 439), (422, 419), (443, 484), (358, 505))}),
    Panel("camera", "camera_standard_panel_mirror", (264, 187), {"photo_camera": (344, 321)}, (.033, .59),
          hitboxes={"photo_camera": ((298, 301), (376, 281), (394, 343), (317, 366))}),
)
CAMERA_VIEWFINDER = (
    Panel("camera_viewfinder", "camera_viewfinder_controls_mirror", (350, 503),
          {"camera_anchor_menu": (381, 530)}, (.09, .55),
          hitboxes={"camera_anchor_menu": ((358, 508), (403, 511), (401, 553), (356, 550))}),
    Panel("camera_anchor", "camera_anchor_options_mirror", (422, 552),
          {"camera_anchor_default": (451, 580), "camera_anchor_local": (499, 582), "camera_anchor_world": (550, 585)}, (.09, .55),
          hitboxes={"camera_anchor_default": ((429, 559), (475, 559), (473, 603), (428, 601)),
                    "camera_anchor_world": ((529, 562), (573, 563), (572, 608), (525, 607))}),
    Panel("camera_anchor", "camera_anchor_left_mirror", (343, 552),
          {"camera_anchor_default": (370, 589), "camera_anchor_local": (420, 586), "camera_anchor_world": (468, 582)}, (.09, .55),
          hitboxes={"camera_anchor_default": ((349, 567), (393, 564), (393, 608), (349, 612)),
                    "camera_anchor_world": ((447, 558), (491, 556), (495, 601), (448, 605))}),
    Panel("camera_anchor", "camera_anchor_default_mirror", (421, 499),
          {"camera_anchor_default": (451, 523), "camera_anchor_local": (501, 526), "camera_anchor_world": (549, 527)}, (.09, .55),
          hitboxes={"camera_anchor_default": ((430, 503), (473, 503), (472, 548), (427, 546)),
                    "camera_anchor_world": ((528, 508), (569, 510), (569, 550), (525, 550))}),
    Panel("camera_anchor", "camera_anchor_front_mirror", (406, 482),
          {"camera_anchor_default": (433, 514), "camera_anchor_local": (480, 517), "camera_anchor_world": (528, 517)},
          (.09, .55), hitboxes={"camera_anchor_default": ((415, 493), (454, 492), (456, 535), (415, 535)),
                                "camera_anchor_world": ((509, 495), (551, 495), (551, 537), (510, 537))}),
    Panel("camera_anchor", "camera_anchor_front_default_mirror", (409, 485),
          {"camera_anchor_default": (433, 512), "camera_anchor_local": (480, 514), "camera_anchor_world": (530, 515)},
          (.09, .55), hitboxes={"camera_anchor_world": ((510, 494), (550, 495), (551, 534), (510, 534))}),
)


def features(gray, mask=None):
    return cv2.SIFT_create(nfeatures=2400, contrastThreshold=.012).detectAndCompute(gray, mask)


@lru_cache(maxsize=None)
def reference(panel_image):
    path = Path(__file__).with_name("menu_templates") / f"{panel_image}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Missing menu reference {panel_image}")
    return features(image[:, :, 0], image[:, :, 3])


@dataclass
class Located:
    panel: Panel
    transform: np.ndarray

    def project(self, points):
        local = np.asarray(points, dtype=np.float32) - np.float32(self.panel.origin)
        return cv2.perspectiveTransform(local.reshape(1, -1, 2), self.transform)[0]


def locate(image, panels):
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = features(gray)
    if descriptors is None or len(keypoints) < 2:
        return None
    candidates = []
    for panel in panels:
        keys, signatures = reference(panel.image)
        pairs = cv2.BFMatcher().knnMatch(signatures, descriptors, k=2)
        good = [match for match, other in pairs if match.distance < .72 * other.distance]
        if len(good) < 12:
            continue
        src = np.float32([keys[match.queryIdx].pt for match in good])
        dst = np.float32([keypoints[match.trainIdx].pt for match in good])
        transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
        if inliers is None:
            continue
        kept = inliers.ravel().astype(bool)
        if kept.sum() >= 12 and kept.mean() >= .65 and min(np.ptp(src[kept], axis=0)) >= 36:
            low, high = src[kept].min(axis=0), src[kept].max(axis=0)
            x0, y0 = low
            x1, y1 = high
            corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            footprint = cv2.perspectiveTransform(corners[None], transform)[0]
            if (not np.isfinite(footprint).all() or not cv2.isContourConvex(footprint)
                    or cv2.contourArea(footprint, oriented=True) < 600):
                continue
            candidates.append((int(kept.sum()), Located(panel, transform)))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def fill_percent(image, located):
    """Estimate the slider position from the contiguous cyan fill across its track."""
    fractions = np.linspace(0, 1, 101, dtype=np.float32)
    start, end = np.float32(located.panel.track)
    points = start + fractions[:, None] * (end - start)
    bands = np.vstack([points + [0, offset] for offset in (-2, -1, 0, 1, 2)])
    pixels = located.project(bands).reshape(5, -1, 2)
    if not np.isfinite(pixels).all():
        return None
    width, height = image.size
    if np.any(pixels < 0) or np.any(pixels >= [width - 1, height - 1]):
        return None
    rgb = np.asarray(image.convert("RGB"))
    sampled = cv2.remap(rgb, pixels[:, :, 0], pixels[:, :, 1], cv2.INTER_LINEAR)
    red, green, blue = sampled.astype(np.int16).transpose(2, 0, 1)
    marked = (green > 66) & (blue > 80) & (green - red > 33)
    filled = marked.sum(axis=0) >= 2
    if not filled[:6].any():
        return 0
    for index in range(5, 97):
        if not filled[index:index + 5].any():
            return index
    return 100


def _panel_disparities(images, feature_pixels, target_pixels, baseline):
    """Fit the disparity of the panel from direct left/right image matches."""
    left, right = images
    pixels = np.unique(np.rint(feature_pixels).astype(int), axis=0)
    pixels = pixels[::max(1, len(pixels)//200)]
    maximum = min(110, int(K[0, 0] * baseline / .16))
    pairs = []
    for x, y in pixels:
        u = match_stereo_pixel(left, right, x, y, maximum)
        if u is not None:
            pairs.append((x, y, u))
    if len(pairs) < 10:
        return None
    matches = np.asarray(pairs, dtype=np.float32)
    xy = matches[:, :2].copy()
    uv = xy.copy()
    uv[:, 0] = matches[:, 2]
    _, kept = cv2.estimateAffine2D(xy, uv, method=cv2.RANSAC, ransacReprojThreshold=1.2)
    if kept is None:
        return None
    good = kept.ravel().astype(bool)
    if good.sum() < max(10, .55*len(matches)) or min(np.ptp(xy[good], axis=0)) < 27:
        return None
    design = np.column_stack((xy[good], np.ones(good.sum())))
    model = np.linalg.lstsq(design, xy[good, 0]-uv[good, 0], rcond=None)[0]
    return np.column_stack((target_pixels, np.ones(len(target_pixels)))) @ model


def stereo_menu_geometry(pair, world=False, slider_fraction=.5, control=None):
    """Estimate a visible menu plane and target in the virtual headset's axes."""
    if pair.size != (1280, 720):
        return {"status": "stereo_unavailable"}
    width, height = 640, 720
    panels = (*LAUNCH, *AUDIO) if control is None else tuple(
        panel for panel in (*LAUNCH, *CAMERA) if control in panel.controls
    )
    left = locate(pair.crop((0, 0, width, height)), panels)
    if left is None:
        return {"status": "panel_unrecognized_in_left_eye"}
    try:
        images, _, pose, baseline = stereo_context(pair)
    except ValueError:
        images, pose, baseline = None, None, BASELINE
    if world and pose is None:
        return {"status": "menu_world_pose_unavailable", "panel": left.panel.name}
    keys, _ = reference(left.panel.image)
    samples = np.float32([key.pt for key in keys])
    center = samples.mean(axis=0) + left.panel.origin
    spread = np.ptp(samples, axis=0) * .18
    points = [center, center + [spread[0], 0], center + [0, spread[1]]]
    if control is not None:
        control_name = control
        points.append(left.panel.controls[control])
    elif left.panel.track:
        control_name = "ui_menu_slider_center" if slider_fraction == .5 else "ui_menu_slider"
        start, end = np.asarray(left.panel.track, dtype=float)
        points.append(start + (end-start) * slider_fraction)
    else:
        control_name = "audio_tab"
        points.append(left.panel.controls["audio"])
    a = left.project(points)
    disparity = _panel_disparities(images, left.project(samples + left.panel.origin), a, baseline) if images else None
    if disparity is None:
        right = locate(pair.crop((width, 0, 2 * width, height)), (left.panel,))
        if right is None:
            return {"status": "panel_unrecognized_in_right_eye", "panel": left.panel.name}
        b = right.project(points)
        disparity = a[:, 0] - b[:, 0]
        vertical_error = float(np.max(np.abs(a[:, 1] - b[:, 1])))
        middle = (a + b) / 2
    else:
        middle = a.copy()
        middle[:, 0] -= disparity / 2
        vertical_error = 0.
    if not np.isfinite(disparity).all() or np.any(disparity < 3) or vertical_error > 5:
        return {"status": "stereo_alignment_unverified", "panel": left.panel.name,
                "vertical_error_px": round(vertical_error, 2)}
    depth = (baseline * K[0, 0]) / disparity
    positions = np.column_stack(((2 * middle[:, 0] / width - 1) * depth,
                                 (1 - 2 * middle[:, 1] / height) * depth, -depth))
    normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
    magnitude = np.linalg.norm(normal)
    if magnitude < .00005:
        return {"status": "panel_plane_unverified", "panel": left.panel.name}
    tilt = np.degrees(np.arccos(np.clip(abs(normal[2]) / magnitude, 0, 1)))
    result = {"status": "measured", "panel": left.panel.name, "control": control_name,
              "target_head_xyz_m": positions[-1].astype(float).round(3).tolist(),
              "panel_tilt_deg": round(float(tilt), 1), "vertical_error_px": round(vertical_error, 2),
              "eye_spacing_m": baseline, "eye_spacing_source": "rendered" if pose is not None else "configured"}
    if pose is not None:
        pixels = np.column_stack((a, np.ones(len(a))))
        rays = pixels @ np.linalg.inv(K).T
        projected = (rays * depth[:, None]) @ pose[:3, :3].T + pose[:3, 3]
        plane = np.cross(projected[1] - projected[0], projected[2] - projected[0])
        plane /= np.linalg.norm(plane)
        result.update(panel_origin_world_xyz_m=projected[0].tolist(), panel_normal_world=plane.tolist(),
                      target_world_xyz_m=projected[-1].tolist(),
                      target_left_image_xy=(a[-1] / [width-1, height-1]).tolist())
        hitbox = (left.panel.hitboxes or {}).get(control_name)
        if hitbox is not None:
            result["target_left_hitbox_px"] = left.project(hitbox).tolist()
    return result


def tab_highlight(image, x, y):
    patch = np.asarray(image.convert("RGB").crop((x-6, y-5, x+7, y+6)), dtype=float)
    red, green, blue = patch.transpose(2, 0, 1)
    return ((green > 84) & (blue > 92) & (green-red > 55)).mean() > .29


class VolumeWorkflow:
    """Open Audio, aim locally, then verify UI/Menu volume by slider readback."""

    def __init__(self, snapshot, menu, view, wrist):
        self.snapshot = snapshot
        self.menu = menu
        self.view = view
        self.wrist = wrist
        self.probes = 0
        self.stage = "find_panel"
        self.frame_id = None
        self.source = None
        self.click_sent = False
        self.last_probe = None

    async def frame(self, panels, after=0, timeout=1.8):
        end = monotonic() + timeout
        while monotonic() < end:
            image, meta = await self.snapshot()
            if meta["frame_id"] > after:
                self.frame_id = after = meta["frame_id"]
                self.source = meta.get("source")
                found = await asyncio.to_thread(locate, image, panels)
                if found is not None:
                    return image, found
            await asyncio.sleep(.07)
        return None

    def target_for(self, located, percent):
        if located.panel.name == "launch_pad":
            return located.project([located.panel.controls["audio"]])[0]
        start, end = np.float32(located.panel.track)
        offset = 0 if self.source == "game_mirror" else 5
        point = start + percent / 100 * (end - start) + [0, offset]
        return located.project([point])[0]

    @staticmethod
    def on_target(delta, launch):
        return np.all(np.abs(delta) <= ([18, 18] if launch else [8, 4]))

    async def center(self, current, percent, panels, minimum_altitude):
        for _ in range(2):
            image, located = current
            altitude = self.target_for(located, percent)[1] / image.height
            if minimum_altitude <= altitude <= .90:
                return current
            gain = 55 if self.source == "game_mirror" else 95
            pitch = float(np.clip((.67 - altitude) * gain, -30, 30))
            await self.view(pitch)
            current = await self.frame(panels, after=self.frame_id, timeout=2.8)
            if current is None:
                return None
        return current

    async def stable_hit(self, panels, percent, screen, launch):
        sample = await self.frame(panels, after=self.frame_id)
        if sample is None:
            return False
        image, located = sample
        refreshed = self.target_for(located, percent)
        if np.max(np.abs(refreshed - screen)) > 5:
            return False
        tip = await asyncio.to_thread(laser_tip, image)
        if tip is None or not self.on_target(np.asarray(tip) * image.size - refreshed, launch):
            return False
        return not launch or tab_highlight(image, *refreshed)

    async def aim(self, current, percent):
        image, located = current
        origin_name = located.panel.name
        width, height = image.size
        size = np.array([width, height])
        screen = self.target_for(located, percent)
        if not np.all((screen > [8, 35]) & (screen < [width - 8, height - 12])):
            return False
        target = screen / size
        launch = origin_name == "launch_pad"
        panels = LAUNCH if launch else AUDIO
        aim_offset = GAME_MIRROR_AIM_OFFSET if self.source == "game_mirror" else located.panel.aim_offset
        nominal = np.clip(target - aim_offset, .005, .995)
        offsets = iter(((0, 0), (0, -.055), (0, .065), (-.07, 0), (.07, 0), (0, -.13)))
        seed = nominal + next(offsets)
        previous = image
        for _ in range(8):
            command = np.clip(seed, .002, .998)
            self.probes += 1
            await self.menu("aim", *map(float, command))
            sample = await self.frame(panels, after=self.frame_id, timeout=2.8)
            if sample is None:
                return False
            image, located = sample
            screen = self.target_for(located, percent)
            target = screen / size
            tip = await asyncio.to_thread(laser_tip, image)
            if tip is None:
                tip = await asyncio.to_thread(laser_tip, image, previous, target)
            previous = image
            self.last_probe = dict(aim=command.tolist(), target=target.tolist(), tip=tip)
            if tip is not None:
                delta = screen - np.asarray(tip) * size
                close = self.on_target(delta, launch)
                if (close and (not launch or tab_highlight(image, *screen))
                        and await self.stable_hit(panels, percent, screen, launch)):
                    return True
                if not close:
                    corrected = np.clip(command + np.clip(delta / size * .9, -.11, .11), .002, .998)
                    if np.linalg.norm(corrected - command) > .003:
                        seed = corrected
                        continue
            offset = next(offsets, None)
            if offset is None:
                break
            seed = nominal + offset
        return False

    async def run(self, percent):
        started = monotonic()

        def result(status, **values):
            if self.last_probe is not None and status not in ("verified", "already_set"):
                values["last_probe"] = self.last_probe
            return dict(status=status, step=self.stage, frame_id=self.frame_id,
                        click_sent=self.click_sent, probes=self.probes,
                        seconds=round(monotonic() - started, 2), **values)

        current = await self.frame((*LAUNCH, *AUDIO), timeout=.9)
        if current is None:
            await self.view(-20)
            current = await self.frame((*LAUNCH, *AUDIO), after=self.frame_id or 0, timeout=.9)
        if current is None:
            await self.wrist()
            await self.menu("toggle")
            current = await self.frame((*LAUNCH, *AUDIO))
        if current is None:
            return result("panel_unrecognized")

        if current[1].panel.name == "launch_pad":
            self.stage = "open_audio"
            if self.source != "game_mirror":
                await self.wrist()
                current = await self.frame(LAUNCH, after=self.frame_id, timeout=2.8)
                if current is None:
                    return result("audio_tab_not_visible")
            current = await self.center(current, percent, LAUNCH, .55)
            if current is None:
                return result("audio_tab_not_visible")
            if not await self.aim(current, percent):
                return result("audio_tab_not_confirmed")
            self.click_sent = True
            await self.menu("click")
            current = await self.frame(AUDIO, after=self.frame_id, timeout=2.8)
            if current is None:
                return result("audio_open_unverified")

        self.stage = "set_ui_menu_volume"
        before = fill_percent(*current)
        if before is None:
            return result("slider_not_visible")
        if abs(before - percent) <= 4:
            return result("already_set", estimated_percent=before)
        current = await self.center(current, percent, AUDIO, .62)
        if current is None:
            return result("slider_not_visible", previous_estimated_percent=before)
        centered = fill_percent(*current)
        if centered is not None:
            before = centered
        if not await self.aim(current, percent):
            return result("slider_aim_unverified", previous_estimated_percent=before)
        self.click_sent = True
        await self.menu("click")
        for _ in range(3):
            current = await self.frame(AUDIO, after=self.frame_id)
            if current is None:
                break
            after = fill_percent(*current)
            if after is not None and abs(after - before) >= 4:
                status = "verified" if abs(after - percent) <= 6 else "changed_outside_tolerance"
                return result(status, previous_estimated_percent=before, estimated_percent=after)
        return result("slider_effect_unverified", previous_estimated_percent=before)
