"""Detect VRChat's visible pickup and release prompts."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def light_text(rgb):
    rgb = rgb.astype(np.float32)
    low = rgb.min(axis=2)
    spread = rgb.max(axis=2) - low
    return np.clip((low - 145) / 100, 0, 1) * np.clip((75 - spread) / 75, 0, 1)


@lru_cache(maxsize=1)
def equip_templates():
    path = Path(__file__).with_name("menu_templates") / "pickup_equip.png"
    with Image.open(path) as image:
        text = light_text(np.asarray(image.convert("RGB")))
    return tuple(cv2.resize(text, None, fx=scale, fy=scale) for scale in (.7, .85, 1., 1.17, 1.35))


@lru_cache(maxsize=1)
def angled_equip_templates():
    templates = []
    reference = equip_templates()[2]
    for scale in (.65, .73, .81, .9, 1.):
        text = cv2.resize(reference, None, fx=scale, fy=scale)
        padded = np.pad(text, 8)
        height, width = padded.shape
        for angle in (-14, -7, 0, 7, 14):
            turn = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            glyphs = cv2.warpAffine(padded, turn, (width, height))
            templates.append((glyphs > .24).astype(np.float32))
    return templates


def frame_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((862, 672)))


@lru_cache(maxsize=1)
def drop_templates():
    root = Path(__file__).with_name("menu_templates")
    templates = []
    for name in ("pickup_drop.png", "pickup_drop_italic.png"):
        with Image.open(root / name) as image:
            glyphs = (light_text(np.asarray(image.convert("RGB"))) > .22).astype(np.float32)
        templates.extend(cv2.resize(glyphs, None, fx=scale, fy=scale) for scale in (.8, .9, 1., 1.1, 1.25))
    return templates


@lru_cache(maxsize=2)
def rotated_drop_templates(indices, angles):
    templates = []
    for index in indices:
        glyphs = drop_templates()[index]
        height, width = glyphs.shape
        for angle in angles:
            turn = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            templates.append(cv2.warpAffine(glyphs, turn, (width, height)))
    return templates


def release_to_drop(path):
    """Find the visible end of the HUD label, which the wrist can partially occlude."""
    text = (light_text(frame_rgb(path)) > .22).astype(np.float32)
    angled = rotated_drop_templates((0, 1, 5, 6), (-4, 4))
    inverted = rotated_drop_templates((1, 2, 3, 6, 7, 8), (176, 180, 184))
    for region, templates, minimum_score in (
        (text, drop_templates(), .74), (text, angled, .68), (text[-272:], inverted, .74),
    ):
        for template in templates:
            _, score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCORR_NORMED))
            if score >= minimum_score:
                return True
    return False


def cyan_pixels(rgb):
    return (rgb[:, :, 0] < 105) & (rgb[:, :, 1] > 125) & (rgb[:, :, 2] > 155)


def pickup_hover(path, box=None):
    rgb = frame_rgb(path)
    text = light_text(rgb)
    cyan = cyan_pixels(rgb)
    x0, y0, x1, y1 = 0, 0, 862, 672
    if box is not None:
        left, top, width, height = box
        x0, y0 = max(0, round((left - .14) * 862)), max(0, round((top - .19) * 672))
        x1, y1 = min(862, round((left + width + .14) * 862)), min(672, round((top + height + .14) * 672))
    for template in equip_templates():
        _, match, _, (left, top) = cv2.minMaxLoc(cv2.matchTemplate(text, template, cv2.TM_CCOEFF_NORMED))
        if match < .64:
            continue
        height, width = template.shape
        surroundings = cyan[max(0, top - height):top + 2 * height, max(0, left - width):left + width]
        center = left + width / 2, top + height / 2
        if (x0 <= center[0] <= x1 and y0 <= center[1] <= y1
                and surroundings.any() and surroundings.mean() >= .012):
            return {"cue": "Equip", "image_point": [round((left + width / 2) / 862, 3),
                                                     round((top + height / 2) / 672, 3)]}
    if box is not None:
        left, top, width, height = box
        prop_cyan = cyan[max(0, round((top - .03) * 672)):round((top + height + .03) * 672),
                         max(0, round((left - .03) * 862)):round((left + width + .03) * 862)]
        if prop_cyan.mean() < .012:
            return None
        region = (text[y0:y1, x0:x1] > .18).astype(np.float32)
        for template in angled_equip_templates():
            height, width = template.shape
            if region.shape[0] < height or region.shape[1] < width:
                continue
            _, score, _, (left, top) = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCORR_NORMED))
            left, top = left + x0, top + y0
            if score >= .68:
                return {"cue": "Equip", "image_point": [round((left + width / 2) / 862, 3),
                                                         round((top + height / 2) / 672, 3)]}
    return None


def pickup_outline(before, after, box):
    """Find a new cyan outline tracing two edges of a selected prop after a wrist-only move."""
    before_rgb, after_rgb = frame_rgb(before), frame_rgb(after)
    new_cyan = cyan_pixels(after_rgb) & ~cyan_pixels(before_rgb)
    left, top, width, height = box
    x = max(0, round((left - .045) * 862))
    y = max(0, round((top - .045) * 672))
    right = round((left + width + .045) * 862)
    bottom = round((top + height + .045) * 672)
    area = new_cyan[y:bottom, x:right].astype(np.uint8)
    area = cv2.morphologyEx(area, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(area)
    top_edge = right_edge = bottom_edge = False
    for offset_x, offset_y, span_x, span_y, pixels in stats[1:count]:
        if pixels < 45:
            continue
        top_edge |= (span_x >= 862 * width * .30 and span_y <= 672 * height * .24
                     and y + offset_y <= 672 * (top + .09)
                     and x + offset_x + span_x >= 862 * (left + .55 * width))
        right_edge |= (span_y >= 672 * height * .13 and span_x <= 862 * width * .20
                       and x + offset_x >= 862 * (left + .70 * width)
                       and y + offset_y <= 672 * (top + .65 * height))
        bottom_edge |= (span_x >= 862 * width * .04 and span_y <= 672 * height * .13
                        and x + offset_x >= 862 * (left + .50 * width)
                        and y + offset_y >= 672 * (top + .82 * height))
    if right_edge and (top_edge or bottom_edge):
        return {"cue": "Equip outline", "image_point": [round(left + width / 2, 3), round(top + height / 2, 3)]}
    # A curved outline can form one large component or have a side covered by the wrist.
    x0, x1 = round(left * 862), round((left + width) * 862)
    y0, y1 = round(top * 672), round((top + height) * 672)
    dx, dy = max(1, round((x1 - x0) * .22)), max(1, round((y1 - y0) * .20))
    middle_rows = new_cyan[y0 + dy:y1 - dy]
    left_side = middle_rows[:, max(0, x0 - dx):x0 + dx].any(axis=1).mean()
    right_side = middle_rows[:, x1 - dx:x1 + dx].any(axis=1).mean()
    middle_columns = new_cyan[:, x0 + dx:x1 - dx]
    top_cap = middle_columns[max(0, y0 - dy):y0 + dy].any(axis=0).mean()
    bottom_cap = middle_columns[y1 - dy:y1 + dy].any(axis=0).mean()
    inside = middle_columns[y0 + dy:y1 - dy].mean()
    side = max(left_side, right_side)
    pale_base = False
    if side >= .34 and bottom_cap < .40 and inside <= .18:
        # On bright wood the blue glow at the base can be too pale to count
        # as cyan. Allow an imprecise box, but require a coherent rim near
        # its base rather than combining glow from different heights.
        band = np.s_[max(0, y1 - round(dy * .5)):y1 + round(dy * .7), x0 + dx:x1 - dx]
        prior = before_rgb[band].astype(np.int16)
        red, green, blue = after_rgb[band].astype(np.int16).transpose(2, 0, 1)
        gain = (blue - red) - (prior[..., 2] - prior[..., 0])
        glow = (blue > 155) & (blue - red > 17) & (green - red > 7) & (gain > 16)
        rim = cv2.dilate(glow.astype(np.uint8), np.ones((25, 1), np.uint8))
        # When a wrist covers more of the side, require a wider base rim.
        pale_base = rim.mean(axis=1).max() >= max(.65, 1.06 - side)
    if inside <= .18 and (pale_base or (side >= .45 and max(top_cap, bottom_cap) >= .40)):
        return {"cue": "Equip outline", "image_point": [round(left + width / 2, 3), round(top + height / 2, 3)]}
    return None


def saturated_pixels(hsv):
    return (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 65)


def hue_mask(hsv, hue):
    distance = np.abs(hsv[:, :, 0].astype(np.int16) - hue)
    return saturated_pixels(hsv) & (np.minimum(distance, 180 - distance) <= 4)


def prop_color_reference(path, box, *, outline_frame=None):
    """Find a distinctive color in a tight selection or beside a verified pickup outline."""
    rgb = frame_rgb(path)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = hsv.shape[:2]
    x, y, w, h = box
    rows = slice(round(y * height), round((y + h) * height))
    columns = slice(round(x * width), round((x + w) * width))
    crop = hsv[rows, columns]
    hues = crop[:, :, 0][saturated_pixels(crop)]
    if not hues.size:
        return None
    hue = int(np.argmax(np.bincount(hues, minlength=180)))
    mask = hue_mask(crop, hue)
    broad_selection = mask.mean() < .23
    if (mask.sum() < .65 * hues.size
            or (broad_selection and (outline_frame is None or mask.mean() < .10))):
        return None
    saturation_low, saturation_high = np.percentile(crop[:, :, 1][mask], [10, 90])
    saturation_range = max(150, saturation_low - 14), min(255, saturation_high + 14)
    saturation = hsv[:, :, 1]
    component_mask = hue_mask(hsv, hue) & (saturation_range[0] <= saturation) & (saturation <= saturation_range[1])
    component_mask = cv2.morphologyEx(component_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask)
    overlap = np.bincount(labels[rows, columns][mask], minlength=count)
    overlap[0] = 0
    component = int(overlap.argmax())
    prop_x, prop_y, prop_width, prop_height, area = stats[component]
    if (component == 0 or min(prop_width, prop_height) < 12 or overlap[component] < .65 * mask.sum()
            or area > 3 * overlap[component] or area < .28 * prop_width * prop_height):
        return None
    if broad_selection:
        new_cyan = cyan_pixels(frame_rgb(outline_frame)) & ~cyan_pixels(rgb)
        prop_right = prop_x + prop_width
        margin = max(2, round(.28 * prop_width))
        left = new_cyan[prop_y:prop_y + prop_height, max(0, prop_x - margin):prop_x]
        right = new_cyan[prop_y:prop_y + prop_height, prop_right:prop_right + margin]
        if max(left.any(axis=1).mean(), right.any(axis=1).mean()) < .20:
            return None
    return hue, int(area), prop_width / prop_height, saturation_range


def observe_prop_color(path, reference):
    """Locate a colored region resembling the prop, without inferring surface contact."""
    hue, reference_area, aspect, (saturation_low, saturation_high) = reference
    hsv = cv2.cvtColor(frame_rgb(path), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    colors = hue_mask(hsv, hue) & (saturation_low <= saturation) & (saturation <= saturation_high)
    mask = cv2.morphologyEx(colors.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [(x, y, w, h) for x, y, w, h, pixels in stats[1:]
                  if .24 * reference_area <= pixels <= 3 * reference_area
                  and pixels >= .28 * w * h and min(w, h) >= 12
                  and .4 * aspect <= w / h <= 2 * aspect]
    if len(candidates) != 1:
        return {"color_match": "ambiguous" if candidates else "not_visible"}
    height, width = hsv.shape[:2]
    box = [round(float(value) / size, 3) for value, size in zip(candidates[0], (width, height, width, height))]
    return {"color_match": "visible", "box": box}


def refine_pickup_box(path, selection):
    """Estimate the outline search region from a unique colored component in the selection."""
    reference = prop_color_reference(path, selection)
    if not reference or not (box := observe_prop_color(path, reference).get("box")):
        return None
    x, y, width, height = box
    # Colored contents can be smaller than the object's outline, particularly for transparent containers.
    left, top = max(0., x - .09 * width), max(0., y - .22 * height)
    right = min(1., x + 1.09 * width)
    bottom = min(1., y + 1.14 * height)
    return left, top, right - left, bottom - top


def expand_partial_pickup_box(path, selection):
    """Recover a unique color region that was partly inside the proposed target box."""
    x, y, width, height = selection
    left, top = max(0., x - .5 * width), max(0., y - height / 3)
    right, bottom = min(1., x + 1.5 * width), min(1., y + 4 * height / 3)
    candidate = refine_pickup_box(path, (left, top, right - left, bottom - top))
    if candidate is None:
        return None
    rx, ry, rw, rh = candidate
    overlap = max(0., min(x + width, rx + rw) - max(x, rx)) * max(0., min(y + height, ry + rh) - max(y, ry))
    return candidate if overlap >= .35 * width * height else None


def prop_texture_reference(path, selection):
    """Keep features from a colorful interior region of the verified pickup."""
    rgb = frame_rgb(path)
    height, width = rgb.shape[:2]
    x, y, w, h = selection
    left, right = round((x + .10 * w) * width), round((x + .90 * w) * width)
    top, bottom = round((y + .06 * h) * height), round((y + .94 * h) * height)
    mask = np.zeros((height, width), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2HSV)
    mask[top:bottom, left:right] = saturated_pixels(hsv).astype(np.uint8) * 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points, descriptors = cv2.SIFT_create(nfeatures=2800).detectAndCompute(gray, mask)
    if descriptors is None or len(points) < 14:
        return None
    center = np.array([(x + w / 2) * width, (y + h / 2) * height])
    return np.asarray([point.pt for point in points]), descriptors, center


def observe_prop_texture(path, reference):
    """Estimate translation, scale, and rotation from repeated prop features."""
    source, descriptors, center = reference
    rgb = frame_rgb(path)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points, current = cv2.SIFT_create(nfeatures=2800).detectAndCompute(gray, None)
    if current is None or len(current) < 2:
        return None
    pairs = cv2.BFMatcher().knnMatch(descriptors, current, k=2)
    matches = [best for best, other in pairs if best.distance < .74 * other.distance]
    if len(matches) < 12:
        return None
    original = source[[match.queryIdx for match in matches]]
    target = np.asarray([points[match.trainIdx].pt for match in matches])
    transform, inliers = cv2.estimateAffinePartial2D(original, target, ransacReprojThreshold=3)
    if transform is None or inliers is None:
        return None
    verified = original[inliers.ravel().astype(bool)]
    distinct = len(np.unique(np.round(verified), axis=0))
    if distinct < 10 or len(verified) < .55 * len(matches) or min(np.ptp(verified, axis=0)) < 14:
        return None
    moved = transform[:, :2] @ center + transform[:, 2]
    return {
        "current_center": [round(float(value), 3) for value in moved / [width, height]],
        "center_displacement": [round(float(value), 3) for value in (moved - center) / [width, height]],
        "scale": round(float(np.linalg.norm(transform[:, 0])), 3),
        "rotation_deg": round(float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))), 1),
        "matched_points": distinct,
    }


def movement_follows_wrist(before, after, wrist_delta):
    """Test for matching 2D motion on lateral or vertical wrist-only moves."""
    if not before or not after or abs(wrist_delta[2]) > .02:
        return None
    first, second = np.asarray(before), np.asarray(after)
    displacement = (second[:2] + second[2:] / 2) - (first[:2] + first[2:] / 2)
    expected = np.array([wrist_delta[0], -wrist_delta[1]])
    active = np.abs(expected) >= .025
    if not active.any():
        return None
    return bool(np.all(displacement[active] * np.sign(expected[active]) >= .025))


def prop_shifted_on_grip_close(before, after):
    """Require a large shift and little overlap to distinguish pickup from hand occlusion."""
    if not before or not after:
        return False
    first, second = np.asarray(before), np.asarray(after)
    overlap_start = np.maximum(first[:2], second[:2])
    overlap_end = np.minimum(first[:2] + first[2:], second[:2] + second[2:])
    overlap = np.maximum(overlap_end - overlap_start, 0).prod()
    smaller_area = min(first[2:].prod(), second[2:].prod())
    center_shift = np.abs((second[:2] + second[2:] / 2) - (first[:2] + first[2:] / 2))
    return bool(overlap < .25 * smaller_area and np.any(center_shift >= np.maximum(.05, .60 * first[2:])))


def apparent_prop_depth(pickup_box, current_box):
    """Flag a change in apparent scale when both boxes fit inside the view."""
    if not pickup_box or not current_box:
        return None
    for x, y, width, height in (pickup_box, current_box):
        if min(x, y, 1 - x - width, 1 - y - height) <= .01:
            return None
    width_ratio = current_box[2] / pickup_box[2]
    height_ratio = current_box[3] / pickup_box[3]
    if min(width_ratio, height_ratio) >= 1.20:
        return "larger_than_pickup"
    if max(width_ratio, height_ratio) <= 1 / 1.20:
        return "smaller_than_pickup"
    return None


def pickup_depth_correction(pickup_box, current_box, wrist_depth):
    """Estimate a same-depth correction from matched prop scale and wrist distance."""
    if not apparent_prop_depth(pickup_box, current_box) or not .25 <= wrist_depth <= .85:
        return None
    width_ratio = current_box[2] / pickup_box[2]
    height_ratio = current_box[3] / pickup_box[3]
    if max(width_ratio, height_ratio) / min(width_ratio, height_ratio) > 1.22:
        return None
    # Use the smaller change to leave margin for occlusion and wrist-to-prop offset.
    ratio = min((width_ratio, height_ratio), key=lambda value: abs(1 - value))
    correction = wrist_depth * (ratio - 1)
    return round(correction, 2)


def support_height_cue(pickup_box, current_box, *, previous_box=None, wrist_delta=None):
    """Estimate when further lowering may be unnecessary from a fixed view."""
    if not pickup_box or not current_box:
        return None

    def bottom(box):
        return box[1] + box[3]

    if wrist_delta and previous_box:
        right, up, forward = wrist_delta
        if (up <= -.05 and abs(right) <= .02 and abs(forward) <= .02
                and .75 <= current_box[2] / previous_box[2] <= 1.3
                and abs(bottom(current_box) - bottom(previous_box)) <= .02):
            return "prop_stopped_following_downward"
    if (apparent_prop_depth(pickup_box, current_box) is None
            and .75 <= current_box[2] / pickup_box[2] <= 1.3
            and abs(bottom(current_box) - bottom(pickup_box)) <= .045):
        return "near_original_support_height"
    return None


def game_mirror_surface(box, views):
    """Estimate a horizontal surface from a rotation reference and two different head positions."""
    with Image.open(views[0][0]) as image:
        width, height = image.size
    sift = cv2.SIFT_create(nfeatures=3000)
    matcher = cv2.BFMatcher()
    frames = []
    for path, state in views:
        if state.get("source") != "game_mirror" or not state.get("commanded_poses"):
            return None
        frame = frame_rgb(path)
        if frame.shape[:2] != (height, width):
            return None
        keys, descriptors = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None)
        if descriptors is None or len(keys) < 20:
            return None
        position = np.asarray(state["commanded_poses"][0][:3])
        orientation = Rotation.from_quat(state["commanded_poses"][0][3:], scalar_first=True)
        world_to_camera = np.diag([1, -1, -1]) @ orientation.as_matrix().T
        frames.append((np.array([key.pt for key in keys]), descriptors, position, world_to_camera))
    start, turned, lowered = frames
    if np.linalg.norm(start[2] - turned[2]) > .025 or np.linalg.norm(start[2] - lowered[2]) < .10:
        return None

    def correspondences(other):
        reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(other[1], start[1], k=2)
                   if len(pair) == 2 and pair[0].distance < .77 * pair[1].distance}
        matches = [pair[0] for pair in matcher.knnMatch(start[1], other[1], k=2)
                   if len(pair) == 2 and pair[0].distance < .77 * pair[1].distance
                   and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
        return start[0][[match.queryIdx for match in matches]], other[0][[match.trainIdx for match in matches]]

    def intrinsics(parameters):
        focal, center_x, center_y = parameters
        return np.array([[focal, 0, center_x], [0, focal, center_y], [0, 0, 1.]])

    original_pixels, turned_pixels = correspondences(turned)
    if len(original_pixels) < 30:
        return None
    rotation = turned[3] @ start[3].T

    def rotation_residual(parameters):
        camera = intrinsics(parameters)
        rays = np.column_stack((original_pixels, np.ones(len(original_pixels)))) @ np.linalg.inv(camera).T
        projected = rays @ rotation.T @ camera.T
        return (projected[:, :2] / projected[:, 2:] - turned_pixels).ravel()

    fit = least_squares(rotation_residual, [.67 * width, .5 * width, .52 * height],
                        bounds=([.4 * width, .4 * width, .4 * height], [1.2 * width, .6 * width, .65 * height]),
                        loss="soft_l1", f_scale=1.5, max_nfev=40)
    error = np.linalg.norm(rotation_residual(fit.x).reshape(-1, 2), axis=1)
    if not fit.success or np.sum(error < 3) < 25 or np.median(error) > 3:
        return None
    camera = intrinsics(fit.x)

    reference_pixels, lowered_pixels = correspondences(lowered)
    left, top, box_width, box_height = box
    normalized = reference_pixels / [width, height]
    inside = np.all((normalized >= [left, top]) & (normalized <= [left + box_width, top + box_height]), axis=1)
    reference_pixels, lowered_pixels = reference_pixels[inside], lowered_pixels[inside]
    if len(reference_pixels) < 5:
        return None

    def projection(frame):
        return camera @ np.column_stack((frame[3], -frame[3] @ frame[2]))

    homogeneous = cv2.triangulatePoints(projection(start), projection(lowered),
                                        reference_pixels.T, lowered_pixels.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        points = homogeneous[:, :3] / homogeneous[:, 3:]
        valid = np.isfinite(points).all(axis=1)
        for frame, pixels in ((start, reference_pixels), (lowered, lowered_pixels)):
            camera_points = (points - frame[2]) @ frame[3].T
            projected = camera_points @ camera.T
            pixel_error = np.linalg.norm(projected[:, :2] / projected[:, 2:] - pixels, axis=1)
            valid &= (camera_points[:, 2] > .2) & (pixel_error < 2.5)
    points = points[valid]
    if len(points) < 5:
        return None
    measured_height = float(np.median(points[:, 1]))
    on_surface = points[np.abs(points[:, 1] - measured_height) <= .035]
    if len(on_surface) < max(5, .7 * len(points)) or np.ptp(on_surface[:, [0, 2]], axis=0).max() < .15:
        return None

    center = [(left + box_width / 2) * width, (top + box_height / 2) * height, 1.]
    direction = start[3].T @ np.linalg.solve(camera, center)
    if direction[1] >= -.03:
        return None
    scale = (measured_height - start[2][1]) / direction[1]
    if not .2 <= scale <= 4:
        return None
    forward = start[3][2].copy()
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    observed_depths = (on_surface - start[2]) @ forward
    target_depth = float(scale * direction @ forward)
    if not observed_depths.min() - .12 <= target_depth <= observed_depths.max() + .12:
        return None
    return {"surface_height_m": round(measured_height, 2), "center_forward_m": round(target_depth, 2),
            "observed_forward_span_m": np.round([observed_depths.min(), observed_depths.max()], 2).tolist(),
            "matching_points": len(on_surface), "rotation_median_error_px": round(float(np.median(error)), 2)}
