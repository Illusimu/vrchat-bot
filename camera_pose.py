"""Read the SV1 binary HUD from an undistorted RGB eye texture."""
import numpy as np


def read_eye(rgb):
    for flipped, pixels in ((False,rgb),(True,rgb[::-1])):
        result = _decode(pixels)
        if result is not None:
            result["flipped_patch"] = flipped
            result["palms"] = {}
            for side, left, magic in (("left", .25, 0x50414C4C), ("right", .46, 0x50414C52)):
                data = _packet(pixels, left, .08, 14, magic)
                if data is not None:
                    axes = data[3:].reshape(3, 3).T
                    if np.allclose(axes.T @ axes, np.eye(3), atol=.005) and np.linalg.det(axes) > .99:
                        result["palms"][side] = (data[:3], axes)
            return result
    return None


def _packet(rgb, left, span_y, rows, magic):
    height, width = rgb.shape[:2]
    x = ((left + (np.arange(32) + .5) * .20 / 32) * width).astype(int)
    y = ((.06 + (np.arange(rows) + .5) * span_y / rows) * height).astype(int)
    intensity = rgb[y[:, None], x].mean(axis=-1)
    words = np.packbits(intensity > 128, axis=1, bitorder="little").copy().view("<u4").ravel()
    if words[0] != magic:
        return None
    checksum = 2166136261
    for word in words[:-1]:
        checksum = ((checksum ^ int(word)) * 16777619) & 0xffffffff
    values = words[1:-1].view("<f4")
    if checksum != words[-1] or not np.isfinite(values).all():
        return None
    return values


def _decode(rgb):
    values = _packet(rgb, .04, .16, 28, 0xC35A76E1)
    if values is None:
        return None
    screen, photo, eye = values[:7], values[7:14], values[14:17]
    axes = values[17:].reshape(3, 3).T
    if not np.allclose(axes.T @ axes, np.eye(3), atol=.005) or np.linalg.det(axes)<.99:
        return None
    if abs(np.linalg.norm(screen[3:])-1)>.01:
        return None
    # The viewer uses OpenCV camera coordinates (right, down, forward).
    flip = np.diag([1., -1., 1.])
    pose = np.eye(4)
    pose[:3, :3] = flip @ axes @ flip
    pose[:3, 3] = flip @ eye
    return dict(pose=pose, screen=screen.copy(), photo=photo.copy(),
                photo_available=bool(np.any(photo) and abs(np.linalg.norm(photo[3:])-1)<.01))
