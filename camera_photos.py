"""Locate and verify still photos written by VRChat."""

import ctypes
from datetime import date, timedelta
import json
from pathlib import Path

from PIL import Image


def photo_folder():
    config = Path.home() / "AppData/LocalLow/VRChat/VRChat/config.json"
    if config.is_file():
        output = json.loads(config.read_text(encoding="utf-8-sig")).get("picture_output_folder")
        if output:
            return Path(output).expanduser()
    pictures = ctypes.create_unicode_buffer(260)
    if ctypes.windll.shell32.SHGetFolderPathW(None, 0x27, None, 0, pictures):
        raise OSError("Windows did not return the Pictures folder")
    return Path(pictures.value) / "VRChat"


def photo_paths(folder):
    today = date.today()
    months = {f"{today + timedelta(days=offset):%Y-%m}" for offset in (-1, 0, 1)}
    return {path for directory in (folder, *(folder / month for month in months))
            for path in directory.glob("VRChat_*.png") if path.is_file()}


def complete_photo(path):
    try:
        with Image.open(path) as photo:
            if photo.format != "PNG":
                return False
            photo.verify()
        return True
    except (OSError, SyntaxError):
        return False
