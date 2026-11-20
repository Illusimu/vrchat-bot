"""Read the active client's confirmed room identity without exposing its join link."""
import hashlib
import os
from pathlib import Path
import re
import threading

import psutil
import win32gui
import win32process

ROOM = re.compile(r"^Joining (wrld_[0-9a-fA-F-]{36}):(\S+)")
_lock = threading.Lock()
_session = None
_offset = 0
_pending = {}
_current = {}


def room_event(message, pending, current):
    if message.startswith("Entering Room: "):
        current.clear()
        pending.clear()
        pending["world_name"] = message.removeprefix("Entering Room: ")
    elif match := ROOM.match(message):
        current.clear()
        world, instance = match.groups()
        pending["world_id"] = world
        pending["instance_key"] = hashlib.sha256((world+":"+instance).encode()).hexdigest()[:24]
    elif message == "Successfully joined room" and "world_id" in pending:
        current.update(pending)
        current["scene_ready"] = False
        pending.clear()
    elif message.startswith("Finishing room transition") and "world_id" in current:
        current["scene_ready"] = True
    elif message in ("OnLeftRoom","OnDisconnected","Beginning room transition","Transitioning to room") or 'Resetting game flow because "Destination instance set"' in message:
        current.clear()
        pending.clear()


def world_context():
    global _session, _offset
    unknown = dict(state="unknown", coordinate_source="unavailable")
    with _lock:
        try:
            hwnd=win32gui.FindWindow(None,"VRChat")
            if not hwnd:
                return unknown
            _,pid=win32process.GetWindowThreadProcessId(hwnd)
            client=psutil.Process(pid)
            if client.name().lower()!="vrchat.exe":
                return unknown
            started=client.create_time()
            folder=Path(os.environ["USERPROFILE"])/"AppData/LocalLow/VRChat/VRChat"
            logs=[p for p in folder.glob("output_log*.txt") if p.stat().st_ctime>=started-3]
            if not logs:
                return unknown
            path=max(logs,key=lambda p:p.stat().st_ctime)
            identity=(started,path)
            if identity!=_session or path.stat().st_size<_offset:
                _session,_offset=identity,0
                _pending.clear()
                _current.clear()
            with path.open("rb") as log:
                log.seek(_offset)
                for line in log:
                    if not line.endswith(b"\n"):
                        break
                    _offset+=len(line)
                    if b"[Behaviour] " in line:
                        room_event(line.split(b"[Behaviour] ",1)[1].decode("utf-8","replace").strip(),_pending,_current)
        except (OSError,KeyError,psutil.Error):
            return unknown
        if not _current:
            return unknown
        return dict(state="joined",coordinate_source="unavailable",**_current)
