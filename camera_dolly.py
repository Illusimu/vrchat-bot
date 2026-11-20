"""Send player-local camera endpoints to VRChat's camera dolly."""
import json
import math
import socket
import urllib.request
import psutil

from pythonosc.osc_packet import OscPacket, ParseError
from pythonosc.osc_message_builder import OscMessageBuilder


class CameraDolly:
    def __init__(self):
        self.url = self.find_client()
        self.original_mode, self.original_stream = self.mode()
        self.input = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.input.bind(("127.0.0.1", 9001))
        except OSError:
            self.input.close()
            raise ValueError("Camera feedback port 9001 is occupied")
        self.input.setblocking(False)
        self.last = None
        self.phase = None
        self.send("/usercamera/Mode", 2)
        self.send("/usercamera/Streaming", False)

    @staticmethod
    def find_client():
        for proc in psutil.process_iter(["name"]):
            if (proc.info["name"] or "").lower() != "vrchat.exe":
                continue
            for conn in proc.net_connections(kind="tcp"):
                if conn.status != psutil.CONN_LISTEN:
                    continue
                url = f"http://127.0.0.1:{conn.laddr.port}"
                try:
                    with urllib.request.urlopen(url + "/?HOST_INFO", timeout=.4) as reply:
                        if "VRChat" in json.load(reply).get("NAME",""):
                            return url
                except (OSError, ValueError):
                    continue
        raise ValueError("Enable OSC in VRChat to connect its stream camera")

    def mode(self):
        with urllib.request.urlopen(self.url + "/usercamera", timeout=.6) as reply:
            info = json.load(reply)["CONTENTS"]
        return info["Mode"]["VALUE"][0], info["Streaming"]["VALUE"][0]

    def observe(self):
        from vr_agent import observe
        if self.mode() != (2, False):
            raise ValueError("Waiting for VRChat Stream mode with Spout off")
        return observe(stream=True)

    def send(self, address, value):
        msg = OscMessageBuilder(address=address)
        msg.add_arg(value)
        self.input.sendto(msg.build().dgram, ("127.0.0.1", 9000))

    def close(self):
        self.stop()
        self.send("/usercamera/Mode", self.original_mode)
        self.send("/usercamera/Streaming", self.original_stream)
        self.input.close()

    def stop(self):
        if self.phase:
            self.send("/dolly/Play", False)
            self.last = None
            self.phase = None

    def events(self):
        for _ in range(256):
            try:
                data = self.input.recv(65535)
            except BlockingIOError:
                break
            try:
                for item in OscPacket(data).messages:
                    if item.message.address == "/dolly/Play" and item.message.params:
                        yield bool(item.message.params[0])
            except ParseError:
                continue

    def start(self, values, now):
        self.idle()
        if self.phase:
            self.send("/dolly/Play", False)
        self.goal = tuple(values)
        # The first shot establishes a player-local reference. Subsequent moves
        # start at the last endpoint acknowledged by this playback session.
        start = self.last or self.goal
        angle = max(abs((b-a+180)%360-180) for a,b in zip(start[3:],values[3:]))
        self.duration = max(.4, math.dist(start[:3],values[:3])/4, angle/150)
        def point(p):
            return dict(IsLocal=True, Position=dict(zip("XYZ",p[:3])),
                        Rotation=dict(zip("XYZ",p[3:])), Zoom=55., Duration=self.duration, Index=-1)
        hold = point(values)
        hold["Duration"] = 60.
        self.path = json.dumps([point(start),hold,point(values)], separators=(",",":"))
        self.phase = "stopping"
        self.started = now

    def idle(self):
        for playing in self.events():
            if self.phase == "holding" and not playing:
                self.phase = None
                self.last = None

    def tick(self, now):
        for playing in self.events():
            if self.phase == "waiting" and playing:
                self.phase = "moving"
                self.play_started = now
            elif self.phase == "moving" and not playing:
                self.phase = None
                self.last = None
                return "camera playback interrupted; position unconfirmed"
        if self.phase == "stopping" and now-self.started >= .2:
            self.send("/dolly/Import", self.path)
            self.phase = "import"
        elif self.phase == "import" and now-self.started >= .6:
            self.send("/dolly/Play", True)
            self.phase = "waiting"
        elif self.phase == "moving" and now-self.play_started >= self.duration:
            self.phase = "holding"
            self.last = self.goal
            return "camera move played; verify endpoint visually during the 60-second hold"
        if now-self.started > self.duration+4:
            self.stop()
            return "camera playback unconfirmed; check OSC and Camera Dolly"
