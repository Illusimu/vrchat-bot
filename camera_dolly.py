"""Send player-local camera endpoints to VRChat's camera dolly."""
from array import array
import json
import math
import socket
import time
from urllib.error import HTTPError
import urllib.request
import psutil

from PIL import Image
from pythonosc.osc_packet import OscPacket, ParseError
from pythonosc.osc_message_builder import OscMessageBuilder


class CameraDolly:
    def __init__(self, *, spout=False):
        self.url = self.find_client()
        self.original_mode, self.original_stream = self.mode()
        self.spout = spout
        self.receiver = None
        self.buffer = None
        if spout:
            if self.original_mode == 0 and not self.original_stream:
                raise ValueError("Open Photo Camera in VRChat before connecting the Spout feed")
            try:
                import SpoutGL
            except ImportError as error:
                raise ValueError("Install SpoutGL in the pilot environment to use its game-mirror camera") from error
            self.receiver_type = SpoutGL.SpoutReceiver
        self.has_dolly = self.dolly_available(self.url)
        self.input = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if self.has_dolly:
                self.input.bind(("127.0.0.1", 9001))
        except OSError as error:
            self.input.close()
            raise ValueError("Camera feedback port 9001 is occupied") from error
        self.input.setblocking(False)
        self.last = None
        self.phase = None
        if not self.stream_ready(self.original_mode, self.original_stream):
            self.enable_stream()

    def enable_stream(self):
        self.send("/usercamera/Mode", 2)
        self.send("/usercamera/Streaming", self.spout)

    @staticmethod
    def dolly_available(url):
        try:
            with urllib.request.urlopen(url + "/dolly", timeout=.6) as reply:
                return "Play" in json.load(reply).get("CONTENTS", {})
        except HTTPError as error:
            if error.code != 404:
                raise
            return False

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
        return self.query_mode(self.url)

    def stream_ready(self, mode, streaming):
        if self.spout:
            # VRChat can keep publishing the fixed feed with the viewfinder closed.
            return streaming and mode in (0, 2)
        return mode == 2 and not streaming

    def zoom(self):
        with urllib.request.urlopen(self.url + "/usercamera/Zoom", timeout=.6) as reply:
            return float(json.load(reply)["VALUE"][0])

    @staticmethod
    def query_settings(url):
        with urllib.request.urlopen(url + "/usercamera", timeout=.6) as reply:
            return json.load(reply)["CONTENTS"]

    @staticmethod
    def query_mode(url):
        info = CameraDolly.query_settings(url)
        return info["Mode"]["VALUE"][0], info["Streaming"]["VALUE"][0]

    @staticmethod
    def query_flying(url):
        with urllib.request.urlopen(url + "/usercamera/Flying", timeout=.6) as reply:
            state = json.load(reply)
        if state.get("ACCESS", 0) & 3 != 3:
            raise ValueError("VRChat did not expose read/write flight controls.")
        return bool(state["VALUE"][0])

    @staticmethod
    def send_flying(enabled):
        message = OscMessageBuilder(address="/usercamera/Flying")
        message.add_arg(enabled)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
            output.sendto(message.build().dgram, ("127.0.0.1", 9000))

    def observe(self):
        if not self.stream_ready(*self.mode()):
            raise ValueError("Waiting for VRChat Stream mode and the selected capture source")
        if self.spout:
            return self.observe_spout()
        from vr_agent import observe
        return observe(stream=True)

    def observe_spout(self):
        # Call on a dedicated worker so the receiver keeps its graphics context.
        if self.receiver is None:
            receiver = self.receiver_type()
            receiver.__enter__()
            self.receiver = receiver
            self.receiver.setReceiverName("VRCSender1")
        sink = self.receiver
        deadline = time.monotonic() + .25
        while time.monotonic() < deadline:
            received = sink.receiveImage(self.buffer, 6408, False, 0)  # GL_RGBA
            if sink.isUpdated():
                self.size = (sink.getSenderWidth(), sink.getSenderHeight())
                self.buffer = array("B", [0]) * (self.size[0] * self.size[1] * 4)
            elif received and self.buffer and sink.isFrameNew():
                frame = Image.frombytes("RGBA", self.size, self.buffer.tobytes()).convert("RGB")
                if max(high for _, high in frame.getextrema()) > 16:
                    return frame
            time.sleep(.02)
        raise ValueError("Waiting for a fresh VRChat camera image over Spout")

    def close_receiver(self):
        if self.receiver is not None:
            self.receiver.__exit__(None, None, None)
            self.receiver = None
            self.buffer = None

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
        if not self.has_dolly:
            raise ValueError("VRChat Camera Dolly controls are unavailable. The camera feed is still available for observation.")
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
        if not self.has_dolly:
            return
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
