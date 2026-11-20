"""MCP tools for the currently running desktop pilot."""
import asyncio
from concurrent.futures import Future
import io
from queue import Empty, SimpleQueue
import threading
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image


class PilotBridge:
    def __init__(self, pilot):
        self.pilot = pilot
        self.queue = SimpleQueue()
        self.server = FastMCP("VRChat Pilot", host="127.0.0.1", port=29746,
                              stateless_http=True, json_response=True)

        @self.server.tool()
        async def observe(view: Literal["headset","left","right","stereo","camera"]="headset") -> Image:
            """Read the menu-compatible headset view, either native eye, their simultaneous stereo pair, or the handheld camera."""
            frame = await self.call("observe",view)
            data = io.BytesIO()
            frame.save(data,"JPEG",quality=90)
            return Image(data=data.getvalue(),format="jpeg")

        @self.server.tool()
        async def menu(action: Literal["toggle","aim","click"], x: float=.5, y: float=.5) -> str:
            """Toggle Quick Menu, aim at normalized headset x/y (origin top left), or click the held aim. Observe hover after aiming."""
            ticket = await self.call("menu",(action,x,y))
            for _ in range(60):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result",ticket)
                if outcome:
                    return outcome
            raise ValueError("No fresh headset frame after menu action; inspect status before retrying")

        @self.server.tool()
        async def menu_scroll(direction: Literal["up","down"], seconds: float=.35) -> str:
            """Scroll the menu panel under the right pointer with a bounded right-stick pulse."""
            if not .1 <= seconds <= 1.5:
                raise ValueError("Duration must be between 0.1 and 1.5 seconds")
            await self.call("menu_scroll",(1. if direction=="up" else -1.,seconds))
            await asyncio.sleep(seconds+.2)
            return "Scroll pulse finished; observe the panel"

        @self.server.tool()
        async def select_by_name(label: str, expected: str="", tooltip: str="") -> str:
            """Select one visible English VRChat menu control. New Instance has defaults; otherwise supply the new screen heading and hover tooltip."""
            ticket = await self.call("select_by_name",(label,expected,tooltip))
            try:
                for _ in range(520):
                    await asyncio.sleep(.1)
                    outcome=await self.call("menu_result",ticket)
                    if outcome:
                        if not outcome.startswith("Verified "):
                            raise ValueError(outcome)
                        return outcome
                raise ValueError("Selection timed out; inspect the current screen before retrying")
            finally:
                await self.call("cancel_selection",ticket)

        @self.server.tool()
        async def target(kind: Literal["view","wrist","camera"], values: list[float],
                         side: Literal["left","right"]="left") -> str:
            """Set view [relative yaw,pitch] in degrees (a quarter-turn is 90), wrist [x,y,z,yaw,pitch,roll] or player-local camera [x,y,z,pitch,yaw,roll]. Positions are metres, angles are degrees."""
            ticket = await self.call("target",(kind,values,side))
            for _ in range(150):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result",ticket)
                if outcome:
                    return outcome
            raise ValueError("Target result unavailable")

        @self.server.tool()
        async def body_action(action: Literal["rest","pickup","place","throw","leap"], values: list[float],
                              side: Literal["left","right"]="right", posture: str="standing") -> str:
            """Rest takes seat height; manipulation takes body-relative xyz and yaw/pitch/roll; leap takes []."""
            ticket = await self.call("body_action",(action,values,side,posture))
            for _ in range(150):
                await asyncio.sleep(.1)
                result = await self.call("menu_result",ticket)
                if result:
                    return result
            raise ValueError("Action result unavailable")

        @self.server.tool()
        async def confirm_fbt() -> str:
            """After opening VRChat Calibrate FBT, hold a T pose and confirm with both triggers."""
            ticket = await self.call("confirm_fbt", None)
            for _ in range(60):
                await asyncio.sleep(.1)
                outcome = await self.call("menu_result", ticket)
                if outcome:
                    return outcome
            raise ValueError("Calibration result unavailable")

        @self.server.tool()
        async def gesture(name: Literal["wave","nod","backflip","jumping-jacks","push-ups"]) -> str:
            """Play a finite full-body tracker clip and wait for completion."""
            ticket=await self.call("gesture",name)
            for _ in range(100):
                await asyncio.sleep(.1)
                outcome=await self.call("menu_result",ticket)
                if outcome:return outcome
            raise ValueError("Gesture timed out")

        @self.server.tool()
        async def camera(enabled: bool) -> str:
            """Connect or disconnect the VRChat Stream camera."""
            return await self.call("camera",enabled)

        @self.server.tool()
        async def run_objective(objective: str) -> str:
            """Have the local visual model execute a text objective in the pilot."""
            return await self.call("objective",objective)

        @self.server.tool()
        async def map_status() -> dict:
            """Get the active world's visual region, estimated location and latest region change."""
            return await self.call("map_status",None)

        @self.server.tool()
        async def open_map() -> dict:
            """Show Desktop Body's current world fly-through viewer."""
            return await self.call("open_map",None)

        @self.server.tool()
        async def world_context() -> dict:
            """Read the confirmed VRChat world ID, name and opaque instance key."""
            from world_context import world_context as read_context
            return await asyncio.to_thread(read_context)

        @self.server.tool()
        async def status() -> str:
            """Read the current pilot status and objective."""
            return await self.call("status",None)

        @self.server.tool()
        async def stop() -> str:
            """Cancel pilot activity and release held inputs."""
            return await self.call("stop",None)

    async def call(self, command, value):
        reply = Future()
        self.queue.put((command,value,reply))
        return await asyncio.wait_for(asyncio.wrap_future(reply),5)

    def drain(self):
        for _ in range(16):
            try:
                command,value,reply = self.queue.get_nowait()
            except Empty:
                break
            if not reply.set_running_or_notify_cancel():
                continue
            try:
                reply.set_result(self.pilot.remote(command,value))
            except Exception as error:
                reply.set_exception(error)

    def start(self):
        threading.Thread(target=self.server.run,kwargs={"transport":"streamable-http"},
                         daemon=True,name="pilot-mcp").start()
