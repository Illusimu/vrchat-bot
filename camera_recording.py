"""Record timestamped snapshots from the pilot's independent camera."""

import asyncio
import base64
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic


class CameraRecording:
    limit = 30.0

    def __init__(self, capture, output):
        self.capture = capture
        self.output = output
        self.task = None

    async def start(self):
        if self.task is not None:
            raise RuntimeError("Stop and save the current camera recording first.")
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("FFmpeg is required to save the camera recording.")
        first = await self.capture()
        self.directory = TemporaryDirectory(prefix="desktop-body-recording-")
        self.timestamps = []
        self.stop_event = asyncio.Event()
        self.started = monotonic()
        self.save_frame(first)
        self.task = asyncio.create_task(self.collect())

    def save_frame(self, frame):
        elapsed = monotonic() - self.started
        path = Path(self.directory.name) / f"{len(self.timestamps):04d}.jpg"
        path.write_bytes(base64.b64decode(frame.data))
        self.timestamps.append(elapsed)

    async def collect(self):
        while not self.stop_event.is_set() and monotonic() - self.started < self.limit:
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=.08)
            except TimeoutError:
                self.save_frame(await self.capture())

    async def stop(self):
        if self.task is None:
            raise RuntimeError("There is no active camera recording.")
        self.stop_event.set()
        try:
            await self.task
            if len(self.timestamps) < 2:
                raise RuntimeError("The camera recording did not capture enough frames.")
            durations = [b - a for a, b in zip(self.timestamps, self.timestamps[1:])]
            durations.append(durations[-1])
            folder = Path(self.directory.name)
            entries = ["ffconcat version 1.0"]
            for index, duration in enumerate(durations):
                entries.extend((f"file '{index:04d}.jpg'", f"duration {duration:.6f}"))
            entries.append(f"file '{len(durations) - 1:04d}.jpg'")
            (folder / "capture.ffconcat").write_text("\n".join(entries) + "\n", encoding="ascii")
            destination = folder / "capture.mp4"
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
            process = await asyncio.create_subprocess_exec(
                self.ffmpeg, "-nostdin", "-loglevel", "error", "-f", "concat", "-i", "capture.ffconcat",
                "-vf", "fps=12,scale=1280:-2", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
                cwd=folder, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, **options)
            try:
                _, error = await process.communicate()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            if process.returncode:
                raise RuntimeError(f"Camera encoding failed: {error.decode(errors='replace').strip()}")
            # Keep the completed file on the trial drive before publishing the final name.
            staging = self.output.with_suffix(".partial.mp4")
            try:
                shutil.copyfile(destination, staging)
                staging.replace(self.output)
            finally:
                staging.unlink(missing_ok=True)
            return {"clip": self.output.name, "duration_seconds": round(sum(durations), 2),
                    "captured_frames": len(durations)}
        finally:
            await self.close()

    async def close(self):
        if self.task is None:
            return
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.task = None
        self.directory.cleanup()
