"""Keep one Windows Graphics Capture session alive for each pilot view."""
import atexit
from pathlib import Path
import sys
from threading import Condition, Lock
from time import monotonic

from PIL import Image


class WindowCapture:
    def __init__(self, hwnd):
        vendor = Path(__file__).with_name("build") / "capture-lib"
        if vendor.is_dir() and str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from windows_capture import WindowsCapture

        self.ready = Condition()
        self.frame = None
        self.sequence = 0
        self.captured_at = 0.
        self.closed = False
        self.capture = WindowsCapture(window_hwnd=hwnd, cursor_capture=False, draw_border=False)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)
        try:
            self.control = self.capture.start_free_threaded()
        except BaseException:
            self.capture.capture = None
            raise

    def on_frame_arrived(self, frame, _control):
        now = monotonic()
        with self.ready:
            if self.closed or now - self.captured_at < .08:
                return
            # Own the pixels before Windows unmaps the native frame.
            self.frame = Image.fromarray(frame.frame_buffer[:, :, 2::-1].copy())
            self.captured_at = now
            self.sequence += 1
            self.ready.notify_all()

    def on_closed(self):
        with self.ready:
            self.closed = True
            self.ready.notify_all()

    def next_frame(self, timeout=3):
        with self.ready:
            previous = self.sequence
            self.ready.wait_for(lambda: self.closed or self.sequence != previous, timeout)
            if self.closed or self.sequence == previous:
                return None
            return self.frame.copy()

    def close(self):
        self.on_closed()
        try:
            if not self.control.is_finished():
                self.control.stop()
            self.control.wait()
        finally:
            # The native wrapper owns bound callbacks back to the Python
            # wrapper. Drop its reference explicitly after the thread exits.
            self.capture.capture = None
            self.capture.frame_handler = self.capture.closed_handler = None
            self.frame = None


_sessions = {}
_lock = Lock()


def capture_window(kind, hwnd, pid):
    key = (hwnd, pid)
    with _lock:
        entry = _sessions.get(kind)
        if entry is None or entry[0] != key or entry[1].closed or entry[1].control.is_finished():
            if entry is not None:
                _sessions.pop(kind)
                entry[1].close()
            session = WindowCapture(hwnd)
            _sessions[kind] = key, session
        else:
            session = entry[1]
    return session.next_frame()


@atexit.register
def close_capture_sessions():
    with _lock:
        while _sessions:
            _, (_, session) = _sessions.popitem()
            session.close()
