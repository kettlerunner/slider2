"""Platform-specific mouse cursor hide/show for slideshow mode."""

import platform
import ctypes
from ctypes import c_char_p, c_ulong, c_void_p
from ctypes.util import find_library

_cursor_state = {
    "hidden": False,
    "system": platform.system(),
    "display": None,
    "window": None,
    "x11": None,
    "xfixes": None,
}


def hide_mouse_cursor():
    """Attempt to hide the system mouse cursor while the slideshow is active."""
    state = _cursor_state
    if state["hidden"]:
        return

    system = state["system"]

    try:
        if system == "Windows":
            ctypes.windll.user32.ShowCursor(False)
            state["hidden"] = True
        elif system == "Darwin":
            try:
                from AppKit import NSCursor  # type: ignore
            except ImportError:
                return
            NSCursor.hide()
            state["hidden"] = True
        elif system == "Linux":
            lib_x11 = find_library("X11")
            lib_xfixes = find_library("Xfixes")
            if not lib_x11 or not lib_xfixes:
                return

            x11 = ctypes.cdll.LoadLibrary(lib_x11)
            xfixes = ctypes.cdll.LoadLibrary(lib_xfixes)

            x11.XOpenDisplay.argtypes = [c_char_p]
            x11.XOpenDisplay.restype = c_void_p
            display = x11.XOpenDisplay(None)
            if not display:
                return

            x11.XDefaultRootWindow.argtypes = [c_void_p]
            x11.XDefaultRootWindow.restype = c_ulong
            root = x11.XDefaultRootWindow(display)

            xfixes.XFixesHideCursor.argtypes = [c_void_p, c_ulong]
            xfixes.XFixesHideCursor.restype = None
            xfixes.XFixesHideCursor(display, root)

            x11.XFlush.argtypes = [c_void_p]
            x11.XFlush.restype = None
            x11.XFlush(display)

            state.update({
                "hidden": True,
                "display": display,
                "window": root,
                "x11": x11,
                "xfixes": xfixes,
            })
    except Exception as exc:
        print(f"Failed to hide mouse cursor: {exc}")


def show_mouse_cursor():
    """Restore the system mouse cursor after the slideshow ends."""
    state = _cursor_state
    if not state["hidden"]:
        return

    system = state["system"]

    try:
        if system == "Windows":
            ctypes.windll.user32.ShowCursor(True)
        elif system == "Darwin":
            try:
                from AppKit import NSCursor  # type: ignore
            except ImportError:
                pass
            else:
                NSCursor.unhide()
        elif system == "Linux":
            display = state.get("display")
            window = state.get("window")
            x11 = state.get("x11")
            xfixes = state.get("xfixes")
            if display and window is not None and x11 and xfixes:
                xfixes.XFixesShowCursor.argtypes = [c_void_p, c_ulong]
                xfixes.XFixesShowCursor.restype = None
                xfixes.XFixesShowCursor(display, window)

                x11.XFlush.argtypes = [c_void_p]
                x11.XFlush.restype = None
                x11.XFlush(display)

                x11.XCloseDisplay.argtypes = [c_void_p]
                x11.XCloseDisplay.restype = ctypes.c_int
                x11.XCloseDisplay(display)
    except Exception as exc:
        print(f"Failed to restore mouse cursor: {exc}")
    finally:
        state.update({
            "hidden": False,
            "display": None,
            "window": None,
            "x11": None,
            "xfixes": None,
        })
