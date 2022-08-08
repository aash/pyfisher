import win32gui
import ctypes
import win32api


def get_wow_hwnd():
    toplist, winlist = [], []
    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
    win32gui.EnumWindows(enum_cb, toplist)

    for hwnd, title in winlist:
        if 'warcraft' in title.lower():
            return hwnd


def get_dpi():
    PROCESS_PER_MONITOR_DPI_AWARE = 2
    MDT_EFFECTIVE_DPI = 0
    shcore = ctypes.windll.shcore
    monitors = win32api.EnumDisplayMonitors()
    hresult = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    assert hresult == 0
    dpiX = ctypes.c_uint()
    dpiY = ctypes.c_uint()
    dpi = {}
    for i, monitor in enumerate(monitors):
        shcore.GetDpiForMonitor(
            monitor[0].handle,
            MDT_EFFECTIVE_DPI,
            ctypes.byref(dpiX),
            ctypes.byref(dpiY)
        )
        dpi[monitor[0].handle] = (dpiX.value, dpiY.value)
    return dpi    


def dpi_to_scale_ratio(dpi):
    STANDARD_DPI = 96
    if len(dpi) != 2 or dpi[0] != dpi[1]:
        raise RuntimeError(f'non conformant DPI:{dpi[0]}x{dpi[1]}')
    return dpi[0] / STANDARD_DPI
