import sys

from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5 import Qt

from PyQt5.QtWidgets import QMainWindow, QApplication, QRubberBand
from PyQt5.QtCore import QTimer, QPoint, QRect, QSize


from main import *
import win32gui
import ctypes



class MainWindow(QMainWindow):
    def __init__(self, rect = (0, 0 , 220, 32), wow_hwnd = 0):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            #QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setWindowOpacity(0.3)
        self.hwnd = wow_hwnd
        r = rect
        self.rect1 = rect
        self.setGeometry(r[0], r[1], r[2], r[3])

        self.lbl = QtWidgets.QLabel(self)
        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.timer_timeout_cnt = 0
        self.hist = []
        self.hist_max = 10

        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Escape:
            QtWidgets.QApplication.instance().exit(0)
        return super().keyPressEvent(a0)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()        
    
    
    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

  
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.rubberBand.hide()


    def update(self):
        screen = QtWidgets.QApplication.primaryScreen()
        r = self.rect1
        screenshot = screen.grabWindow(self.hwnd)
        self.lbl.resize(screenshot.size())
        self.lbl.setPixmap(screenshot)
        self.timer_timeout_cnt += 1
        sub = screenshot.copy(self.rubberBand.geometry())
        self.hist.append(sub)
        while len(self.hist) > self.hist_max:
            self.hist.pop(0)
        print(len(self.hist))


if __name__ == '__main__':
    hwnd = get_wow_hwnd()
    # dpis = get_dpi()
    # ratio = dpi_to_scale_ratio(next(iter(dpis.values())))
    # print(ratio)
    PROCESS_PER_MONITOR_DPI_AWARE = 2
    MDT_EFFECTIVE_DPI = 0
    shcore = ctypes.windll.shcore
    hresult = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    r = win32gui.GetWindowRect(hwnd)
    rc = win32gui.GetClientRect(hwnd)
    c = win32gui.ClientToScreen(hwnd, (0,0))
    print(c)
    # r1 = map(lambda x: x * ratio, r)
    #print(list(r), r[2] - r[0], r[3] - r[1])
    app = QApplication(sys.argv)
    r = (c[0], c[1], rc[2], rc[3])
    #r1 = tuple(map(lambda x: x * 0.5, r))
    mywindow = MainWindow(r, hwnd)
    mywindow.show()
    app.exec_()