import sys
from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QRubberBand
from PyQt5.QtCore import QTimer, QPoint, QRect, QSize
from main import *
import win32gui
import ctypes
import ahk


class MainWindow(QMainWindow):
    def __init__(self, rect = (0, 0 , 220, 32), wow_hwnd = 0, ahk = ahk.AHK()):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            #QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setWindowOpacity(0.3)
        self.hwnd = wow_hwnd
        r = rect
        self.rect1 = rect
        self.setGeometry(r[0], r[1], r[2], r[3])
        self.lbl = QtWidgets.QLabel(self)
        self.timer = QTimer()
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.timer_timeout_cnt = 0
        self.hist = []
        self.hist_max = 20
        self.selection = QRect()
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.ahk = ahk


    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Escape:
            for i, img in enumerate(self.hist):
                img.save(f'img{str(i).zfill(2)}.jpg')
            QtWidgets.QApplication.instance().exit(0)
        if a0.key() == QtCore.Qt.Key.Key_0:
            # save frame
            ahk.key_press()
            # init bobber wait
            # after a period of time save another frame
            # find bobber coord
            pass
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
            self.selection = self.rubberBand.geometry()
            print(self.selection)
            self.rubberBand.hide()


    def update(self):
        screen = QtWidgets.QApplication.primaryScreen()
        r = self.rect1
        screenshot = screen.grabWindow(self.hwnd)
        self.lbl.resize(screenshot.size())
        self.lbl.setPixmap(screenshot)
        self.timer_timeout_cnt += 1
        sub = screenshot.copy(self.selection)
        print(self.selection)
        self.hist.append(sub)
        while len(self.hist) > self.hist_max:
            self.hist.pop(0)


if __name__ == '__main__':
    ahk = ahk.AHK()
    hwnd = get_wow_hwnd()
    PROCESS_PER_MONITOR_DPI_AWARE = 2
    MDT_EFFECTIVE_DPI = 0
    shcore = ctypes.windll.shcore
    hresult = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    r = win32gui.GetWindowRect(hwnd)
    rc = win32gui.GetClientRect(hwnd)
    c = win32gui.ClientToScreen(hwnd, (0,0))
    app = QApplication(sys.argv)
    r = (c[0], c[1], rc[2], rc[3])
    mywindow = MainWindow(r, hwnd, ahk)
    mywindow.show()
    app.exec()
