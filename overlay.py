import sys
from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QRubberBand, QLabel
from PyQt5.QtCore import QTimer, QPoint, QRect, QSize, QRectF
from PyQt5.QtGui import QImage, QPainter, QColor
from main import *
import win32gui
import ctypes
import ahk
from enum import Enum
from datetime import *
from time import time
import numpy as np
import cv2

def millis_now():
    return int(time() * 1000)

class WindowState(Enum):
    NORMAL = 'normal'
    SELECT_FISHING_AREA = 'select-fishing-area'
    WAIT_BOBBER = 'wait-bobber'


def convertQImageToMat(img: QImage) -> np.ndarray: 
    '''  Converts a QImage into an opencv MAT format  '''
    img = img.convertToFormat(4)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(h, w, 4)
    return arr

class MainWindow(QMainWindow):
    def __init__(self, rect = (0, 0 , 220, 32), wow_hwnd = 0, ahk = ahk.AHK()):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            #QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setWindowOpacity(0.4)
        self.hwnd = wow_hwnd
        self.state = WindowState.NORMAL
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
        self.txtStatus = QLabel(self)
        self.txtStatus.setFixedWidth(400)
        self.txtStatus.setStyleSheet("color: #ffffff; background-color: black;")
        self.txtStatus.setText(self.state.name)
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.ahk = ahk
        self.wow = ahk.find_window(title=b'World of Warcraft')
        self.countdown = millis_now()
        self.coord = None
        assert self.wow != None


    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Escape:
            for i, img in enumerate(self.hist):
                img.save(f'img{str(i).zfill(2)}.png')
            QtWidgets.QApplication.instance().exit(0)
        if a0.key() == QtCore.Qt.Key.Key_S:
            if self.state == WindowState.NORMAL:
                self.state = WindowState.SELECT_FISHING_AREA
                self.txtStatus.setText(self.state.name)
        if a0.key() == QtCore.Qt.Key.Key_0:
            # save frame
            #ahk.key_press()
            self.hist[-1].save('before.png')
            self.wow.send('3')
            self.countdown = millis_now()
            self.state = WindowState.WAIT_BOBBER
            # init bobber wait
            # after a period of time save another frame
            # find bobber coord
            pass
        return super().keyPressEvent(a0)


    def mousePressEvent(self, event):
        if self.state == WindowState.SELECT_FISHING_AREA:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.origin = QPoint(event.pos())
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
    
    
    def mouseMoveEvent(self, event):
        if self.state == WindowState.SELECT_FISHING_AREA:
            if not self.origin.isNull():
                self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

  
    def mouseReleaseEvent(self, event):
        if self.state == WindowState.SELECT_FISHING_AREA:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.state = WindowState.NORMAL
                self.txtStatus.setText(self.state.name)
                self.selection = self.rubberBand.geometry()
                #print(self.selection)
                self.rubberBand.hide()


    def update(self):
        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.hwnd)
        self.lbl.resize(screenshot.size())
        self.lbl.setPixmap(screenshot)
        self.timer_timeout_cnt += 1
        if not self.selection.isEmpty():
            sub = screenshot.copy(self.selection)
            self.hist.append(sub)
            while len(self.hist) > self.hist_max:
                self.hist.pop(0)

            if self.state == WindowState.WAIT_BOBBER:
                if millis_now() - self.countdown > 3000:
                    self.hist[-1].save('after.png')
                    self.state = WindowState.NORMAL

                    import cv2 as cv
                    import matplotlib.pyplot as plt
                    #img = cv.imread('after.png')

                    img = convertQImageToMat(self.hist[-1].toImage())
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
                    hue = hls[:,:,0]
                    r, mask = cv.threshold(hue, 0, 10, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
                    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
                    for i, c in enumerate(centroids):
                        if stats[i][cv.CC_STAT_AREA] > 400:
                            cc = tuple(map(int, c))
                            self.coord = (cc[0] + self.selection.left(), cc[1] + self.selection.top())
                            c = self.coord
                            #cv.circle(img, tuple(map(int, c)), radius=7, color=(255,255,255), thickness=1)
            if self.coord and type(self.coord) == tuple:
                d = 3
                p = QPainter()
                p.begin(self.lbl.pixmap())
                c = self.coord
                p.setPen(QColor(255, 0, 0))
                p.drawEllipse(c[0], c[1], d, d)
                p.end()



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
