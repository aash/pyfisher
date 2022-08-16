import sys
from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QRubberBand, QLabel, QStackedLayout, QVBoxLayout, QWidget, QSpacerItem
from PyQt5.QtCore import QTimer, QPoint, QRect, QSize
from PyQt5.QtGui import QImage, QPainter, QColor, QVector2D, QPixmap
from main import *
import win32gui
import ctypes
import ahk
import traceback
from enum import Enum
from datetime import *
from time import time
import numpy as np
import cv2
from minimal import BobberDetector
from PyQt5.QtCore import pyqtSlot, QRunnable, QThreadPool, pyqtSignal, QObject, QMutexLocker, QMutex

def millis_now():
    return int(time() * 1000)

class WindowState(Enum):
    NORMAL = 'normal'
    SELECT_FISHING_AREA = 'select-fishing-area'
    WAIT_BOBBER = 'wait-bobber'
    TRACK_BOBBER = 'track-bobber'
    TRIGGER = 'trigger'
    START_FISHING = 'start-fishing'


def convertQImageToMat(img: QImage, dims=4) -> np.ndarray: 
    '''  Converts a QImage into an opencv MAT format  '''
    img = img.convertToFormat(4)
    # if dims == 4:
    #     img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    # elif dims == 3:
    #     img = img.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    # else:
    #     raise RuntimeError('invalid image format, only 3,4 channels supported')
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(h, w, 4)
    if dims == 4:
        return arr
    elif dims == 3:
        arr_out = np.zeros((*arr.shape[:2], 3), np.uint8)
        arr_out[:,:,0] = arr[:,:,0]
        arr_out[:,:,1] = arr[:,:,1]
        arr_out[:,:,2] = arr[:,:,2]
        return arr_out
    else:
        raise RuntimeError(f'unsupported color channel dimension: {dims}')

class MainWindow(QMainWindow):
    def __init__(self, rect = (0, 0 , 220, 32), wow_hwnd = 0, ahk = ahk.AHK()):
        QMainWindow.__init__(self)
        # self.setWindowFlags(
        #     #QtCore.Qt.WindowStaysOnTopHint |
        #     QtCore.Qt.WindowType.FramelessWindowHint |
        #     QtCore.Qt.WindowType.X11BypassWindowManagerHint
        # )
        self.setWindowOpacity(0.6)

        w = QWidget()
        self.w = w
        w.setLayout(QVBoxLayout())
        self.setCentralWidget(w)

        self.hwnd = wow_hwnd
        self.state = WindowState.NORMAL
        r = rect
        self.rect1 = rect
        self.setGeometry(r[0], r[1], r[2], r[3])
        self.lbl = QLabel()
        self.timer = QTimer()
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.timer_timeout_cnt = 0
        self.hist = []
        self.bhist = []
        self.hist_max = 20
        self.selection = QRect()
        #self.stack = QVBoxLayout()
        #self.stack.setDirection(QVBoxLayout.Direction.Down)
        # self.stack.addWidget(self.windowStatus)
        # self.stack.addWidget(self.txtStatus)
        # self.l = QVBoxLayout()
        # self.l.addChildLayout(self.stack)
        #self.setLayout(self.stack)


        self.txtStatus = QLabel()
        self.txtStatus.setFixedWidth(400)
        self.txtStatus.setStyleSheet("color: #ffffff; background-color: black;")
        self.txtStatus.setText(self.state.name)
        #self.sp = QSpacerItem(1,1)
        self.windowStatus = QLabel()
        self.windowStatus.setFixedWidth(400)
        self.windowStatus.setStyleSheet("color: #ffffff; background-color: black;")
        self.setLayoutDirection(QtCore.Qt.LayoutDirection.LayoutDirectionAuto)
        self.fish_loop_toggle = False
        self.cvtb = {True: 'Y', False: 'N'}
        self.fish_loop_toggle_fmt = lambda: f'L:{self.cvtb[self.fish_loop_toggle]}'
        self.windowStatus.setText(self.fish_loop_toggle_fmt())

        w.layout().addWidget(self.lbl)
        w.layout().addWidget(self.txtStatus)
        w.layout().addWidget(self.windowStatus)

        #self.stack.addWidget(self.txtStatus)

        # self.l.addWidget(self.lbl)
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.ahk = ahk
        self.wow = ahk.find_window(title=b'World of Warcraft')
        self.countdown = millis_now()
        self.coord = None
        self.pcoord = None
        self.det = BobberDetector('yolov5/best.pt')
        self.rect = None
        self.threadpool = QThreadPool()
        self.mut = QMutex()
        self.tt = None
        self.msk = None
        '''
        self.setGeometry(100,100, 640, 480)
        l = QVBoxLayout()
        self.l3 = QLabel()
        self.p = QPixmap()
        self.p.load("after.png")
        self.p = self.p.scaled(640, 480)
        self.l3.setPixmap(self.p)
        self.l3.resize(self.p.size())
        l.addWidget(self.l3)
        w = QWidget()
        w.setLayout(l)
        self.setCentralWidget(w)
        self.l1 = QLabel()
        self.l1.setText("label1")
        self.l1.setFixedWidth(400)
        self.l1.setFixedHeight(15)
        self.l2 = QLabel()
        self.l2.setText("label2")
        self.l2.setFixedHeight(15)
        self.l2.setFixedWidth(400)
        l.addWidget(self.l1, QtCore.Qt.AlignmentFlag.AlignTop)
        l.addSpacing(1)
        l.addWidget(self.l2, QtCore.Qt.AlignmentFlag.AlignTop)
        l.addStretch(1)
        '''
        assert self.wow != None

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Escape:
            # for i, img in enumerate(self.hist):ex
            #     img.save(f'img{str(i).zfill(2)}.png')
            self.threadpool.clear()
            QtWidgets.QApplication.instance().exit(0)
        if a0.key() == QtCore.Qt.Key.Key_S:
            if self.state == WindowState.NORMAL:
                self.state = WindowState.SELECT_FISHING_AREA
                self.txtStatus.setText(self.state.name)
        if a0.key() == QtCore.Qt.Key.Key_0:
            if not self.selection.isEmpty():
                self.state = WindowState.START_FISHING
                self.txtStatus.setText(self.state.name)
                self.countdown = millis_now()
            else:
                print('set up fishing area')
        if a0.key() == QtCore.Qt.Key.Key_L:
            self.fish_loop_toggle = not self.fish_loop_toggle
            self.windowStatus.setText(self.fish_loop_toggle_fmt())
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

    def worker_complete(self):
        self.worker_busy = False

    def worker_handle_result(self, res):
        if len(res) == 1:
            with QMutexLocker(self.mut):
                self.state = WindowState.TRACK_BOBBER
                rect, conf, l = next(iter(res))
                self.rect = tuple(rect)
        else:
            with QMutexLocker(self.mut):
                self.state = WindowState.START_FISHING if self.fish_loop_toggle else WindowState.NORMAL
                self.countdown = millis_now()
            if not res:
                print('bobber not found')
            else:
                print('many bobbers found')


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

            if self.state == WindowState.START_FISHING:
                if millis_now() - self.countdown > 1000:
                    self.wow.send('3')
                    self.countdown = millis_now()
                    self.state = WindowState.WAIT_BOBBER

            
            if self.state == WindowState.WAIT_BOBBER:
                if millis_now() - self.countdown > 3000:
                    self.hist[-1].save('after.png')
                    #self.state = WindowState.NORMAL
                    img = convertQImageToMat(self.hist[-1].toImage())
                    img = img[:,:,:3]
                    worker = Worker(self.det.detect, img)
                    #worker = Worker(lambda x: [], img)
                    worker.signals.result.connect(self.worker_handle_result)
                    worker.signals.finished.connect(self.worker_complete)
                    self.worker_busy = True
                    self.threadpool.start(worker)

 #or self.state == WindowState.TRIGGER:
            if self.state == WindowState.TRACK_BOBBER:
                if millis_now() - self.countdown > 3000:
                    if self.rect and type(self.rect) == tuple:
                        c = self.rect
                        qimg = screenshot.copy(QRect( \
                            self.selection.topLeft() + QPoint(*c[:2]), \
                            self.selection.topLeft() + QPoint(*c[2:]) + QPoint(0, 30))).toImage()
                        #qimg.save(f"bobber{self.timer_timeout_cnt}.png")
                        img = convertQImageToMat(qimg, 3)
                        #msk, cent, a, _ = (img)
                        x, y, a, msk = sol(img)
                        cent = None
                        if x and y:
                            cent = [x, y]
                        if cent != None:
                            self.bhist.append(cent)
                            while len(self.bhist) > self.hist_max:
                                self.bhist.pop(0)
                            self.pcoord = self.coord
                            self.coord = tuple(cent)
                            self.msk = msk
                            # if a < 900:
                            #     qimg.save(f"fail{millis_now()}.png")
                        else:
                            qimg.save(f"fail{millis_now()}.png")
                            self.state = WindowState.START_FISHING if self.fish_loop_toggle else WindowState.NORMAL
                            self.rect = None
                            self.coord = None
                            self.pcoord = None
                            self.bhist.clear()
                            self.countdown = millis_now()
                            self.msk = None

                        if self.pcoord != None:
                            l = (QVector2D(*self.coord) - QVector2D(*self.pcoord)).length()
                            print(l)
                            if l > 5.0:
                                self.state = WindowState.TRIGGER
                                self.countdown = millis_now()
                if millis_now() - self.countdown > 20000:
                    self.state = WindowState.START_FISHING if self.fish_loop_toggle else WindowState.NORMAL
                    self.rect = None
                    self.coord = None
                    self.pcoord = None
                    self.bhist.clear()
                    self.countdown = millis_now()
                    self.msk = None


            if self.state == WindowState.TRIGGER:
                if millis_now() - self.countdown < 20000:
                    c = win32gui.ClientToScreen(self.hwnd, (0,0))
                    #print(c)
                    p = QPoint(*c) + self.selection.topLeft() + QPoint(*self.rect[:2]) + QPoint(*self.bhist[0])
                    print("caught!!!")
                    #self.wow.activate()
                    self.ahk.mouse_move(p.x(), p.y())
                    self.ahk.right_click()
                    self.activateWindow()
                    self.rect = None
                    self.coord = None
                    self.pcoord = None
                    self.bhist.clear()
                    self.msk = None
                    self.state = WindowState.START_FISHING if self.fish_loop_toggle else WindowState.NORMAL
                    self.countdown = millis_now()

            if self.coord != None:
                p = QPainter()
                # print(self.msk.shape, self.msk.min(), self.msk.max())
                img2 = np.zeros((*self.msk.shape, 4), np.uint8)
                img2[:,:,0] = 0
                img2[:,:,1] = 0
                img2[:,:,2] = self.msk
                img2[:,:,3] = self.msk
                height, width = self.msk.shape
                bytesPerLine = 4 * width
                qImg = QImage(img2.data, width, height, bytesPerLine, QImage.Format_ARGB32)
                p.begin(self.lbl.pixmap())
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver);
                p.drawImage(self.selection.topLeft() + QPoint(*self.rect[:2]), qImg)
                p.end()
            if self.rect and type(self.rect) == tuple:
                if millis_now() - self.countdown < 500 and self.state == WindowState.WAIT_BOBBER:
                    p = QPainter()
                    p.begin(self.lbl.pixmap())
                    p.setPen(QColor(255, 0, 0))
                    c = self.rect
                    p.drawRect(QRect( \
                        self.selection.topLeft() + QPoint(*c[:2]), \
                        self.selection.topLeft() + QPoint(*c[2:])))
                    p.end()
            if self.selection:
                p = QPainter()
                p.begin(self.lbl.pixmap())
                p.setPen(QColor(0x135492))
                p.drawRect(self.selection)
                p.end()


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


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
