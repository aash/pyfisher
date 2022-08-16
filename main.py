import win32gui
import ctypes
import win32api
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


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

def plot_images(imgs, images_per_row = 3):
    """Plot a series of images"""
    plt.gray()
    plt.rcParams.update({"figure.facecolor":  (0.0, 0.0, 0.0, 0.5)})
    rows = math.ceil(len(imgs)/images_per_row)
    f = plt.figure(figsize=(38, rows * 7))
    for k, l in enumerate(imgs):
        f.add_subplot(rows, images_per_row, k + 1)
        plt.imshow(l)
    plt.show()


def dilate_erode(img, ds = 5):
    """Combined dilate-erode filter"""
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    im = img
    im = cv.dilate(im, element)
    im = cv.erode(im, element)
    return im


def dilate(img, ds = 5):
    """Combined dilate-erode filter"""
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    im = img
    im = cv.dilate(im, element)
    return im


def erode(img, ds = 5):
    """Combined dilate-erode filter"""
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    im = img
    im = cv.erode(im, element)
    return im


def fill_gaps(mask):
    imgs = []
    mask_h, mask_w = mask.shape
    inv = np.invert(mask)
    imgs.append(inv)
    n, l, s, c = cv.connectedComponentsWithStats(inv, connectivity=8)
    label_idx = next((i for i, (_, _, w, h, _) in enumerate(s) \
        if i != 0 and w == mask_w and h == mask_h), None)
    label = (l == label_idx).astype(np.uint8)
    label *= 255
    filled = np.invert(label)
    imgs.append(filled)
    return filled, imgs


def filter_connected_components(mask, predicate):
    _, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    return list(i for i, st in enumerate(stats) \
        if i != 0 and predicate(i, st, labels, centroids)), labels, centroids

def find_threshold(ch0, th=0.01):
    x = 64
    amount = 1.0
    w, h = ch0.shape
    while amount > th:
        mask = (ch0 > x).astype(np.uint8)
        #mask = dilate_erode(mask, 3)
        cnt = np.count_nonzero(mask > 0)
        amount = cnt / (w*h)
        x += 3
    mask *= 255
    return mask

def solution(img, t, plot=False):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2LUV)
    hue = hsv[:,:,0]

    def in_range(x, a, b):
        return a < x and x < b

    sol = False
    if t == None:
        t = 128
        while not sol or t > 250:
            thresh = (hue > t).astype(np.uint8)
            nb_components, output, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=4)
            for i in range(1, nb_components):
                q = ((output==i).astype(np.uint8))
                x, y, w, h, a = stats[i]
                if in_range(w/h, 0.79, 1.26) and in_range(a, 9, 25) and in_range(a / (w*h), 0.5, 1.0):
                    #print(t)
                    sol = True
                    # x = int(centroids[i][0])
                    # y = int(centroids[i][1])
                    img = cv.bitwise_and(img, img, mask=q)
            if not sol:
                t += 2
    else:
        #print('qwe')
        thresh = (hue > t).astype(np.uint8)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=4)
        for i in range(1, nb_components):
            q = ((output==i).astype(np.uint8))
            x, y, w, h, a = stats[i]
            if in_range(w/h, 0.79, 1.26) and in_range(a, 9, 25) and in_range(a / (w*h), 0.5, 1.0):
                sol = True
                # x = int(centroids[i][0])
                # y = int(centroids[i][1])
                img = cv.bitwise_and(img, img, mask=q)

    if plot:
        plot_images([img, hue, thresh])
    if sol:
        return t, x, y, w, h, a
    else:
        return None

# def sol1(img):
#     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#     img = dilate_erode(img, 5)
#     red1 = (0, 0, 0)
#     red2 = (9, 255, 255)
#     imgs = []
#     hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
#     m_red = cv.inRange(hsv, red1, red2)
#     nb_components, output, stats, centroids = cv.connectedComponentsWithStats(m_red, connectivity=4)
#     m = max(zip(stats[1:], range(1,nb_components)), key=lambda x: x[0][cv.CC_STAT_AREA])
#     max_lab = (output == m[1]).astype(np.uint8)
#     max_lab *= 255
#     imgs += [img, m_red, cv.bitwise_and(img, img, mask=max_lab)]
#     imgs.append(max_lab)
#     return max_lab, tuple(map(int, centroids[m[1]])), m[0][cv.CC_STAT_AREA], imgs


# def sol2(img):
#     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#     img = dilate_erode(img, 3)
#     imgs = []
#     hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
#     hue = hsv[:,:,0]
#     sat = hsv[:,:,1]
#     m_red = ((hue < 14) * (sat > 70)).astype(np.uint8)
#     m_red *= 255
#     # imgs += [img, hsv, m_red, cv.bitwise_and(img, img, mask=m_red)]
#     nb_components, output, stats, centroids = cv.connectedComponentsWithStats(m_red, connectivity=4)
#     if nb_components > 1:
#         m = max(zip(stats[1:], range(1,nb_components)), key=lambda x: x[0][cv.CC_STAT_AREA])
#         max_lab = (output == m[1]).astype(np.uint8)
#         max_lab *= 255
#         # imgs.append(max_lab)
#         return max_lab, tuple(map(int, centroids[m[1]])), m[0][cv.CC_STAT_AREA], imgs
#     else:
#         return None, None, None, imgs


def sol3(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    
    imgs = []
    hsv = cv.cvtColor(img, cv.COLOR_RGB2Luv)
    hue = hsv[:,:,0]
    #sat = hsv[:,:,1]
    #m_red = (((108 < hue) * (hue < 120)) * (80 > sat)).astype(np.uint8)
    m_red = (hue > 167).astype(np.uint8)
    m_red *= 255
    #m_red = dilate_erode(m_red, 1)
    # imgs += [img, hsv, m_red, cv.bitwise_and(img, img, mask=m_red)]
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(m_red, connectivity=4)
    if nb_components > 1:
        m = max(zip(stats[1:], range(1,nb_components)), key=lambda x: x[0][cv.CC_STAT_AREA])
        max_lab = (output == m[1]).astype(np.uint8)
        max_lab *= 255
        # imgs.append(max_lab)
        return max_lab, tuple(map(int, centroids[m[1]])), m[0][cv.CC_STAT_AREA], imgs
    else:
        return None, None, None, imgs


def sol(img):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    h,w,_ = img.shape
    rect = (2,2, w-4, h-4)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #img = img*mask2[:,:,np.newaxis]
    # img = dilate_erode(img, 3)
    # img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # img = img[:,:,0]
    # img = (img > 96).astype(np.uint8)
    # img *= 255

    img = np.ones(img.shape[:2], np.uint8)*mask2[:,:]
    img *= 255
    # img_ = img.copy()
    # img = dilate_erode(img, 5)
    # print(img.shape)

    # nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=4)
    # if nb_components > 1:
    #     m = max(zip(stats[1:], range(1,nb_components)), key=lambda x: x[0][cv.CC_STAT_AREA])
    #     x, y = tuple(map(int, centroids[m[1]]))
    #     max_lab = (output == m[1]).astype(np.uint8)
    #     max_lab *= 255
    
    a = np.count_nonzero(img)
    if a < 100:
        return None, None, None, None
    center = [ np.average(indices) for indices in np.where(img >= 255) ]
    x, y = map(int, center)

    #img[x][y] = 0
    #print(img.min(), img.max())

    img1 = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    img1[:,:,0] = img[:,:]
    img1[:,:,1] = img[:,:]
    img1[:,:,2] = img[:,:]
    img1[x][y] = (255,0,0)
    # img[x][y] = 0
    return x, y, a, img