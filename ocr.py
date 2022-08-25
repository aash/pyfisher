
import matplotlib.pyplot as plt
import numpy as np
from main import *
from functools import partial
import cv2
import math
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.draw import *
from matplotlib.figure import Figure
from main import *
import cv2
import random
import string
import sys
from random import randint
from skimage.transform import resize
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from joblib import dump, load

classify = not os.path.exists('./svc_digits.model')

if classify:
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.rcParams.update({"figure.facecolor":  (0.0, 0.0, 0.0, 0.4)})
    # fig = plt.figure(figsize=(4,1))
    # ax1 = fig.add_axes([-1,-1,2,2])
    # ax1.set_facecolor((0,0,0))
    # canvas = FigureCanvasAgg(fig)
    imgsfn = [(f'sample{i}.png', f'sample{i}.txt') for i in range(4)]
    chs = []
    max_lhs = []
    target = []
    for ifn, tfn in imgsfn:
        X = cv2.imread(ifn)
        luv = cv2.cvtColor(X, cv2.COLOR_BGR2LUV)
        l = luv[:,:,0]
        l = (l > 128).astype(np.uint8)
        l *= 255
        #print(l.min(), l.max())
        h, w = l.shape
        lines = segment_line_in_image(l)

        with open(tfn) as f:
            st = f.read().strip()
        txt_lines = st.split('\n')
        if txt_lines[-1] == '':
            txt_lines.pop()
        assert (len(lines) // 2) == len(txt_lines)

        
        max_lh = max([(y1 - y0) for (y0, _), (y1, _) in zip(lines[::2], lines[1::2])])
        max_lhs.append(max_lh)
        #print(f'max line height: {max_lh}')
        for (y0, _), (y1, _), st in zip(lines[::2], lines[1::2], txt_lines):
            ln = l[y0:y1,:]
            lh, _ = ln.shape
            chars = segment_chars_in_line(ln)
            chars_cnt = len(chars) // 2
            assert chars_cnt == len(st)
            target.extend(list(st.replace('.', '')))
            for (x0, _), (x1, _) in zip(chars[::2], chars[1::2]):
                ch = ln[0:lh,x0:x1]
                # skip dots
                if np.count_nonzero(ch) < 10:
                    continue
                _, cw = ch.shape
                ch_ = np.zeros((max_lh + 4, max_lh + 4), dtype=np.uint8)
                x0 = (max_lh + 4 - cw) // 2
                x1 = x0 + cw
                y0 = (max_lh + 4 - lh) // 2
                y1 = y0 + lh
                ch_[y0:y1,x0:x1] = ch
                ch_ = ch_.astype(np.float64) / 16.0
                assert ch_.shape == (max_lh + 4, max_lh + 4)
                chs.append(ch_)
    assert len(set(max_lhs)) == 1
    assert len(target) == len(chs)
    n_samples = len(chs)
    chs_ = np.array(chs)
    data = chs_.reshape((n_samples, -1))

    # imgs = []
    # for i in range(n_samples):
    #     imgs.append(resize(digits.images[i,:,:], (8,8)))
    # imgsa = np.array(imgs)
    #data = images.reshape((n_samples, -1))
    #data = imgsa.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.01, shuffle=True
    )
    print(f'X_train.shape {X_train.shape}, n_samples {n_samples}')
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
else:
    clf = load('./svc_digits.model')


img = cv2.imread('66.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

l = luv[:,:,0]
#l = cv2.GaussianBlur(l, (3,3), 0)
# l = np.invert(l)
l = (l > 128).astype(np.uint8)
l *= 255

h, w = l.shape

if classify:
    dump(clf, 'svc_digits.model')


imgs = [img, l]
lbls = ['orig', 'threshold']
images_per_row = 5
# plt.gray()
plt.rcParams.update({"figure.facecolor":  (0.0, 0.0, 0.0, 0.4)})

lines = segment_line_in_image(l)

assert (len(lines) % 2) == 0 
print(lines)


rgba = cv2.cvtColor(l, cv2.COLOR_RGB2BGRA)
rgba[:,:,3] = 128
h, w = rgba.shape[:2]
overlay = np.zeros((h, w, 4), dtype = "uint8")

def draw_rect(overlay, x0, y0, x1, y1):
    overlay[y0:y1, x0:x1] = (255, 0, 0, 128)


lines_cnt = len(lines) // 2
for (y0, _), (y1, _), li in zip(lines[::2], lines[1::2], range(lines_cnt)):
    ln_img = l[y0:y1,:].copy()
    chars = segment_chars_in_line(ln_img)
    lh, _ = ln_img.shape
    print(lh)
    if li == lines_cnt - 4:
        chars_cnt = len(chars)//2 - 1
        char_imgs = []
        for (x0, _), (x1, _) in zip(chars[::2], chars[1::2]):
            char_img = ln_img[0:lh, x0:x1]
            draw_rect(overlay, x0, y0, x1, y1)
            if np.count_nonzero(char_img) < 9:
                print('its a dot')
            else:
                _, cw = char_img.shape
                char_img_ = np.zeros((lh, lh), dtype=np.uint8)
                x0 = (lh - cw)//2
                x1 = x0 + cw
                char_img_[:,x0:x1] = char_img
                char_img_ = np.pad(char_img_, 1)
                #assert char_img_.shape == (lh + 2, lh + 2)
                char_img_f_ = char_img_.astype(np.float64) / 16.0

                # from skimage.transform import resize
                char_img_f_ = resize(char_img_f_, (17, 17))
                char_imgs.append(char_img_f_)
        imgdata = np.array(char_imgs)
        imgdata1 = imgdata.reshape((chars_cnt, -1))
        predicted = clf.predict(imgdata1)
        print(predicted)
        for p, i in zip(predicted, char_imgs):
            imgs.append(i)
            lbls.append(p)

    # ln_img = trim_left_right(ln_img)
    # ln_img = np.pad(ln_img, 10)
    # imgs.append(ln_img)
    # lbls.append(f'line {y0},{y1}')

blend = alpha_blend(rgba, overlay)
imgs.append(blend)
lbls.append('blend')


# #rows = math.ceil(len(imgs)/images_per_row)
predicted = range(len(imgs))
_, axes = plt.subplots(nrows=1, ncols=len(imgs), figsize=(12, 3))
for ax, image, lbl in zip(axes, imgs, lbls):
    # ax.set_axis_off()
    # image = image.reshape(8, 8)
    # ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    # ax.set_title(f"Prediction: {prediction}")
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(lbl)

# predicted = clf.predict(X_test)
# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

# f = plt.figure(figsize=(18, 3))
# for k, l in enumerate(imgs):
#     f.add_subplot(rows, images_per_row, k + 1)
#     plt.imshow(l)
plt.show()

