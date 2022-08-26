
import os
from re import X
import sys

import cv2
from joblib import dump, load
from skimage.transform import resize
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from functools import reduce
from operator import mul
from main import fill_gaps
import math


DIGITS_MODEL_FEATURE_MAX_VALUE = 16.0
DIGITS_MODEL_FILENAME = './svc_digits.model'
DIGITS_MODEL_IMAGE_SHAPE = (17, 17)
DIGITS_MODEL_FEATURES_SIZE = reduce(mul, DIGITS_MODEL_IMAGE_SHAPE)
GAPS_COUNT = {
    '0': 1,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 1,
    '5': 0,
    '6': 1,
    '7': 0,
    '8': 2,
    '9': 1,
}
CONNECTED_COMPONENTS_CROPPED = {
    '0': [4, 5],
    '1': [2],
    '2': [4],
    '3': [4],
    '4': [4],
    '5': [4],
    '6': [5, 6],
    '7': [3],
    '8': [7, 8],
    '9': [5, 6],
}
CONFUSIONS = {
    '1': ['4'],
    '4': ['1'],
    '6': ['8', '5'],
    '8': ['6'],
    '5': ['6']
}


from functools import partial

# :)
forward = lambda x: x

class HSegDir(Enum):
    LEFT_TO_RIGHT = partial(forward)
    RIGHT_TO_LEFT = partial(reversed)

class VSegDir(Enum):
    TOP_TO_BOTTOM = partial(forward)
    BOTTOM_TO_TOP = partial(reversed)

class OCR:

    def __init__(self):
        pass

    def segmentize(self, mat: np.ndarray,
                   vdir: VSegDir = VSegDir.TOP_TO_BOTTOM,
                   hdir: HSegDir = HSegDir.LEFT_TO_RIGHT):
        for ln, (ln_img, y0, y1) in enumerate(segment_line_in_image(l)):
            for cn, char_img, x0, x1 in enumerate(segment_chars_in_line(ln_img)):
                yield char_img, ln, cn, x0, y0, x1, y1

def alpha_blend(image, overlay):
    srcRGB = image[...,:3]
    dstRGB = overlay[...,:3]
    srcA = image[...,3]/255.0
    dstA = overlay[...,3]/255.0
    outA = srcA + dstA*(1-srcA)
    outRGB = (srcRGB*srcA[...,np.newaxis] + dstRGB*dstA[...,np.newaxis]*(1-srcA[...,np.newaxis])) / outA[...,np.newaxis]
    outRGBA = np.dstack((outRGB,outA*255)).astype(np.uint8)
    return outRGBA


def segment_line_in_image(image: np.ndarray, vdir: VSegDir = VSegDir.TOP_TO_BOTTOM):
    bg, lines, h, y0 = True, [], image.shape[0], 0
    for i in vdir.value(range(h)):
        pbg = bg
        bg = np.count_nonzero(image[i, :]) == 0
        if (not bg) and pbg:
            y0 = i
        if (not pbg) and bg:
            correction = 1 if vdir == VSegDir.BOTTOM_TO_TOP else 0
            reng = tuple(vdir.value((y0 + correction, i + correction)))
            yield (image[slice(*reng), :].copy(), *reng)

def segment_chars_in_line(ln_image: np.ndarray, hdir: HSegDir = HSegDir.LEFT_TO_RIGHT):
    bg = True
    h, w = ln_image.shape[:2]
    x0 = 0
    for i in hdir.value(range(w)):
        pbg = bg
        bg = np.count_nonzero(ln_image[:, i]) == 0
        if (not bg) and pbg:
            x0 = i
        if (not pbg) and bg:
            correction = 1 if hdir == HSegDir.RIGHT_TO_LEFT else 0
            reng = tuple(hdir.value((x0 + correction, i + correction)))
            yield (ln_image[:, slice(*reng)].copy(), *reng)


# todo: update, segmentation is broken
def prepare_train_data():
    imgsfn = [(f'sample{i}.png', f'sample{i}.txt') for i in range(4)]
    assert map(lambda p: os.path.exists(p[0]) and os.path.exists(p[1]), imgsfn)
    chs = []
    max_lhs = []
    target = []
    for ifn, tfn in imgsfn:
        X = cv2.imread(ifn)
        luv = cv2.cvtColor(X, cv2.COLOR_BGR2LUV)
        l = luv[:,:,0]
        l = (l > 128).astype(np.uint8)
        l *= 255
        lines = segment_line_in_image(l)
        with open(tfn) as f:
            st = f.read().strip()
        txt_lines = st.split('\n')
        if txt_lines[-1] == '':
            txt_lines.pop()
        assert (len(lines) // 2) == len(txt_lines)
        max_lh = max([y1 - y0 for (y0, _), (y1, _) in zip(lines[::2], lines[1::2])])
        max_lhs.append(max_lh)
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
    assert set(target) == set('0123456789')
    n_samples = len(chs)
    chs_ = np.array(chs)
    data = chs_.reshape((n_samples, -1))
    return data, target

if __name__ == '__main__':
    model_exists = os.path.exists(DIGITS_MODEL_FILENAME)

    if model_exists:
        clf = load('./svc_digits.model')
    else:
        data, target = prepare_train_data()
        clf = svm.SVC(gamma=0.001)
        clf.fit(data, target)
        dump(clf, './svc_digits.model')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.rcParams.update({"figure.facecolor": (0.0, 0.0, 0.0, 0.4)})


    test_file_name = 'sample1'
    img = cv2.imread(f'{test_file_name}.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    l = luv[:,:,0]
    l = (l > 128).astype(np.uint8)
    l *= 255
    h, w = l.shape
    if 'show_origin' in locals():
        imgs = [img, l]
        lbls = ['orig', 'threshold']
    # imgs, lbls = [], []
    images_per_row = 5
    rgba = cv2.cvtColor(l, cv2.COLOR_RGB2BGRA)
    rgba[:,:,3] = 128
    h, w = rgba.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)


    def draw_rect(overlay, x0, y0, x1, y1):
        overlay[y0:y1, x0:x1] = (255, 0, 0, 128)

    def to_model_format(char_img):
        ch, cw = char_img.shape
        padded = np.zeros(DIGITS_MODEL_IMAGE_SHAPE, dtype=np.uint8)
        lh = DIGITS_MODEL_IMAGE_SHAPE[0]
        x0 = (lh - cw) // 2
        x1 = x0 + cw
        y0 = (lh - ch) // 2
        y1 = y0 + ch
        padded[y0:y1, x0:x1] = char_img
        padded = padded.astype(np.float64) / DIGITS_MODEL_FEATURE_MAX_VALUE
        padded = resize(np.pad(padded, 1), DIGITS_MODEL_IMAGE_SHAPE)
        assert padded.shape[:2] == DIGITS_MODEL_IMAGE_SHAPE
        assert padded.max() <= DIGITS_MODEL_FEATURE_MAX_VALUE
        return padded

    full_txt = ''
    for li, (ln_img, y0, y1) in enumerate(segment_line_in_image(l, VSegDir.TOP_TO_BOTTOM)):
        lh = ln_img.shape[0]
        char_imgs = []
        orig_char_imgs = []
        for char_img, x0, x1 in segment_chars_in_line(ln_img, HSegDir.LEFT_TO_RIGHT):
            #print(f'char: {x0}:{x1}')
            draw_rect(overlay, x0, y0, x1, y1)
            # skip dot char, it has small amount if non-zero pixels
            if np.count_nonzero(char_img) < 9:
                continue
            if 'show_char_images' in locals():
                imgs.append(char_img)
                lbls.append('')
            orig_char_imgs.append(char_img)
            char_imgs.append(to_model_format(char_img))
        chars_cnt = len(char_imgs)
        imgdata = np.array(char_imgs)
        imgdata = imgdata.reshape((chars_cnt, -1))
        predicted = clf.predict(imgdata)

        # eliminate confusions by analysing connected components
        def get_gaps_count(digit: np.ndarray) -> int:
            img_pad = np.pad(digit, 1)
            img_inv_pad = np.invert(img_pad)
            filled, _ = fill_gaps(img_pad, conn=4)
            n, o, _, _ = cv2.connectedComponentsWithStats(img_inv_pad, connectivity=4)
            cnt_gaps = -1
            for j in range(n):
                c = (o == j).astype(np.uint8)
                if c.max() != 255:
                    c *= 255
                masked = cv2.bitwise_and(c, c, mask=filled)
                if np.count_nonzero(masked) != 0:
                    cnt_gaps += 1
            return cnt_gaps


        def get_gap_count_mapping(ch: str):
            assert len(ch) == 1
            m = list(map(lambda x: GAPS_COUNT[x],CONFUSIONS[ch]))
            if len(set(m)) == len(CONFUSIONS[ch]):
                return dict(zip(m, CONFUSIONS[ch]))
            else:
                return None

        for i, ch in enumerate(predicted):
            if ch in CONFUSIONS:
                img = orig_char_imgs[i]
                cnt_gaps = get_gaps_count(img)
                if GAPS_COUNT[ch] != cnt_gaps:
                    # imgs.append(img)
                    # lbls.append(f"err: {ch} g{cnt_gaps}")
                    m = get_gap_count_mapping(ch)
                    if m and cnt_gaps in m:
                        predicted[i] = m[cnt_gaps]
                        # print('correction', ch, m[cnt_gaps])
                    else:
                        pass
                        # print('could not find correction', ch)

        st = ''.join(predicted)
        #st = st[:1] + '.' + st[1:]
        full_txt += st + '\n'
        #print(st)

        if 'show_line_imgs' in locals():
            imgs.append(ln_img)
            lbls.append(st)
        if 'break_at_first_line' in locals():
            break
    assert 'li' in locals()
    lines_cnt = li + 1
    print(f'lines: {lines_cnt}')

    blend = alpha_blend(rgba, overlay)
    if 'show_blend' in locals():
        imgs.append(blend)
        lbls.append('blend')

    # todo: compare to true data
    txt_file_name = f'{test_file_name}.txt'
    if os.path.exists(txt_file_name):
        with open(txt_file_name) as f:
            expected_txt = f.read()

        actual = full_txt.strip(' \n').split('\n')
        expected = expected_txt.replace('.', '').strip(' \n').split('\n')

        errs = 0
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                print(f'{i} line,', end='')
            l = list(filter(None, [f'[{j}]{ac}/{ec}' for j, (ac, ec) in enumerate(zip(a, e)) if ac != ec]))
            print(*l, sep=', ', end='')
            errs += len(l)
            if len(l):
                print()

        print(f'errors: {errs}')

    if 'imgs' in locals():
        assert len(imgs) == len(lbls)
        # imgs_per_row = 6
        # nrows = math.ceil(len(imgs) / imgs_per_row)
        # _, axes = plt.subplots(nrows=nrows, ncols=imgs_per_row, figsize=(2 * imgs_per_row, 2 * nrows))
        if len(imgs):
            _, axes = plt.subplots(nrows=1, ncols=len(imgs), figsize=(12, 2))
            # for ax, image, lbl in zip(axes, imgs, lbls):
            #     ax.set_axis_off()
            #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            #     ax.set_title(lbl)
            for ax, image, lbl in zip(axes, imgs, lbls):
                ax.set_axis_off()
                ax.imshow(image, 'gray', interpolation="nearest")
                ax.set_title(lbl)

            plt.show()

