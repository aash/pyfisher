import cv2
import numpy as np
from ocr import *


def test_ocr_single_line_bottom_to_top():
    for test_file_name in [f'sample{i}' for i in range(4)] + ['66', '33', '55']:
        img = cv2.imread(f'{test_file_name}.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        l = luv[:,:,0]
        l = (l > 128).astype(np.uint8)
        l *= 255
        ocr = OCR()
        actual = ''.join(ocr(l, vdir=VSegDir.BOTTOM_TO_TOP))
        txt_file_name = f'{test_file_name}.txt'
        with open(txt_file_name) as f:
            expected_txt = f.read()
        expected = expected_txt.strip('\n').split('\n')
        assert expected[-1] == actual
