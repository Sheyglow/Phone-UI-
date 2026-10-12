"""Cuts the screen out of each supplied phone mockup, with its rounded-corner mask.

Uses job/ui/edited/<name>.png when present (story text edits), else job/ui/<name>.png.
The screen is the region enclosed by the mockup's dark bezel.

Usage: python3 job/tools/extract_screens.py -> job/ui/screens/<name>.png (RGBA)
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(os.path.dirname(HERE), 'ui')
NAMES = ['luma-discover-daniel', 'luma-chat-daniel', 'wagon-discover-dating-jacob',
         'wagon-discover-friendship-ethan', 'wagon-match', 'wagon-chat-marcus']


def source(name):
    edited = os.path.join(UI, 'edited', name + '.png')
    return edited if os.path.exists(edited) else os.path.join(UI, name + '.png')


def extract(name):
    im = cv2.imread(source(name))
    h, w = im.shape[:2]
    dark = (im.max(2) < 14).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cand = [i for i in range(1, n) if st[i, 3] > h * .7 and st[i, 2] > w * .4]
    bez = max(cand, key=lambda i: st[i, 4])
    ff = (lab == bez).astype(np.uint8) * 255
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 128)
    inside = (ff == 0).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inside)
    scr = lab2 == 1 + np.argmax(st2[1:, 4])
    ys, xs = np.nonzero(scr)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.dstack([im[y0:y1, x0:x1], (scr[y0:y1, x0:x1] * 255).astype(np.uint8)])
    os.makedirs(os.path.join(UI, 'screens'), exist_ok=True)
    cv2.imwrite(os.path.join(UI, 'screens', name + '.png'), rgba)
    print(name, 'screen', (x0, y0, x1, y1))


if __name__ == '__main__':
    for n in NAMES:
        extract(n)
