"""Finds thumbs and fingers over each tracked screen, in screen coordinates.

Each frame is rectified into a 1000x2000 screen space with the tracked quad.
Occluders are skin-coloured pixels (or, on blank white screens, anything that
isn't white) connected to the screen border, since a thumb always reaches in
from outside. For each occluder blob, the tip is the point farthest from where
it enters the screen.

Usage: python3 job/tools/thumbs.py [screen ...]
Writes job/tracks/<screen>-thumbs.json: per frame, a list of tips
[x, y, area] in screen units (x 0..1000, y 0..2000).
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
SW, SH = 1000, 2000


def occluders(rect, mode):
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    b, r = rect[..., 0].astype(int), rect[..., 2].astype(int)
    if mode == 'white':
        occ = ~((s < 22) & (v > 225))
    else:
        occ = (r - b > 30) & (s > 45) & (v > 60) & ((h < 25) | (h > 165))
    occ = occ.astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return occ


def tips(occ):
    n, lab, st, _ = cv2.connectedComponentsWithStats(occ)
    out = []
    border = np.zeros_like(occ, bool)
    border[:12], border[-12:], border[:, :12], border[:, -12:] = True, True, True, True
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 4000:
            continue
        comp = lab == i
        entry = comp & border
        if not entry.any():
            continue  # not connected to the outside: UI content, not a thumb
        ey, ex = np.nonzero(entry)
        e = np.array([ex.mean(), ey.mean()])
        ys, xs = np.nonzero(comp)
        d = (xs - e[0]) ** 2 + (ys - e[1]) ** 2
        j = np.argmax(d)
        out.append([int(xs[j]), int(ys[j]), int(st[i, cv2.CC_STAT_AREA])])
    return out


def run(name):
    t = json.load(open(os.path.join(JOB, 'tracks', name + '.json')))
    q = np.array(t['corners'], np.float32)
    cap = cv2.VideoCapture(os.path.join(JOB, 'source', 'main.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t['first'])
    dst = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
    rows = []
    for i in range(len(q)):
        ok, fr = cap.read()
        M = cv2.getPerspectiveTransform(q[i], dst)
        rect = cv2.warpPerspective(fr, M, (SW, SH))
        rows.append(tips(occluders(rect, t['mode'])))
    with open(os.path.join(JOB, 'tracks', name + '-thumbs.json'), 'w') as f:
        json.dump({'first': t['first'], 'tips': rows}, f)
    for i in range(0, len(rows), 2):
        print(t['first'] + i, rows[i])


if __name__ == '__main__':
    for n in sys.argv[1:]:
        run(n)
