"""Tracks the phone screens in the Wagon commercial edit (job/source/main.mp4).

Each screen is keyed by its brightness: the AI phones show either blank white
screens or a white Luma UI. Every frame, the bright region's edges are fitted
with straight lines and intersected into a quad. If the screen is partly out of
frame or the key looks wrong, the previous quad is carried forward with optical
flow instead. The corners are then smoothed over time.

Usage: python3 job/tools/track_screens.py [screen ...]
Writes job/tracks/<screen>.json with per-frame corners [TL, TR, BR, BL] in
source pixels (854x480).
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(JOB), 'spot'))
from track import fit_quad, order_corners  # noqa: E402

SRC = os.path.join(JOB, 'source', 'main.mp4')
FPS = 24

# name -> (first frame, last frame inclusive, roi x0, x1, mode)
# mode 'white': blank white screen. 'ui': white AI UI with text, bubbles and a keyboard.
SCREENS = {
    's01-luma': (17, 125, 0, 854, 'ui'),
    's02-luma-chat': (126, 191, 0, 854, 'ui'),
    's10-left': (1936, 2023, 0, 418, 'white'),
    's10-right': (1936, 2023, 420, 854, 'white'),
    's11-her': (2025, 2093, 0, 854, 'white'),
    's12-his': (2094, 2156, 0, 854, 'white'),
}


def bright_mask(frame, mode):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    if mode == 'white':
        m = (s < 22) & (v > 225)
    else:
        # white UI plus its grey keyboard, blue bubbles and dark text
        m = ((s < 40) & (v > 175)) | ((hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (s > 90) & (v > 150))
    m = m.astype(np.uint8)
    k = 9 if mode == 'ui' else 5
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


def bezel_fit(comp, frame, rough):
    """Refit the quad's four sides using only boundary points that border the dark bezel.

    Where a thumb covers the screen edge, the boundary borders skin instead and is ignored,
    so an occluded corner no longer tilts its sides.
    """
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=len).reshape(-1, 2).astype(np.float64)
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
    c = rough.mean(0)
    H, W = v.shape
    keep = []
    for p in pts:
        d = p - c
        d /= np.linalg.norm(d) + 1e-9
        x, y = np.round(p + d * 3).astype(int)  # just outside the white area
        keep.append(0 <= x < W and 0 <= y < H and v[y, x] < 80)
    pts = pts[np.array(keep)]
    lines = []
    for i in range(4):
        a, b = rough[i], rough[(i + 1) % 4]
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        rel = pts - a
        t = rel @ u / L
        sel = pts[(t > .12) & (t < .88) & (np.abs(rel @ n) < max(5, L * .05))]
        if len(sel) < 10:
            return rough
        lines.append(cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel())
    out = []
    for i in range(4):
        (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = lines[i - 1], lines[i]
        A = np.array([[vx1, -vx2], [vy1, -vy2]])
        if abs(np.linalg.det(A)) < 1e-6:
            return rough
        s = np.linalg.solve(A, [x2 - x1, y2 - y1])[0]
        out.append([x1 + s * vx1, y1 + s * vy1])
    out = np.array(out)
    return out if np.abs(out - rough).max() < 20 else rough


def key_quad(frame, mode, x0, x1, ref_area):
    m = bright_mask(frame, mode)
    m[:, :x0] = 0
    m[:, x1:] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, 'none', None
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    x, y, w, h, area = st[k]
    H, W = m.shape
    if x <= max(x0, 1) or y <= 1 or x + w >= min(x1, W) - 1 or y + h >= H - 1:
        return None, 'edge', None
    ys, xs = np.nonzero(lab == k)
    q = order_corners(fit_quad(np.stack([xs, ys], 1)))
    comp = (lab == k).astype(np.uint8)
    if mode == 'white':
        q = bezel_fit(comp, frame, q)
    if ref_area and not (.7 < cv2.contourArea(q.astype(np.float32)) / ref_area < 1.4):
        return None, 'area', None
    return q, 'ok', comp


def flow_quad(prev_gray, gray, quad):
    """Carry the quad forward with a similarity transform fitted to tracked features on the phone."""
    mask = np.zeros_like(prev_gray)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    pts = cv2.goodFeaturesToTrack(prev_gray, 200, .01, 4, mask=mask)
    if pts is None or len(pts) < 6:
        return quad
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    ok = ok.ravel() == 1
    if ok.sum() < 6:
        return quad
    M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=2)
    if M is None:
        return quad
    return quad @ M[:, :2].T + M[:, 2]


def smooth(q, sigma=1.2):
    F = len(q)
    flat = q.reshape(F, -1).copy()
    r = int(3 * sigma)
    k = np.exp(-.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(flat, ((r, r), (0, 0)), mode='edge')
    out = np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(flat.shape[1])], 1)
    return out.reshape(q.shape)


def track(name):
    f0, f1, x0, x1, mode = SCREENS[name]
    cap = cv2.VideoCapture(SRC)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(f0, f1 + 1):
        ok, fr = cap.read()
        frames.append(fr)
    # reference area: median over frames where the key is clean
    areas = []
    for fr in frames[::3]:
        q, why, _ = key_quad(fr, mode, x0, x1, None)
        if q is not None:
            areas.append(cv2.contourArea(q.astype(np.float32)))
    ref_area = float(np.median(areas)) if areas else None

    keyed, status, comps = [], [], []
    for fr in frames:
        q, why, comp = key_quad(fr, mode, x0, x1, ref_area)
        keyed.append(q)
        status.append(why)
        comps.append(comp)
    # reject keyed quads whose shape (width/height) strays from the shot's typical shape
    def shape(q):
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        return w / h
    med = np.median([shape(q) for q in keyed if q is not None])
    for i, q in enumerate(keyed):
        if q is not None and abs(shape(q) / med - 1) > .1:
            keyed[i], status[i] = None, 'shape'
    # forward pass with flow, snapping to the keyed quad whenever it is valid
    grays = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fr in frames]
    start = next(i for i, q in enumerate(keyed) if q is not None)
    quads = [None] * len(frames)
    quads[start] = keyed[start]
    for i in range(start + 1, len(frames)):
        pred = flow_quad(grays[i - 1], grays[i], quads[i - 1])
        if keyed[i] is not None and mode == 'white':
            # refit the sides from the previous frame's outline, so a thumb resting on a
            # corner can't tilt them (the per-frame rough outline would follow the thumb)
            quads[i] = bezel_fit(comps[i], frames[i], quads[i - 1])
        else:
            quads[i] = keyed[i] if keyed[i] is not None else pred
        if keyed[i] is None:
            status[i] = 'flow-' + status[i]
    for i in range(start - 1, -1, -1):  # backward for any head frames without a key
        quads[i] = flow_quad(grays[i + 1], grays[i], quads[i + 1])
        status[i] = 'flow-back'
    q = smooth(np.array(quads, np.float64))
    os.makedirs(os.path.join(JOB, 'tracks'), exist_ok=True)
    with open(os.path.join(JOB, 'tracks', name + '.json'), 'w') as f:
        json.dump({'first': f0, 'last': f1, 'mode': mode, 'corners': np.round(q, 2).tolist(), 'status': status}, f)
    from collections import Counter
    print(f'{name}: {len(frames)} frames, ref area {ref_area}, {dict(Counter(s.split("-")[0] for s in status))}')


if __name__ == '__main__':
    for n in sys.argv[1:] or list(SCREENS):
        track(n)
