"""Tracks green phone screens in the supplied footage.

For each frame it keys the screen green, fits a straight line to each of the
four screen edges (so rounded corners, tracking markers and fingers crossing
the screen don't pull the corners), and intersects those lines to get a sharp
quad. A second pass rejects outliers and smooths the corners over time.

Usage: python3 spot/track.py            # tracks every clip in CLIPS
Writes footage/tracks/<clip>.json with per-frame corners [TL, TR, BR, BL].
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# clip name -> (number of screens, key threshold on g - max(r, b))
CLIPS = {
    'gs-pov-lap': (1, 45),
    'gs-park-pair': (2, 55),
    'gs-couch-evening': (1, 38),
    'gs-autumn-hand': (1, 90),
}


def green_dominance(frame):
    f = frame.astype(np.int16)
    return f[..., 1] - np.maximum(f[..., 0], f[..., 2])


def order_corners(pts):
    """Return corners as TL, TR, BR, BL for a portrait screen."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coords, starting near -pi (left)
    tl = np.argmin(pts.sum(axis=1))
    return np.roll(pts, -tl, axis=0)


def densify(poly, step=2.0):
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(n):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def rough_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.12, 40):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float64)
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return box.astype(np.float64)


def line_intersect(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    b = np.array([x2 - x1, y2 - y1])
    s = np.linalg.solve(A, b)[0]
    return np.array([x1 + s * vx1, y1 + s * vy1])


def fit_quad(mask_pts):
    hull = cv2.convexHull(mask_pts.astype(np.int32))
    quad = order_corners(rough_quad(hull))
    dense = densify(hull.reshape(-1, 2).astype(np.float64))
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        u = ab / L
        n = np.array([-u[1], u[0]])
        rel = dense - a
        t = rel @ u / L
        dist = np.abs(rel @ n)
        # wide band first: on rounded screens the rough corners sit inside the true edge
        mid = (t > .18) & (t < .82)
        sel = dense[mid & (dist < max(8, L * .07))]
        if len(sel) < 8:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        # then refit on the points that sit close to that line
        d2 = np.abs((dense[:, 0] - x0) * vy - (dense[:, 1] - y0) * vx)
        sel = dense[mid & (d2 < 3)]
        if len(sel) >= 8:
            vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        lines.append((vx, vy, x0, y0))
    try:
        corners = np.array([line_intersect(lines[i - 1], lines[i]) for i in range(4)])
    except np.linalg.LinAlgError:
        return quad
    # line i runs from corner i to i+1, so corner i = intersection of lines i-1 and i
    if np.abs(corners - quad).max() > 0.25 * np.linalg.norm(quad[2] - quad[0]):
        return quad
    return corners


def detect(frame, n_screens, thresh):
    d = green_dominance(frame)
    m = (d > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    grouped = cv2.dilate(m, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grouped)
    if n <= 1:
        return None
    order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:n_screens] + 1
    quads = []
    for i in order:
        ys, xs = np.nonzero((lab == i) & (m > 0))
        if len(xs) < 500:
            return None
        quads.append(fit_quad(np.stack([xs, ys], axis=1)))
    quads.sort(key=lambda q: q[:, 0].mean())  # left to right
    return np.array(quads)


def smooth(track, sigma=1.3):
    """track: (F, S, 4, 2) with NaN for missing frames."""
    F = track.shape[0]
    flat = track.reshape(F, -1)
    # fill gaps, then reject spikes against a local median
    idx = np.arange(F)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = ~np.isnan(col)
        col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        med = np.array([np.median(col[max(0, i - 3):i + 4]) for i in range(F)])
        bad = np.abs(col - med) > 10
        col[bad] = med[bad]
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(flat, ((r, r), (0, 0)), mode='edge')
    out = np.stack([np.convolve(padded[:, j], k, mode='valid') for j in range(flat.shape[1])], axis=1)
    return out.reshape(track.shape)


def track_clip(name, n_screens, thresh):
    cap = cv2.VideoCapture(os.path.join(ROOT, 'footage', name + '.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        q = detect(frame, n_screens, thresh)
        raw.append(q if q is not None and len(q) == n_screens else np.full((n_screens, 4, 2), np.nan))
    raw = np.array(raw, dtype=np.float64)
    missing = int(np.isnan(raw[:, 0, 0, 0]).sum())
    sm = smooth(raw.copy())
    out = os.path.join(ROOT, 'footage', 'tracks', name + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'fps': fps, 'frames': len(raw), 'screens': n_screens, 'thresh': thresh,
                   'corners': np.round(sm, 2).tolist()}, f)
    print(f'{name}: {len(raw)} frames, {missing} undetected, fps {fps:.3f}')


if __name__ == '__main__':
    names = sys.argv[1:] or list(CLIPS)
    for name in names:
        track_clip(name, *CLIPS[name])
