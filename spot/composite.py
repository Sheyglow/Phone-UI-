"""Screen replacement: pins exact UI images onto tracked green screens.

The UI pixels are only perspective-warped, never regenerated, so text stays
exactly as supplied. Hands and fingers over the screen come from the key, so
they stay in front of the UI. Tracking markers are filled, and green spill is
removed from edges.
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_ASPECT = 924 / 2000


def load_track(name):
    with open(os.path.join(ROOT, 'footage', 'tracks', name + '.json')) as f:
        t = json.load(f)
    t['corners'] = np.array(t['corners'], dtype=np.float64)
    return t


def screen_aspect(corners):
    """Median width/height of the tracked screen (per screen)."""
    tl, tr, br, bl = [corners[:, :, i] for i in range(4)]
    w = (np.linalg.norm(tr - tl, axis=-1) + np.linalg.norm(br - bl, axis=-1)) / 2
    h = (np.linalg.norm(bl - tl, axis=-1) + np.linalg.norm(br - tr, axis=-1)) / 2
    return np.median(w / h, axis=0)


def fit_ui(ui, aspect):
    """Pad the UI with its own edge pixels so it fills a screen of this aspect without distortion."""
    h, w = ui.shape[:2]
    if aspect > UI_ASPECT:
        extra = int(round(h * aspect - w))
        return cv2.copyMakeBorder(ui, 0, 0, extra // 2, extra - extra // 2, cv2.BORDER_REPLICATE)
    extra = int(round(w / aspect - h))
    return cv2.copyMakeBorder(ui, extra // 2, extra - extra // 2, 0, 0, cv2.BORDER_REPLICATE)


_ui_cache = {}


def ui_image(name):
    if name not in _ui_cache:
        im = cv2.imread(os.path.join(ROOT, 'screens', name + '.webp'), cv2.IMREAD_COLOR)
        assert im is not None, name
        _ui_cache[name] = im
    return _ui_cache[name]


def slide(a, b, k):
    """iOS-style push: b enters from the right over a. k in 0..1."""
    h, w = a.shape[:2]
    e = 1 - (1 - k) ** 3
    x = int(round(w * (1 - e)))
    out = a.copy()
    shift = int(round(w * .3 * e))
    if shift:
        out[:, :w - shift] = a[:, shift:]
        out[:, w - shift:] = a[:, -1:]
    out = (out * (1 - .35 * e)).astype(np.uint8)
    out[:, x:] = b[:, :w - x]
    return out


def crossfade(a, b, k):
    return cv2.addWeighted(a, 1 - k, b, k, 0)


def composite(frame, quads, textures, gain=1.0, glare=0.06):
    """frame: BGR uint8 at output scale. quads: (S,4,2) in frame pixels. textures: list of BGR UI images."""
    H, W = frame.shape[:2]
    f = frame.astype(np.float32)
    d = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    out = f.copy()
    for q, tex in zip(quads, textures):
        pad = 30 + int(np.ptp(q[:, 0]) * .06)
        x0, y0 = np.floor(q.min(axis=0) - pad).astype(int)
        x1, y1 = np.ceil(q.max(axis=0) + pad).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        qq = (q - [x0, y0]).astype(np.float32)
        rw, rh = x1 - x0, y1 - y0

        # size the texture near its on-screen size first, so the warp doesn't alias
        qh = max(np.linalg.norm(qq[3] - qq[0]), np.linalg.norm(qq[2] - qq[1]))
        s = min(1.0, (qh * 1.25) / tex.shape[0])
        t = cv2.resize(tex, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else tex
        th, tw = t.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), qq)
        warped = cv2.warpPerspective(t, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped = cv2.GaussianBlur(warped, (0, 0), .55).astype(np.float32) * gain

        # matte: soft key, limited to the tracked screen
        # matte margins scale with the phone's size on screen
        qw = np.linalg.norm(qq[1] - qq[0])
        ks = lambda px: np.ones((max(3, int(px * qw / 300) | 1),) * 2, np.uint8)
        qmask = np.zeros((rh, rw), np.uint8)
        cv2.fillConvexPoly(qmask, np.round(qq).astype(np.int32), 255, lineType=cv2.LINE_AA)
        inner = cv2.erode(qmask, ks(5)) > 128
        dd = d[y0:y1, x0:x1]
        a = np.clip((dd - 10) / 28, 0, 1)
        a *= cv2.dilate(qmask, ks(15)).astype(np.float32) / 255
        # fill tracking markers: holes that green surrounds on all sides. Fingers reach
        # the screen from outside, so their ring is never all green and they stay.
        holes = ((a < .5) & inner).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(holes)
        qarea = cv2.contourArea(qq)
        keyed = a > .5
        fill = np.zeros((rh, rw), bool)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area > qarea * .03:
                continue
            bx0, by0, bx1, by1 = max(x - 6, 0), max(y - 6, 0), min(x + w + 6, rw), min(y + h + 6, rh)
            comp = (lab[by0:by1, bx0:bx1] == i).astype(np.uint8)
            ring = (cv2.dilate(comp, np.ones((9, 9), np.uint8)) > 0) & (comp == 0)
            if keyed[by0:by1, bx0:bx1][ring].mean() > .85:
                grown = cv2.dilate(comp, np.ones((5, 5), np.uint8)) > 0
                fill[by0:by1, bx0:bx1] |= grown
        a[fill] = 1
        a = cv2.GaussianBlur(a, (0, 0), max(.7, .75 * qw / 300))[..., None]

        # despill hands and bezel around the screen
        src = out[y0:y1, x0:x1]
        near = cv2.dilate(qmask, ks(15)) > 0
        g_lim = (src[..., 0] + src[..., 2]) / 2 + 4  # average despill: pulls green edges back to skin
        src[..., 1] = np.where(near, np.minimum(src[..., 1], g_lim), src[..., 1])

        # glare fixed to the room, so it slides across the screen as the phone moves
        yy, xx = np.mgrid[y0:y1, x0:x1]
        band = np.exp(-(((xx * .8 + yy * .45) / W - .62) / .07) ** 2)[..., None] * 255 * glare
        screen = np.clip(warped + band, 0, 255)

        out[y0:y1, x0:x1] = src * (1 - a) + screen * a
    return np.clip(out, 0, 255).astype(np.uint8)
