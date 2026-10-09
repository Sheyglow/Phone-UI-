"""Renders the Wagon commercial: the supplied edit with every phone screen replaced.

- Drops S1's first swipe (source frames 0-16), so the film starts on frame 17.
- Upscales the 854x480 edit to 1920x1080, then composites at that size so the UI
  stays sharp.
- S1: Luma Discover. Edward is swiped away, Daniel appears, and the thumb's tap
  fills the like heart blue.
- S2: Luma chat. "See you at 8 😊" is typed letter by letter on the keyboard, then sent.
- S9: Wagon logo on the split line, animated split light, and floating Discover phones.
- S10: Dating and Friendship Discover. The card follows her thumb's drag, and each
  heart is pressed when the thumb reaches it. The logo and split light carry on.
- S11: It's a Match. The centre heart bursts when her thumb reaches it.
- S12: Marcus chat. His typed reply sends when his thumb taps.
- End: the Wagon card, animated element by element.

Usage: python3 job/tools/render_job.py [--stills] [--from F --to F] [--out path]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ui_luma  # noqa: E402

SRC = os.path.join(JOB, 'source', 'main.mp4')
W, H, FPS = 1920, 1080, 24
SX, SY = W / 854, H / 480
START = 17            # first source frame kept (S1's first swipe removed)
LAST_SRC = 2640       # last source frame
END_FRAMES = 120      # end card length (5 s)
XFADE = 12            # S16 into the end card
STATES = os.path.join(JOB, 'ui', 'states')
MG = os.path.join(JOB, 'mg')


def ease_out(k):
    k = min(max(k, 0.0), 1.0)
    return 1 - (1 - k) ** 3


def ease_io(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3 - 2 * k)


def spring(k):
    """0 -> overshoot -> 1."""
    k = min(max(k, 0.0), 1.0)
    return 1 - math.exp(-6 * k) * math.cos(9 * k)


def img(name, flags=cv2.IMREAD_COLOR):
    path = name if os.path.isabs(name) else os.path.join(STATES, name + '.png')
    a = cv2.imread(path, flags)
    assert a is not None, path
    return a


def load_track(name):
    t = json.load(open(os.path.join(JOB, 'tracks', name + '.json')))
    q = np.array(t['corners'], np.float64)
    q[..., 0] *= SX
    q[..., 1] *= SY
    th = os.path.join(JOB, 'tracks', name + '-thumbs.json')
    tips = json.load(open(th))['tips'] if os.path.exists(th) else None
    return t['first'], q, tips


# ------------------------------------------------------------------ compositing
def warp_into(frame, tex, quad, roi):
    x0, y0, x1, y1 = roi
    qq = (quad - [x0, y0]).astype(np.float32)
    qh = max(np.linalg.norm(qq[3] - qq[0]), np.linalg.norm(qq[2] - qq[1]))
    s = min(1.0, qh * 1.3 / tex.shape[0])
    t = cv2.resize(tex, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else tex
    th, tw = t.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), qq)
    flags = cv2.INTER_LINEAR
    if t.ndim == 2:
        return cv2.warpPerspective(t, M, (x1 - x0, y1 - y0), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpPerspective(t, M, (x1 - x0, y1 - y0), flags=flags, borderMode=cv2.BORDER_REPLICATE)


def roi_of(quad, pad=24):
    x0, y0 = np.floor(quad.min(0) - pad).astype(int)
    x1, y1 = np.ceil(quad.max(0) + pad).astype(int)
    return max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)


def skin_occluder(reg, inside):
    """Thumbs over the screen: skin (and nails) connected to skin outside the screen."""
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(np.int16) for i in range(3)]
    b, r = reg[..., 0].astype(np.int16), reg[..., 2].astype(np.int16)
    warm = (h < 25) | (h > 165)
    strong = warm & (r - b > 30) & (s > 45) & (v > 60)
    weak = warm & (r - b > 12) & (s > 16) & (v > 70)
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak)
    if n <= 1:
        return np.zeros(reg.shape[:2], np.float32)
    out_in = ~inside
    has_strong = np.bincount(lab[strong], minlength=n) > 30
    outside = np.bincount(lab[out_in & (weak > 0)], minlength=n) > 30
    keep = has_strong & outside
    keep[0] = False
    occ = keep[lab].astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    occ = cv2.erode(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.GaussianBlur(occ.astype(np.float32), (0, 0), 1.6)


def white_key(reg):
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    return np.clip((26 - s) / 14, 0, 1) * np.clip((v - 200) / 30, 0, 1)


def fill_enclosed(a, inside, max_frac=.03):
    """Treat small non-white islands fully enclosed by white (the AI dynamic island) as screen."""
    holes = ((a < .5) & inside).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes)
    total = inside.sum()
    out = a.copy()
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area > total * max_frac:
            continue
        comp = (lab[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, np.ones((7, 7), np.uint8)) > 0) & (comp == 0)
        sub = a[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4]
        if (sub[ring] > .5).mean() > .9:
            grown = cv2.dilate(comp, np.ones((5, 5), np.uint8)) > 0
            out[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4][grown] = 1
    return out


GLARE_SEED = np.random.default_rng(11)


def finish_screen(warped, quad, roi):
    """Settle the crisp UI into the soft upscaled footage: slight blur, glare, grain."""
    x0, y0, x1, y1 = roi
    w = cv2.GaussianBlur(warped, (0, 0), .75).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    band = np.exp(-(((xx * .7 + yy * .5) / W - .78) / .09) ** 2)[..., None] * 14
    noise = GLARE_SEED.normal(0, 2.2, w.shape[:2])[..., None]
    return np.clip(w + band + noise, 0, 255)


def composite_screen(frame, quad, tex, mode, shape_mask=None):
    roi = roi_of(quad)
    x0, y0, x1, y1 = roi
    reg = frame[y0:y1, x0:x1]
    qmask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(qmask, np.round((quad - [x0, y0]) * 4).astype(np.int32), 255, cv2.LINE_AA, shift=2)
    inside = qmask > 127
    warped = warp_into(frame, tex, quad, roi)
    if mode == 'white':
        a = white_key(reg) * (cv2.dilate(qmask, np.ones((5, 5), np.uint8)) / 255.0)
        a = fill_enclosed(a, cv2.erode(qmask, np.ones((9, 9), np.uint8)) > 127)
        # choke the thumbs by ~3 px: their edge pixels are half white in the upscaled footage,
        # which otherwise leaves a jagged white halo. Then soften the edge.
        # (the occluder is taken over the whole region, so the screen's own edge isn't eroded)
        occ = (a < .5).astype(np.uint8)
        occ = cv2.erode(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        hard = np.where(inside, 1 - occ, (a > .5)).astype(np.float32)
        a = cv2.GaussianBlur(hard, (0, 0), 1.4) * (cv2.dilate(qmask, np.ones((5, 5), np.uint8)) / 255.0)
    else:
        shape = warp_into(frame, shape_mask, quad, roi).astype(np.float32) / 255
        shape = cv2.dilate(shape, np.ones((3, 3), np.uint8))
        a = shape * (1 - skin_occluder(reg, inside))
    a = a[..., None]
    screen = finish_screen(warped, quad, roi)
    frame[y0:y1, x0:x1] = np.clip(reg * (1 - a) + screen * a, 0, 255).astype(np.uint8)


def over(base, layer, x, y, alpha_mult=1.0):
    """Alpha-composite an RGBA layer onto base at (x, y), clipped to the frame."""
    h, w = layer.shape[:2]
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, base.shape[1]), min(y + h, base.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    L = layer[Y0 - y:Y1 - y, X0 - x:X1 - x].astype(np.float32)
    a = L[..., 3:] / 255 * alpha_mult
    B = base[Y0:Y1, X0:X1].astype(np.float32)
    base[Y0:Y1, X0:X1] = np.clip(B * (1 - a) + L[..., :3] * a, 0, 255).astype(np.uint8)


def add(base, rgb, x, y, mult=1.0):
    """Additive (light) blend of an RGB layer."""
    h, w = rgb.shape[:2]
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, base.shape[1]), min(y + h, base.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    L = rgb[Y0 - y:Y1 - y, X0 - x:X1 - x].astype(np.float32) * mult
    base[Y0:Y1, X0:X1] = np.clip(base[Y0:Y1, X0:X1].astype(np.float32) + L, 0, 255).astype(np.uint8)


def transform_rgba(layer, scale=1.0, angle=0.0):
    h, w = layer.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(layer, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ------------------------------------------------------------------ S1: Luma Discover
class S1:
    SWIPE = (49, 57)      # Edward's card flies off with the thumb's drag
    LIKE = 62             # the thumb presses the heart

    def __init__(self):
        self.first, self.q, _ = load_track('s01-luma')
        self.daniel_empty = img('luma-daniel-empty')
        self.daniel_liked = img('luma-daniel-liked')
        self.edward = img('luma-edward')
        self.card = img('luma-edward-card', cv2.IMREAD_UNCHANGED)
        self.heart = img('luma-heart-layer', cv2.IMREAD_UNCHANGED)
        a = cv2.imread(os.path.join(JOB, 'ui', 'screens', 'luma-discover-daniel.png'), cv2.IMREAD_UNCHANGED)[..., 3]
        self.shape = shape_without_notch(a)

    def texture(self, f):
        a, b = self.SWIPE
        if f < a:
            return self.edward
        if f < self.LIKE:
            base = self.daniel_empty.copy()
            k = (f - a) / (b - a)
            if k < 1:
                x0, y0, x1, y1 = ui_luma.CARD
                e = k * k
                card = transform_rgba(self.card[y0:y1, x0:x1], 1.0, 22 * e)  # rotates as it leaves
                cx = (x0 + x1) / 2 - 1100 * e
                cy = (y0 + y1) / 2 + 120 * k
                over(base, card, int(cx - card.shape[1] / 2), int(cy - card.shape[0] / 2))
            return base
        k = (f - self.LIKE) / 9
        if k >= 1:
            return self.daniel_liked
        base = self.daniel_empty.copy()
        # fill grows from the outline with a small overshoot, plus a soft ring
        s = .35 + .65 * spring(k)
        cx, cy = ui_luma.HEART_C
        # scale the full-screen heart layer about the button centre
        big = cv2.resize(self.heart, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        ox, oy = int(cx - cx * s), int(cy - cy * s)
        over(base, big, ox, oy)
        ring = np.zeros_like(base)
        cv2.circle(ring, (cx, cy), int(60 + 50 * k), (242, 137, 63), 5, cv2.LINE_AA)
        base = cv2.addWeighted(base, 1, ring, .55 * (1 - k), 0)
        return base


def shape_without_notch(alpha):
    """Screen outline mask with the notch/island hole filled (the UI draws its own)."""
    m = (alpha > 127).astype(np.uint8) * 255
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 128)
    return np.where(ff == 128, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ S2: Luma chat
class S2:
    MSG = 'See you at 8 😊'
    KEYS = [128, 131, 135, 138, 142, 145, 149, 152, 156, 159, 163]  # frames of each keystroke after "See "
    SEND = 167
    DISMISS = 8

    def __init__(self):
        self.first, self.q, _ = load_track('s02-luma-chat')
        self.final = ui_luma.load('luma-chat-daniel')
        self.sent = img('luma-chat-sent')
        a = cv2.imread(os.path.join(JOB, 'ui', 'screens', 'luma-chat-daniel.png'), cv2.IMREAD_UNCHANGED)[..., 3]
        self.shape = shape_without_notch(a)
        self.cache = {}

    def texture(self, f):
        if f >= self.SEND + self.DISMISS:
            return self.sent
        if f >= self.SEND:
            k = ease_io((f - self.SEND) / self.DISMISS)
            off = int(round(k * (ui_luma.CHAT_BAR[0] - (ui_luma.KB_TOP - (ui_luma.CHAT_BAR[1] - ui_luma.CHAT_BAR[0]) - 6))))
            t = ui_luma.chat_typing(self.final, '', cursor=False, kb_offset=off)
            # the sent bubble rises into place as the keyboard drops
            r0, r1 = ui_luma.BUBBLE_ROWS
            bub = self.sent[r0:r1].astype(np.float32)
            dy = int(30 * (1 - k))
            dst = t[r0 + dy:r1 + dy].astype(np.float32)
            a = min(1.0, (f - self.SEND + 1) / 4)
            t[r0 + dy:r1 + dy] = (dst * (1 - a) + bub[:r1 - r0 - max(0, r1 + dy - t.shape[0])] * a).astype(np.uint8)
            return t
        n = 4 + sum(1 for k in self.KEYS if f >= k)
        n = min(n, len(self.MSG))
        blink = (f // 6) % 2 == 0 or f - max([k for k in self.KEYS if k <= f] or [0]) < 6
        key = (n, blink)
        if key not in self.cache:
            self.cache[key] = ui_luma.chat_typing(self.final, self.MSG[:n], cursor=blink)
        return self.cache[key]


# ------------------------------------------------------------------ Wagon interactions
META = None


def meta():
    global META
    if META is None:
        META = json.load(open(os.path.join(STATES, 'wagon-meta.json')))
    return META


def heart_shape(size):
    t = np.linspace(0, 2 * np.pi, 60)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], 1) / 17 * size / 2
    return pts


def press_button(tex, center, ring_r, color, t):
    """Button press: squeeze, pop with overshoot, glow ring and a few hearts floating up.

    t is frames since the press (negative: nothing yet)."""
    if t < 0:
        return tex
    out = tex.copy()
    cx, cy = center
    r = ring_r + 8
    patch = tex[cy - r:cy + r, cx - r:cx + r].copy()
    if t < 3:
        s = 1 - .12 * (t / 3)
    else:
        s = .88 + .12 * spring((t - 3) / 12) + .06 * math.exp(-(t - 6) ** 2 / 10)
    if abs(s - 1) > .005:
        bg = cv2.GaussianBlur(tex[cy - r - 10:cy + r + 10, cx - r - 10:cx + r + 10], (0, 0), 6)[10:-10, 10:-10]
        mask = np.zeros((2 * r, 2 * r), np.float32)
        cv2.circle(mask, (r, r), r - 2, 1, -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]
        scaled = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        sm = cv2.resize(mask, None, fx=s, fy=s)[..., None] if s != 1 else mask
        canvas = bg.astype(np.float32)
        ph = scaled.shape[0]
        o = (2 * r - ph) // 2
        if o >= 0:
            canvas[o:o + ph, o:o + ph] = canvas[o:o + ph, o:o + ph] * (1 - sm) + scaled * sm
            out[cy - r:cy + r, cx - r:cx + r] = (out[cy - r:cy + r, cx - r:cx + r] * (1 - mask) + canvas * mask).astype(np.uint8)
        else:
            c = -o
            sc = scaled[c:c + 2 * r, c:c + 2 * r]
            smc = sm[c:c + 2 * r, c:c + 2 * r]
            out[cy - r:cy + r, cx - r:cx + r] = (canvas * (1 - smc) + sc * smc).astype(np.uint8)
    glow = np.zeros_like(out)
    k = min(t / 16, 1)
    if k < 1:
        cv2.circle(glow, (cx, cy), int(ring_r * (1 + .9 * ease_out(k))), color, 6, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 3)
        out = cv2.add(out, (glow * (1 - k)).astype(np.uint8))
    # floating hearts
    rng = np.random.default_rng(int(cx * 7 + cy))
    for i in range(5):
        k2 = (t - i * 1.5) / 22
        if not 0 < k2 < 1:
            continue
        dx = rng.uniform(-60, 60)
        pts = heart_shape(26 * (1 - .4 * k2)) + [cx + dx * k2, cy - ring_r - 10 - 160 * ease_out(k2)]
        lay = out.copy()
        cv2.fillPoly(lay, [np.round(pts * 4).astype(np.int32)], color, cv2.LINE_AA, shift=2)
        out = cv2.addWeighted(out, 1 - (1 - k2) * .9, lay, (1 - k2) * .9, 0)
    return out


def thumb_at(tips, i, region):
    """Largest thumb tip inside region (x0,y0,x1,y1 in 1000x2000 screen units), or None."""
    if tips is None or i >= len(tips):
        return None
    best = None
    for x, y, area in tips[i]:
        if region[0] <= x <= region[2] and region[1] <= y <= region[3] and y > 150:
            if best is None or area > best[2]:
                best = (x, y, area)
    return best


class S10Left:
    """Dating Discover: the card follows her left thumb's drag, then her right thumb likes."""
    PRESS = 1978

    def __init__(self):
        self.first, self.q, self.tips = load_track('s10-left')
        self.tex = img('wagon-dating')
        m = meta()['wagon-dating']
        self.heart, self.ring, self.color, self.card = m['heart'], m['ring'], tuple(m['color']), m['card']
        h, w = self.tex.shape[:2]
        self.sx, self.sy = w / 1000, h / 2000
        # the left thumb's drag offsets, smoothed
        offs = []
        start = None
        for i in range(len(self.q)):
            t = thumb_at(self.tips, i, (0, 300, 620, 1300))
            if t and start is None:
                start = t
            offs.append(((t[0] - start[0]) * .35, (t[1] - start[1]) * .25) if t and start else None)
        cur = np.zeros(2)
        self.drag = []
        for o in offs:
            target = np.array(o) if o is not None else np.zeros(2)
            cur = cur + (target - cur) * (.6 if o is not None else .3)
            self.drag.append(cur.copy())

    def texture(self, f):
        i = f - self.first
        tex = self.tex
        dx, dy = self.drag[min(i, len(self.drag) - 1)]
        if abs(dx) + abs(dy) > .5:
            tex = tex.copy()
            x0, y0, x1, y1 = self.card
            bg = tuple(int(v) for v in self.tex[y0 + 40, 3])
            card = np.dstack([self.tex[y0:y1, x0:x1], (ui_luma.rounded_mask(x1 - x0, y1 - y0, 28) * 255).astype(np.uint8)])
            tex[y0:y1, x0:x1] = bg
            card = transform_rgba(card, 1.0, -dx * .05)
            over(tex, card, int((x0 + x1) / 2 + dx * self.sx - card.shape[1] / 2), int((y0 + y1) / 2 + dy * self.sy - card.shape[0] / 2))
        return press_button(tex, self.heart, self.ring, self.color, f - self.PRESS)


class S10Right:
    PRESS = 1972

    def __init__(self):
        self.first, self.q, self.tips = load_track('s10-right')
        self.tex = img('wagon-friendship')
        m = meta()['wagon-friendship']
        self.heart, self.ring, self.color = m['heart'], m['ring'], tuple(m['color'])

    def texture(self, f):
        return press_button(self.tex, self.heart, self.ring, self.color, f - self.PRESS)


class S11:
    PRESS = 2037

    def __init__(self):
        self.first, self.q, _ = load_track('s11-her')
        self.tex = img('wagon-match')
        m = meta()['wagon-match']
        self.heart, self.ring, self.color = m['heart'], m['ring'], tuple(m['color'])

    def texture(self, f):
        tex = self.tex
        # the match screen settles in at the start of the shot with a gentle heartbeat glow
        beat = .5 + .5 * math.sin((f - self.first) / 24 * 2 * math.pi * 1.1)
        glow = np.zeros_like(tex)
        cv2.circle(glow, tuple(self.heart), self.ring + 6, self.color, 4, cv2.LINE_AA)
        tex = cv2.add(tex, (cv2.GaussianBlur(glow, (0, 0), 6) * (.35 + .35 * beat)).astype(np.uint8))
        return press_button(tex, self.heart, self.ring, self.color, f - self.PRESS)


class S12:
    SEND = 2108

    def __init__(self):
        self.first, self.q, _ = load_track('s12-his')
        self.draft = img('wagon-chat-draft')
        self.sent = img('wagon-chat-sent')
        m = meta()['wagon-chat']
        self.bubble, self.send = m['bubble'], m['send']

    def texture(self, f):
        if f < self.SEND:
            return self.draft
        t = f - self.SEND
        if t >= 8:
            return self.sent
        out = self.sent.copy()
        x0, y0, x1, y1 = self.bubble
        # bubble pops up from the bottom right; until then the area shows the empty chat
        k = ease_out(t / 7)
        page = self.draft[y0 - 4:y1 + 4, x0 - 4:x1 + 4]
        out[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = page
        bub = self.sent[y0:y1, x0:x1]
        s = .6 + .4 * spring(t / 8)
        b2 = cv2.resize(bub, None, fx=s, fy=s)
        bh, bw = b2.shape[:2]
        px, py = int(x1 - bw), int(y1 - bh + 40 * (1 - k))
        lay = out.copy()
        lay[py:py + bh, px:px + bw] = b2
        out = cv2.addWeighted(out, 1 - k, lay, k, 0)
        return press_button(out, self.send, 36, (52, 116, 252), t)


# ------------------------------------------------------------------ S9/S10 graphics
class SplitGraphics:
    """Wagon logo on the split line, animated light along the line, floating phones (S9)."""
    S9, S10_END = (1860, 1935), 2023
    LINE_X = {'s9': 422 * SX, 's10': 417.5 * SX}

    def __init__(self):
        self.icon = cv2.imread(os.path.join(MG, 'icon.png'), cv2.IMREAD_UNCHANGED)
        self.word = cv2.imread(os.path.join(MG, 'wordmark.png'), cv2.IMREAD_UNCHANGED)
        self.phones = [cv2.imread(os.path.join(MG, p), cv2.IMREAD_UNCHANGED) for p in ('phone-dating.png', 'phone-friendship.png')]
        isz = 168
        self.icon_s = cv2.resize(self.icon, (isz, int(isz * self.icon.shape[0] / self.icon.shape[1])), interpolation=cv2.INTER_AREA)
        ww = 236
        self.word_s = cv2.resize(self.word, (ww, int(ww * self.word.shape[0] / self.word.shape[1])), interpolation=cv2.INTER_AREA)
        ph = 560
        self.phones_s = [cv2.resize(p, (int(ph * p.shape[1] / p.shape[0]), ph), interpolation=cv2.INTER_AREA) for p in self.phones]
        rng = np.random.default_rng(3)
        self.sparks = [(rng.uniform(0, 1), rng.uniform(.3, 1.0), rng.choice([-1, 1]), rng.uniform(2, 4)) for _ in range(14)]

    def light(self, frame, x, t):
        """Animated split light: energy pulses travelling along the line, breathing glow, sparks."""
        x0, x1 = int(x - 80), int(x + 80)
        xs = np.arange(x0, x1) - x
        ys = np.arange(H)
        core = np.exp(-(xs / 3.0) ** 2)
        halo = np.exp(-(xs / 26.0) ** 2)
        pulses = (np.exp(-(((ys - ((t * 620) % (H + 600) - 300)) / 140) ** 2))
                  + .8 * np.exp(-(((ys - (H + 300 - (t * 470 + 400) % (H + 600))) / 110) ** 2)))
        breath = .55 + .25 * math.sin(t * 2 * math.pi * .9) + .06 * math.sin(t * 37)
        inten = (breath + .9 * pulses)[:, None]
        col = np.array([235, 70, 255], np.float32)  # BGR magenta
        glow = (core * .9 + halo * .55)[None, :] * inten
        layer = glow[..., None] * col[None, None, :] * .55
        reg = frame[:, x0:x1].astype(np.float32)
        frame[:, x0:x1] = np.clip(reg + layer, 0, 255).astype(np.uint8)
        for (y0, speed, d, size) in self.sparks:
            y = (y0 * H + d * speed * t * 300) % H
            r = int(size)
            sp = np.zeros((4 * r + 1, 4 * r + 1, 3), np.float32)
            cv2.circle(sp, (2 * r, 2 * r), r, (255, 200, 255), -1, cv2.LINE_AA)
            sp = cv2.GaussianBlur(sp, (0, 0), r * .8)
            add(frame, sp, int(x - 2 * r + 6 * math.sin(t * 3 + y0 * 9)), int(y - 2 * r), .9)

    def logo(self, frame, x, k_in, t):
        cy = 476
        s = .55 + .45 * spring(k_in)
        a = min(1.0, k_in * 2.5)
        icon = cv2.resize(self.icon_s, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        ih, iw = icon.shape[:2]
        # bloom behind the icon
        bloom = np.zeros((ih + 160, iw + 160, 3), np.float32)
        bloom[80:80 + ih, 80:80 + iw] = icon[..., :3].astype(np.float32) * (icon[..., 3:] / 255)
        bloom = cv2.GaussianBlur(bloom, (0, 0), 28)
        add(frame, bloom, int(x - iw / 2 - 80), int(cy - ih / 2 - 80), (.55 + .15 * math.sin(t * 4)) * a)
        over(frame, icon, int(x - iw / 2), int(cy - ih / 2), a)
        kw = min(max((k_in - .35) / .65, 0), 1)
        if kw > 0:
            wd = self.word_s
            # soft shadow so the wordmark reads over both halves
            sh = np.zeros(wd.shape[:2] + (4,), np.uint8)
            sh[..., 3] = cv2.GaussianBlur(wd[..., 3], (0, 0), 8)
            over(frame, sh, int(x - wd.shape[1] / 2), int(cy + 100 + 16 * (1 - ease_out(kw))), .6 * kw)
            over(frame, wd, int(x - wd.shape[1] / 2), int(cy + 96 + 16 * (1 - ease_out(kw))), ease_out(kw))

    def phones_float(self, frame, t, k_in):
        for side, ph in enumerate(self.phones_s):
            e = ease_out(k_in)
            ang = (5 if side == 0 else -5) * (1 - .4 * e)
            p = transform_rgba(ph, 1.0, ang)
            bob = 8 * math.sin(t * 2.2 + side * 1.7)
            if side == 0:
                x = int(-p.shape[1] + (24 + p.shape[1]) * e)
            else:
                x = int(W - (24 + p.shape[1]) * e)
            y = int(H / 2 - p.shape[0] / 2 + 30 + bob)
            shadow = np.zeros_like(p)
            shadow[..., 3] = cv2.GaussianBlur(p[..., 3], (0, 0), 18)
            over(frame, shadow, x + 14, y + 24, .55 * e)
            over(frame, p, x, y, min(1.0, e * 1.5))

    def apply(self, frame, f):
        if self.S9[0] <= f <= self.S9[1]:
            t = (f - self.S9[0]) / FPS
            x = self.LINE_X['s9']
            self.light(frame, x, t)
            self.phones_float(frame, t, min(1.0, t / .6))
            self.logo(frame, x, min(1.0, max(t - .25, 0) / .7), t)
        elif self.S9[1] < f <= self.S10_END:
            t = (f - self.S9[0]) / FPS
            x = self.LINE_X['s10']
            self.light(frame, x, t)
            self.logo(frame, x, 1.0, t)


# ------------------------------------------------------------------ end card
class EndCard:
    ORDER = [  # element, start, duration (seconds), style
        ('icon', .25, .8, 'pop'),
        ('wordmark', .75, .6, 'rise'),
        ('script-real', 1.2, .9, 'wipe'),
        ('script-stronger', 1.45, .9, 'wipe'),
        ('script-brighter', 1.7, .9, 'wipe'),
        ('tagline', 1.3, .9, 'wipe'),
        ('divider', 2.15, .5, 'grow'),
        ('sober', 2.4, .7, 'rise'),
    ]

    def __init__(self):
        sz = (W, H)
        self.full = cv2.resize(cv2.imread(os.path.join(MG, 'endcard-full.png')), sz, interpolation=cv2.INTER_CUBIC)
        self.plate = cv2.resize(cv2.imread(os.path.join(MG, 'endcard-plate.png')), sz, interpolation=cv2.INTER_CUBIC)
        self.masks = {n: cv2.resize(cv2.imread(os.path.join(MG, f'endcard-{n}.png'), 0), sz).astype(np.float32) / 255
                      for n, *_ in self.ORDER}
        self.arcs = cv2.resize(cv2.imread(os.path.join(MG, 'endcard-arcs.png'), 0), sz).astype(np.float32) / 255
        self.boxes = {}
        for n, m in self.masks.items():
            ys, xs = np.nonzero(m > .05)
            self.boxes[n] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    def frame(self, t):
        out = self.plate.astype(np.float32)
        full = self.full.astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        for name, start, dur, style in self.ORDER:
            k = (t - start) / dur
            if k <= 0:
                continue
            k = min(k, 1.0)
            m = self.masks[name]
            x0, y0, x1, y1 = self.boxes[name]
            if style == 'wipe':
                edge = x0 + (x1 - x0 + 60) * ease_io(k) - 30
                a = m * np.clip((edge - xx) / 30, 0, 1)
                src = full
            elif style == 'grow':
                cx = (x0 + x1) / 2
                half = (x1 - x0) / 2 * ease_out(k) + 1
                a = m * np.clip((half - np.abs(xx - cx)) / 12, 0, 1)
                src = full
            elif style == 'rise':
                dy = int(18 * (1 - ease_out(k)))
                a = np.roll(m, dy, axis=0) * ease_out(k)
                src = np.roll(full, dy, axis=0)
            else:  # pop: scale the icon region about its centre with overshoot
                s = .7 + .3 * spring(k)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
                src = cv2.warpAffine(full, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                a = cv2.warpAffine(m, M, (W, H)) * min(1.0, k * 3)
            out = out * (1 - a[..., None]) + src * a[..., None]
        # shimmer travelling along the glowing arcs, and a pulse on the bright dots
        band = np.exp(-(((xx - ((t * 520) % (W + 400) - 200)) / 160) ** 2))
        out += (self.arcs * band)[..., None] * np.array([255, 120, 255], np.float32) * .6
        # slow push-in
        s = 1 + .035 * ease_io(t / 5)
        M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
        out = cv2.warpAffine(np.clip(out, 0, 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_LINEAR)
        return out


# ------------------------------------------------------------------ main
def build():
    return {'s1': S1(), 's2': S2(), 's10l': S10Left(), 's10r': S10Right(), 's11': S11(), 's12': S12(),
            'split': SplitGraphics(), 'end': EndCard()}


def process(frame_src, f, R):
    frame = cv2.resize(frame_src, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if 17 <= f <= 125:
        s = R['s1']
        composite_screen(frame, s.q[f - s.first], s.texture(f), 'ui', s.shape)
    elif 126 <= f <= 191:
        s = R['s2']
        composite_screen(frame, s.q[f - s.first], s.texture(f), 'ui', s.shape)
    elif 1936 <= f <= 2023:
        for key in ('s10l', 's10r'):
            s = R[key]
            composite_screen(frame, s.q[f - s.first], s.texture(f), 'white')
    elif 2024 <= f <= 2093:
        s = R['s11']
        composite_screen(frame, s.q[f - s.first], s.texture(f), 'white')
    elif 2094 <= f <= 2156:
        s = R['s12']
        composite_screen(frame, s.q[f - s.first], s.texture(f), 'white')
    R['split'].apply(frame, f)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(JOB, 'output', 'wagon-commercial.mp4'))
    ap.add_argument('--stills', nargs='*', type=int, help='source frames to export as JPEGs (or end-card frames as -N)')
    ap.add_argument('--from', dest='f0', type=int, default=START)
    ap.add_argument('--to', dest='f1', type=int, default=LAST_SRC + END_FRAMES)
    args = ap.parse_args()
    R = build()
    cap = cv2.VideoCapture(SRC)
    if args.stills is not None:
        d = os.path.join(JOB, 'output', 'stills')
        os.makedirs(d, exist_ok=True)
        for f in args.stills:
            if f < 0:
                out = R['end'].frame(-f / FPS)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                out = process(fr, f, R)
            cv2.imwrite(os.path.join(d, f'f{f:05d}.jpg'), out, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print('stills in', d)
        return
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    picture = args.out.replace('.mp4', '-picture.mp4')
    enc = subprocess.Popen([ff, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
                            '-pix_fmt', 'yuv420p', picture], stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.f0)
    last = None
    total = args.f1 - args.f0 + 1
    for f in range(args.f0, args.f1 + 1):
        if f <= LAST_SRC:
            ok, fr = cap.read()
            out = process(fr, f, R)
            last = out
        else:
            t = (f - LAST_SRC - 1) / FPS
            out = R['end'].frame(t)
            k = min(1.0, (f - LAST_SRC) / XFADE)
            if k < 1 and last is not None:
                out = cv2.addWeighted(last, 1 - k, out, k, 0)
            if f > LAST_SRC + END_FRAMES - 12:  # fade out the very end
                out = (out * ((LAST_SRC + END_FRAMES - f) / 12)).astype(np.uint8)
        enc.stdin.write(out.tobytes())
        if (f - args.f0) % 48 == 0:
            print(f'\r{f - args.f0}/{total}', end='', flush=True)
    enc.stdin.close()
    enc.wait()
    # audio: the edit's own track from the new start, with silence under the end card
    dur = total / FPS
    subprocess.run([ff, '-loglevel', 'error', '-y', '-i', picture, '-ss', f'{args.f0 / FPS:.4f}', '-i', SRC,
                    '-filter_complex', f'[1:a]apad,atrim=0:{dur:.3f}[a]', '-map', '0:v', '-map', '[a]',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', args.out], check=True)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
