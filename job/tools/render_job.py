"""Renders the Wagon commercial: the supplied edit with every phone screen replaced.

The screens tell the story shot to shot:
- S1 (couch): Edward's card swipes off on her thumb (the second swipe; the first is cut),
  Daniel appears, her tap fills the heart blue, then Luma opens the chat with Daniel and the
  keyboard comes up as she starts typing.
- S2 (close-up): "See you at 8 😊" typed letter by letter, then sent.
- S9: the Wagon logo arrives on the split line with a ring burst; Discover phones float in.
- S10: Dating (Jacob) and Friendship (Marcus) Discover; each heart is pressed by the thumb.
- S11: her Discover screen; she taps Jacob's photo, a heart pops, then It's a Match.
- S12: his chat with Marcus: his message is in the field, he sends it, Marcus types, Marcus replies.
- End: the Wagon card dissolves in over the last shot and builds element by element.

Usage: python3 job/tools/render_job.py [--stills F ...] [--from F --to F] [--out path]
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
START = 6             # first source frame kept (S1's first swipe, frames 0-5, removed)
LAST_SRC = 2640       # last source frame
END_START = 2618      # the end card starts dissolving in over the last shot
END_LAST = 2726       # last frame of the film (the card holds after the source runs out)
XFADE = 14            # dissolve length into the end card
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
    """Edward -> swipe -> Daniel -> like -> chat opens -> keyboard up -> starts typing."""
    SWIPE = (8, 16)       # her thumb's swipe in the source
    LIKE = 62             # the thumb presses the heart
    TO_CHAT = (77, 83)    # Luma opens the chat with Daniel
    KB_UP = (99, 105)     # she taps the field: keyboard rises
    KEYS = {108: 'S', 113: 'Se', 118: 'See'}

    def __init__(self):
        self.first, self.q, _ = load_track('s01-luma')
        self.daniel_empty = img('luma-daniel-empty')
        self.daniel_liked = img('luma-daniel-liked')
        self.edward = img('luma-edward')
        self.card = img('luma-edward-card', cv2.IMREAD_UNCHANGED)
        self.heart = img('luma-heart-layer', cv2.IMREAD_UNCHANGED)
        a = cv2.imread(os.path.join(JOB, 'ui', 'screens', 'luma-discover-daniel.png'), cv2.IMREAD_UNCHANGED)[..., 3]
        self.shape = shape_without_notch(a)
        self.chat = ui_luma.load('luma-chat-daniel')
        self.cache = {}

    def fit(self, t):
        h, w = self.daniel_empty.shape[:2]
        return cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA) if t.shape[:2] != (h, w) else t

    def chat_state(self, down, text='', cursor=False):
        key = (round(down, 3), text, cursor)
        if key not in self.cache:
            self.cache[key] = self.fit(ui_luma.chat_typing(self.chat, text, cursor=cursor, down=down))
        return self.cache[key]

    def discover(self, f):
        a, b = self.SWIPE
        if f < a:
            return self.edward
        if f <= b:
            # Daniel's card rises from behind as Edward's is thrown left
            k = (f - a) / (b - a)
            e = k ** 1.6
            base = self.daniel_empty.copy()
            x0, y0, x1, y1 = ui_luma.CARD
            s = .94 + .06 * ease_out(k)
            dan = self.daniel_empty[y0:y1, x0:x1]
            cw, ch = int((x1 - x0) * s), int((y1 - y0) * s)
            small = cv2.resize(dan, (cw, ch), interpolation=cv2.INTER_AREA)
            base[y0:y1, x0:x1] = self.daniel_empty[y0 - 6, x0 + 40]
            ox, oy = x0 + ((x1 - x0) - cw) // 2, y0 + ((y1 - y0) - ch) // 2
            m = ui_luma.rounded_mask(cw, ch, ui_luma.CARD_R)[..., None]
            base[oy:oy + ch, ox:ox + cw] = (base[oy:oy + ch, ox:ox + cw] * (1 - m) + small * m).astype(np.uint8)
            card = transform_rgba(self.card[y0:y1, x0:x1], 1.0, 14 * e)
            cx = (x0 + x1) / 2 - 900 * e
            cy = (y0 + y1) / 2 - 60 * e
            over(base, card, int(cx - card.shape[1] / 2), int(cy - card.shape[0] / 2))
            return base
        if f < self.LIKE:
            return self.daniel_empty
        k = (f - self.LIKE) / 9
        if k >= 1:
            return self.daniel_liked
        base = self.daniel_empty.copy()
        s = .35 + .65 * spring(k)
        cx, cy = ui_luma.HEART_C
        big = cv2.resize(self.heart, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        over(base, big, int(cx - cx * s), int(cy - cy * s))
        ring = np.zeros_like(base)
        cv2.circle(ring, (cx, cy), int(60 + 50 * k), (242, 137, 63), 5, cv2.LINE_AA)
        return cv2.addWeighted(base, 1, ring, .55 * (1 - k), 0)

    def texture(self, f):
        a, b = self.TO_CHAT
        if f < a:
            return self.discover(f)
        if f <= b:
            return ios_push(self.daniel_liked, self.chat_state(1.0), (f - a) / (b - a))
        u0, u1 = self.KB_UP
        if f < u0:
            return self.chat_state(1.0)
        if f <= u1:
            return self.chat_state(1 - ease_out((f - u0) / (u1 - u0)), '', True)
        text = ''
        for kf, t in sorted(self.KEYS.items()):
            if f >= kf:
                text = t
        last = max([kf for kf in self.KEYS if kf <= f] or [u1])
        blink = (f - last) < 6 or (f // 6) % 2 == 0
        return self.chat_state(0.0, text, blink)


def ios_push(a, b, k):
    """b slides in from the right over a (a shifts left 30% and dims)."""
    h, w = a.shape[:2]
    e = ease_io(k)
    out = np.zeros_like(a)
    sh = int(round(w * .3 * e))
    out[:, :w - sh] = a[:, sh:]
    out = (out * (1 - .3 * e)).astype(np.uint8)
    x = int(round(w * (1 - e)))
    out[:, x:] = b[:, :w - x]
    if 0 < x < w:  # soft shadow on the incoming screen's edge
        sw = min(14, x)
        grad = np.linspace(.75, 1, sw)[None, :, None]
        out[:, x - sw:x] = (out[:, x - sw:x] * grad).astype(np.uint8)
    return out


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
    """Dating Discover (Jacob): her thumb presses the heart."""
    PRESS = 1978

    def __init__(self):
        self.first, self.q, self.tips = load_track('s10-left')
        self.tex = img('wagon-dating')
        m = meta()['wagon-dating']
        self.heart, self.ring, self.color = m['heart'], m['ring'], tuple(m['color'])

    def texture(self, f):
        return press_button(self.tex, self.heart, self.ring, self.color, f - self.PRESS)


class S10Right:
    """Friendship Discover (Marcus): his thumb presses the heart."""
    PRESS = 1972

    def __init__(self):
        self.first, self.q, self.tips = load_track('s10-right')
        self.tex = img('wagon-friendship')
        m = meta()['wagon-friendship']
        self.heart, self.ring, self.color = m['heart'], m['ring'], tuple(m['color'])

    def texture(self, f):
        return press_button(self.tex, self.heart, self.ring, self.color, f - self.PRESS)


def big_heart(tex, center, size, alpha, color=(255, 85, 200)):
    """A filled heart with a soft glow, drawn onto tex (the double-tap like)."""
    out = tex.astype(np.float32)
    lay = np.zeros(tex.shape[:2], np.float32)
    pts = heart_shape(size) + center
    cv2.fillPoly(lay, [np.round(pts * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(lay, (0, 0), size * .18)
    col = np.array(color, np.float32)
    out = out + glow[..., None] * col * .55 * alpha
    a = lay[..., None] * alpha
    out = out * (1 - a) + (col * .92 + 20) * a
    # a soft highlight on the upper left lobe
    hl = np.zeros_like(lay)
    cv2.ellipse(hl, (int(center[0] - size * .22), int(center[1] - size * .18)), (int(size * .12), int(size * .07)),
                -30, 0, 360, 1.0, -1, cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), size * .04) * lay
    out = out + hl[..., None] * 120 * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


class S11:
    """Her Discover (Jacob): she taps his photo, a heart pops, then It's a Match."""
    TAP = 2038
    TO_MATCH = (2046, 2053)

    def __init__(self):
        self.first, self.q, self.tips = load_track('s11-her')
        self.disc = img('wagon-dating')
        self.match = img('wagon-match')
        md, mm = meta()['wagon-dating'], meta()['wagon-match']
        self.btn, self.ring = md['heart'], md['ring']
        self.mheart, self.mring, self.color = mm['heart'], mm['ring'], tuple(mm['color'])
        h, w = self.disc.shape[:2]
        # where her finger lands, from the thumb tracking (screen units -> texture pixels)
        tip = thumb_at(self.tips, self.TAP - 1 - self.first, (0, 300, 1000, 1700)) or (500, 1000, 0)
        self.tap = (tip[0] / 1000 * w, tip[1] / 2000 * h - 40)

    def texture(self, f):
        a, b = self.TO_MATCH
        if f < a:
            t = f - self.TAP
            tex = press_button(self.disc, self.btn, self.ring, (255, 85, 200), t - 2)
            if t >= 0:
                k = t / 8
                size = 150 * (.3 + .7 * spring(min(k, 1)))
                alpha = 1.0 if f < a - 2 else .7
                tex = big_heart(tex, self.tap, size, alpha)
            return tex
        beat = .5 + .5 * math.sin((f - b) / FPS * 2 * math.pi * 1.2)
        m = self.match.copy()
        glow = np.zeros_like(m)
        cv2.circle(glow, tuple(self.mheart), self.mring + 8, self.color, 5, cv2.LINE_AA)
        m = cv2.add(m, (cv2.GaussianBlur(glow, (0, 0), 7) * (.3 + .45 * beat)).astype(np.uint8))
        if f <= b:
            k = ease_io((f - a) / (b - a))
            # the match screen blooms in over Discover
            s = 1.06 - .06 * k
            h, w = m.shape[:2]
            M = np.float32([[s, 0, w / 2 * (1 - s)], [0, s, h / 2 * (1 - s)]])
            mz = cv2.warpAffine(m, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            src = self.texture(a - 1)
            return cv2.addWeighted(src, 1 - k, mz, k, 0)
        return m


class S12:
    """His chat with Marcus, built up live from the supplied conversation."""
    SEND = 2106
    TYPING = (2118, 2135)
    REPLY = 2136

    def __init__(self):
        self.first, self.q, _ = load_track('s12-his')
        self.s = {k: img(f'wagon-chat-{k}') for k in ('empty', 'draft', 'sent', 'typing', 'full')}
        m = meta()['wagon-chat']
        self.orange, self.reply, self.dots, self.send = m['orange'], m['reply'], m['dots'], m['send']

    def pop_in(self, base, src_state, box, t, n=7, origin='br'):
        """Pop a bubble (box y0,y1,x0,x1 of src_state) into base: scale from its corner + fade."""
        y0, y1, x0, x1 = box
        if t >= n:
            out = base.copy()
            out[y0:y1, x0:x1] = src_state[y0:y1, x0:x1]
            return out
        k = max(t, 0) / n
        s = .55 + .45 * spring(k)
        a = min(1.0, k * 2.2)
        patch = src_state[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        sp = cv2.resize(patch, (max(1, int(pw * s)), max(1, int(ph * s))), interpolation=cv2.INTER_LINEAR)
        out = base.copy()
        sh, sw = sp.shape[:2]
        if origin == 'br':
            px, py = x1 - sw, y1 - sh + int(30 * (1 - k))
        else:
            px, py = x0, y1 - sh + int(20 * (1 - k))
        reg = out[py:py + sh, px:px + sw].astype(np.float32)
        out[py:py + sh, px:px + sw] = (reg * (1 - a) + sp * a).astype(np.uint8)
        return out

    def texture(self, f):
        if f < self.SEND:
            return self.s['draft']
        t = f - self.SEND
        if f < self.TYPING[0]:
            out = self.pop_in(self.s['empty'], self.s['sent'], self.orange, t)
            return press_button(out, self.send, 34, (52, 116, 252), t)
        if f < self.REPLY:
            out = self.s['typing'].copy()
            x0, y0, x1, y1 = self.dots
            k = min(1.0, (f - self.TYPING[0]) / 4)
            cy = (y0 + y1) // 2
            for i in range(3):
                ph = (f - self.TYPING[0]) / 4.0 - i * .6
                lift = 6 * max(0.0, math.sin(ph * math.pi / 1.5)) if ph > 0 else 0
                cx = int(x0 + (x1 - x0) * (.3 + .2 * i))
                cv2.circle(out, (cx, int(cy - lift)), 7, (150, 150, 150), -1, cv2.LINE_AA)
            return cv2.addWeighted(self.s['sent'], 1 - k, out, k, 0)
        return self.pop_in(self.s['sent'], self.s['full'], self.reply, f - self.REPLY, 8, origin='bl')


# ------------------------------------------------------------------ S9/S10 graphics
class SplitGraphics:
    """Animated split light (S9 + S10); in S9 the logo arrives with a ring burst and the
    Discover phones float in on each side."""
    S9, S10_END = (1860, 1935), 2023
    LINE_X = {'s9': 422 * SX, 's10': 417.5 * SX}
    RING = 1870           # ring burst on the split line
    LOGO = 1873           # icon pops in
    PHONES = 1880         # phones float in

    def __init__(self):
        self.icon = cv2.imread(os.path.join(MG, 'icon.png'), cv2.IMREAD_UNCHANGED)
        self.word = cv2.imread(os.path.join(MG, 'wordmark.png'), cv2.IMREAD_UNCHANGED)
        self.phones = [cv2.imread(os.path.join(MG, p), cv2.IMREAD_UNCHANGED) for p in ('phone-dating.png', 'phone-friendship.png')]
        isz = 180
        self.icon_s = cv2.resize(self.icon, (isz, int(isz * self.icon.shape[0] / self.icon.shape[1])), interpolation=cv2.INTER_AREA)
        ww = 210
        self.word_s = cv2.resize(self.word, (ww, int(ww * self.word.shape[0] / self.word.shape[1])), interpolation=cv2.INTER_AREA)
        ph = 690
        self.phones_s = []
        for p in self.phones:
            p = cv2.resize(p, (int(ph * p.shape[1] / p.shape[0]), ph), interpolation=cv2.INTER_AREA)
            # magenta rim light that picks up the split glow
            pad = 40
            big = cv2.copyMakeBorder(p, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            rim = cv2.GaussianBlur(cv2.dilate(big[..., 3], np.ones((9, 9), np.uint8)), (0, 0), 14).astype(np.float32) / 255
            glow = np.zeros_like(big)
            glow[..., :3] = (np.array([235, 80, 255], np.float32) * rim[..., None]).astype(np.uint8)
            glow[..., 3] = (rim * 170).astype(np.uint8)
            self.phones_s.append((glow, big))
        rng = np.random.default_rng(3)
        self.sparks = [(rng.uniform(0, 1), rng.uniform(.3, 1.0), rng.choice([-1, 1]), rng.uniform(2, 4)) for _ in range(14)]

    def light(self, frame, x, t):
        """Energy pulses travelling along the line, a breathing glow and drifting sparks."""
        x0, x1 = int(x - 80), int(x + 80)
        xs = np.arange(x0, x1) - x
        ys = np.arange(H)
        core = np.exp(-(xs / 3.0) ** 2)
        halo = np.exp(-(xs / 26.0) ** 2)
        pulses = (np.exp(-(((ys - ((t * 620) % (H + 600) - 300)) / 140) ** 2))
                  + .8 * np.exp(-(((ys - (H + 300 - (t * 470 + 400) % (H + 600))) / 110) ** 2)))
        breath = .55 + .25 * math.sin(t * 2 * math.pi * .9) + .06 * math.sin(t * 37)
        inten = (breath + .9 * pulses)[:, None]
        col = np.array([235, 70, 255], np.float32)
        glow = (core * .9 + halo * .55)[None, :] * inten
        layer = glow[..., None] * col[None, None, :] * .55
        frame[:, x0:x1] = np.clip(frame[:, x0:x1].astype(np.float32) + layer, 0, 255).astype(np.uint8)
        for (y0, speed, d, size) in self.sparks:
            y = (y0 * H + d * speed * t * 300) % H
            r = int(size)
            sp = np.zeros((4 * r + 1, 4 * r + 1, 3), np.float32)
            cv2.circle(sp, (2 * r, 2 * r), r, (255, 200, 255), -1, cv2.LINE_AA)
            sp = cv2.GaussianBlur(sp, (0, 0), r * .8)
            add(frame, sp, int(x - 2 * r + 6 * math.sin(t * 3 + y0 * 9)), int(y - 2 * r), .9)

    def ring(self, frame, cx, cy, f):
        k = (f - self.RING) / 12
        if not 0 <= k <= 1:
            return
        r = int(40 + 230 * ease_out(k))
        lay = np.zeros((2 * r + 40, 2 * r + 40, 3), np.float32)
        cv2.circle(lay, (r + 20, r + 20), r, (255, 245, 255), 4, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.2) + cv2.GaussianBlur(lay, (0, 0), 8) * .8
        add(frame, lay, int(cx - r - 20), int(cy - r - 20), 1 - k)

    def logo(self, frame, x, f, t):
        cy = 450
        k_in = (f - self.LOGO) / 10
        if k_in <= 0:
            return
        k_in = min(k_in, 1.0)
        s = .4 + .6 * spring(k_in)
        a = min(1.0, k_in * 3)
        icon = cv2.resize(self.icon_s, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        ih, iw = icon.shape[:2]
        bloom = np.zeros((ih + 200, iw + 200, 3), np.float32)
        bloom[100:100 + ih, 100:100 + iw] = icon[..., :3].astype(np.float32) * (icon[..., 3:] / 255)
        bloom = cv2.GaussianBlur(bloom, (0, 0), 34)
        add(frame, bloom, int(x - iw / 2 - 100), int(cy - ih / 2 - 100), (.7 + .15 * math.sin(t * 4)) * a)
        over(frame, icon, int(x - iw / 2), int(cy - ih / 2), a)
        kw = min(max((f - self.LOGO - 6) / 8, 0), 1)
        if kw > 0:
            wd = self.word_s
            sh = np.zeros(wd.shape[:2] + (4,), np.uint8)
            sh[..., 3] = cv2.GaussianBlur(wd[..., 3], (0, 0), 8)
            yy = int(cy + self.icon_s.shape[0] / 2 + 10 + 14 * (1 - ease_out(kw)))
            over(frame, sh, int(x - wd.shape[1] / 2 + 3), yy + 4, .7 * kw)
            over(frame, wd, int(x - wd.shape[1] / 2), yy, ease_out(kw))

    def phones_float(self, frame, f, t):
        k_in = (f - self.PHONES) / 10
        if k_in <= 0:
            return
        e = ease_out(min(k_in, 1.0))
        for side, (glow, ph) in enumerate(self.phones_s):
            ang = (6 if side == 0 else -6) * (1 - .3 * e)
            g = transform_rgba(glow, 1.0, ang)
            p = transform_rgba(ph, 1.0, ang)
            bob = 9 * math.sin(t * 2.0 + side * 1.7)
            dx = (1 - e) * 120 * (-1 if side == 0 else 1)
            x = int((10 if side == 0 else W - p.shape[1] - 10) + dx)
            y = int(H - p.shape[0] + 30 + bob)
            shadow = np.zeros_like(p)
            shadow[..., 3] = cv2.GaussianBlur(p[..., 3], (0, 0), 22)
            over(frame, shadow, x + 18, y + 26, .55 * e)
            over(frame, g, x, y, e * (.8 + .2 * math.sin(t * 5)))
            over(frame, p, x, y, e)

    def apply(self, frame, f):
        if self.S9[0] <= f <= self.S9[1]:
            t = (f - self.S9[0]) / FPS
            x = self.LINE_X['s9']
            self.light(frame, x, t)
            self.phones_float(frame, f, t)
            self.ring(frame, x, 450, f)
            self.logo(frame, x, f, t)
        elif self.S9[1] < f <= self.S10_END:
            self.light(frame, self.LINE_X['s10'], (f - self.S9[0]) / FPS)


# ------------------------------------------------------------------ end card
class EndCard:
    """The supplied Wagon card, built element by element (t in seconds from END_START)."""
    ORDER = [  # element, start, duration, style
        ('icon', .5, .7, 'pop'),
        ('wordmark', 1.0, .5, 'rise'),
        ('tagline', 1.3, .8, 'wipe'),
        ('script-real', 1.7, .8, 'wipe'),
        ('script-brighter', 1.95, .8, 'wipe'),
        ('script-stronger', 2.2, .8, 'wipe'),
        ('divider', 1.75, .5, 'grow'),
        ('sober', 2.0, .9, 'type'),
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
        # letters of SOBER. SOCIAL. CONNECTED. for the typed reveal
        m = (self.masks['sober'] > .2).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((5, 1), np.uint8)))
        self.letters = sorted([st[i, 0] + st[i, 2] for i in range(1, n) if st[i, 4] > 4])
        # path along the arcs for travelling sparkles: the arc pixels ordered by x
        ys, xs = np.nonzero(self.arcs > .5)
        order = np.argsort(xs)
        self.arc_pts = np.stack([xs[order], ys[order]], 1)
        # the heart on the right arc (for its pulse) and the icon box (for the shine)
        self.heart_box = (int(1472 / 1672 * W) - 20, int(151 / 941 * H) - 20, int(1557 / 1672 * W) + 20, int(230 / 941 * H) + 20)
        self.yy, self.xx = np.mgrid[0:H, 0:W]

    def spark(self, out, x, y, r, mult):
        lay = np.zeros((6 * r + 1, 6 * r + 1, 3), np.float32)
        c = 3 * r
        cv2.circle(lay, (c, c), r, (255, 230, 255), -1, cv2.LINE_AA)
        cv2.line(lay, (c - 3 * r, c), (c + 3 * r, c), (255, 200, 255), 1, cv2.LINE_AA)
        cv2.line(lay, (c, c - 3 * r), (c, c + 3 * r), (255, 200, 255), 1, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), r * .6) * 2.2
        x0, y0 = int(x - c), int(y - c)
        h, w = lay.shape[:2]
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        if X1 > X0 and Y1 > Y0:
            out[Y0:Y1, X0:X1] += lay[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * mult

    def frame(self, t):
        out = self.plate.astype(np.float32)
        full = self.full.astype(np.float32)
        xx = self.xx
        for name, start, dur, style in self.ORDER:
            k = (t - start) / dur
            if k <= 0:
                continue
            k = min(k, 1.0)
            m = self.masks[name]
            x0, y0, x1, y1 = self.boxes[name]
            src = full
            if style == 'wipe':
                edge = x0 + (x1 - x0 + 60) * ease_io(k) - 30
                a = m * np.clip((edge - xx) / 30, 0, 1)
            elif style == 'type':
                n = int(round(len(self.letters) * k))
                edge = self.letters[n - 1] + 2 if n > 0 else x0 - 10
                a = m * (xx <= edge)
            elif style == 'grow':
                cx = (x0 + x1) / 2
                half = (x1 - x0) / 2 * ease_out(k) + 1
                a = m * np.clip((half - np.abs(xx - cx)) / 12, 0, 1)
            elif style == 'rise':
                dy = int(18 * (1 - ease_out(k)))
                a = np.roll(m, dy, axis=0) * ease_out(k)
                src = np.roll(full, dy, axis=0)
            else:  # pop with overshoot
                s = .6 + .4 * spring(k)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
                src = cv2.warpAffine(full, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                a = cv2.warpAffine(m, M, (W, H)) * min(1.0, k * 3)
            out = out * (1 - a[..., None]) + src * a[..., None]
        # ring burst as the icon arrives
        x0, y0, x1, y1 = self.boxes['icon']
        icx, icy = (x0 + x1) / 2, (y0 + y1) / 2
        k = (t - .45) / .6
        if 0 <= k <= 1:
            r = int(90 + 260 * ease_out(k))
            ring = np.zeros((H, W, 3), np.float32)
            cv2.circle(ring, (int(icx), int(icy)), r, (255, 240, 255), 4, cv2.LINE_AA)
            out += (cv2.GaussianBlur(ring, (0, 0), 1.2) + cv2.GaussianBlur(ring, (0, 0), 9) * .9) * (1 - k)
        # shine sweeping across the icon
        k = (t - 2.7) / .6
        if 0 <= k <= 1:
            im = self.masks['icon'] > .9
            band = np.exp(-(((xx - icx) + (self.yy - icy) * .6 - (-260 + 520 * ease_io(k))) / 26) ** 2)
            out += (band * im)[..., None] * np.array([255, 255, 255], np.float32) * .55
        # light travelling along the arcs, with sparkles riding it
        head = (t * 900) % (W + 600) - 300
        band = np.exp(-(((xx - head) / 140) ** 2))
        out += (self.arcs * band)[..., None] * np.array([255, 140, 255], np.float32) * .7
        pts = self.arc_pts
        for j, off in enumerate((0, 700, 1300)):
            hx = (t * 900 + off) % (W + 600) - 300
            sel = pts[np.abs(pts[:, 0] - hx) < 2]
            if len(sel):
                px, py = sel[len(sel) // 2]
                self.spark(out, px, py, 4, .9)
        # the heart outline pulses
        hx0, hy0, hx1, hy1 = self.heart_box
        beat = max(0.0, math.sin(t * 2 * math.pi * 1.1)) ** 3
        reg = out[hy0:hy1, hx0:hx1]
        hm = self.arcs[hy0:hy1, hx0:hx1][..., None]
        reg += cv2.GaussianBlur(hm, (0, 0), 6)[..., None].reshape(hm.shape) * np.array([255, 120, 255], np.float32) * .9 * beat
        # slow push-in
        s = 1 + .04 * ease_io(t / 4.5)
        M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
        return cv2.warpAffine(np.clip(out, 0, 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ main
def build():
    return {'s1': S1(), 's2': S2(), 's10l': S10Left(), 's10r': S10Right(), 's11': S11(), 's12': S12(),
            'split': SplitGraphics(), 'end': EndCard()}


def process(frame_src, f, R):
    frame = cv2.resize(frame_src, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if 6 <= f <= 125:
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


def film_frame(cap_frame, f, R, last):
    """(output frame, composited source frame) for film frame f.

    Past the end of the source, `last` (the final composited source frame) sits under the card."""
    shot = process(cap_frame, f, R) if f <= LAST_SRC else last
    out = shot
    if f >= END_START:
        t = (f - END_START) / FPS
        card = R['end'].frame(t)
        k = min(1.0, (f - END_START) / XFADE)
        if k < 1:
            # the shot washes into the card's violet before the card resolves
            wash = (out.astype(np.float32) * .55 + np.array([110, 30, 60], np.float32) * .45)
            mixed = out.astype(np.float32) * (1 - ease_io(min(1, k * 1.6))) + wash * ease_io(min(1, k * 1.6))
            out = np.clip(mixed * (1 - ease_io(k)) + card.astype(np.float32) * ease_io(k), 0, 255).astype(np.uint8)
        else:
            out = card
        if f > END_LAST - 10:
            out = (out * ((END_LAST - f) / 10)).astype(np.uint8)
    return out, shot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(JOB, 'output', 'wagon-commercial.mp4'))
    ap.add_argument('--stills', nargs='*', type=int, help='source frames to export as JPEGs')
    ap.add_argument('--from', dest='f0', type=int, default=START)
    ap.add_argument('--to', dest='f1', type=int, default=END_LAST)
    args = ap.parse_args()
    R = build()
    cap = cv2.VideoCapture(SRC)
    if args.stills is not None:
        d = os.path.join(JOB, 'output', 'stills')
        os.makedirs(d, exist_ok=True)
        for f in args.stills:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(f, LAST_SRC))
            ok, fr = cap.read()
            last = process(fr, LAST_SRC, R) if f > LAST_SRC else None
            out, _ = film_frame(fr, f, R, last)
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
        fr = None
        if f <= LAST_SRC:
            ok, fr = cap.read()
        out, shot = film_frame(fr, f, R, last)
        if f <= LAST_SRC:
            last = shot
        enc.stdin.write(out.tobytes())
        if (f - args.f0) % 48 == 0:
            print(f'\r{f - args.f0}/{total}', end='', flush=True)
    enc.stdin.close()
    enc.wait()
    # audio: the edit's own track from the new start, padded with silence under the end card
    dur = total / FPS
    subprocess.run([ff, '-loglevel', 'error', '-y', '-i', picture, '-ss', f'{args.f0 / FPS:.4f}', '-i', SRC,
                    '-filter_complex', f'[1:a]apad,atrim=0:{dur:.3f}[a]', '-map', '0:v', '-map', '[a]',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', args.out], check=True)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
