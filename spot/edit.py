"""Cuts the Kinly launch spot from the supplied footage.

Timeline -> per frame: reframe the source (crop + slow push), pin the exact
app UI onto tracked green screens, grade every shot to one look, lay titles
over it, add grain, and pipe the frames to ffmpeg.

Usage: python3 spot/edit.py [--out output/kinly-launch-spot.mp4] [--stills]
Needs: spot/track.py run first, and spot/build/ (supers.mjs, render.mjs --end-only).
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite import composite, crossfade, fit_ui, load_track, screen_aspect, slide, ui_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1920, 1080, 25
FOOT = os.path.join(ROOT, 'footage')
BUILD = os.path.join(ROOT, 'spot', 'build')


# ---------------------------------------------------------------- UI schedules
def pov_ui(t):
    """Welcome until the thumb press, then push to step 2."""
    k = min(max((t - 7.3) / .45, 0), 1)
    if k <= 0:
        return ['01-welcome']
    if k >= 1:
        return ['02-intent']
    return [('slide', '01-welcome', '02-intent', k)]


# ---------------------------------------------------------------- timeline
# src_in: seconds into the source. crop: (x, y, w, h) in source pixels, 16:9.
# push: zoom at start and end of the shot. grade: (b, g, r) gains, exposure.
SHOTS = [
    dict(clip='broll-friends-hands', src_in=.5, dur=3.6, push=(1.0, 1.07),
         grade=((1.0, .98, 1.0), 1.0)),
    dict(clip='gs-pov-lap', src_in=3.9, dur=6.4, push=(1.0, 1.06), ui=pov_ui,
         grade=((1.0, 1.0, 1.0), .97)),
    dict(clip='gs-autumn-hand', src_in=1.0, dur=4.0, crop=(0, 990, 1080, 608), push=(1.0, 1.1),
         ui=lambda t: ['03-discover'], grade=((1.02, .97, .96), .98)),
    dict(clip='broll-smile', src_in=3.4, dur=3.0, push=(1.04, 1.1),
         grade=((1.0, .95, 1.0), 1.0)),
    dict(clip='gs-park-pair', src_in=.4, dur=4.4, push=(1.0, 1.06),
         ui=lambda t: ['04-match', '04-match'], grade=((1.0, .93, 1.0), .97)),
    dict(clip='broll-friends-hands', src_in=6.6, dur=3.0, push=(1.08, 1.02),
         grade=((1.0, .98, 1.0), 1.0)),
    dict(clip='gs-couch-evening', src_in=1.6, dur=5.6, crop=(0, 300, 720, 405), push=(1.0, 1.07),
         ui=lambda t: ['05-chat'], grade=((1.1, 1.0, .9), 1.06)),
    dict(clip='endcard', src_in=0, dur=4.4, xfade=.6, graded=False),
]

# (layer, in, out) in spot seconds; fades are 0.45s with a small rise
SUPERS = [
    ('open-1', .45, 3.25), ('open-2', 1.05, 3.25), ('open-3', 1.65, 3.25),
    ('meet', 4.3, 6.9), ('choose', 8.2, 9.95),
    ('match', 17.9, 20.9), ('circle', 21.9, 24.4), ('plans', 25.4, 29.4),
]


# ---------------------------------------------------------------- readers
class Clip:
    def __init__(self, name):
        path = os.path.join(BUILD, 'endcard.mp4') if name == 'endcard' else os.path.join(FOOT, name + '.mp4')
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1
        self.frame = None
        self.track = None
        tpath = os.path.join(FOOT, 'tracks', name + '.json')
        if os.path.exists(tpath):
            self.track = load_track(name)
            self.aspect = screen_aspect(self.track['corners'])

    def at(self, t):
        i = min(int(round(t * self.fps)), self.n - 1)
        if i != self.pos:
            if i != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
            self.pos = i
        return i, self.frame


_clips = {}


def clip(name):
    if name not in _clips:
        _clips[name] = Clip(name)
    return _clips[name]


# ---------------------------------------------------------------- look
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = (1 - .42 * np.clip(((xx / W - .5) ** 2 / .34 + (yy / H - .5) ** 2 / .3) - .35, 0, 1) ** 1.3)[..., None]
PLUM = np.array([.30, .06, .24], np.float32)    # BGR lift in the shadows
PEACH = np.array([.40, .62, 1.0], np.float32)   # BGR warmth in the highlights
rng = np.random.default_rng(3)
GRAIN = [cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), .7)[..., None] * .017 for _ in range(6)]


def grade(img, gains, exposure):
    x = img.astype(np.float32) / 255 * np.array(gains, np.float32) * exposure
    x = np.clip(x, 0, 1)
    s = x * x * (3 - 2 * x)
    x = x + (s - x) * .32
    lum = (x[..., 0:1] * .114 + x[..., 1:2] * .587 + x[..., 2:3] * .299)
    x = lum + (x - lum) * .86
    x = x + PLUM * (1 - lum) ** 3 * .09 + PEACH * lum ** 3 * .05
    x = x * .96 + .018  # soft blacks, rolled-off whites
    return x * VIGNETTE


def finish(x, n):
    x = x + GRAIN[n % len(GRAIN)] * (0.6 + x.mean(axis=2, keepdims=True))
    return np.clip(x * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- titles
_layers = {}


def layer(name):
    if name not in _layers:
        im = cv2.imread(os.path.join(BUILD, 'supers', name + '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        _layers[name] = im
    return _layers[name]


def ease(k):
    return 1 - (1 - k) ** 3


def apply_supers(x, t):
    for name, a, b in SUPERS:
        if not (a <= t <= b):
            continue
        k = min(ease(min((t - a) / .45, 1)), 1 - ease(max(0, min((t - (b - .4)) / .4, 1))))
        if k <= 0:
            continue
        L = layer(name)
        dy = int(round((1 - k) * 22))
        rgb, al = L[..., :3], L[..., 3:4] * k
        # scrim: darken softly behind the text so it reads on any shot
        ys, xs = np.nonzero(L[..., 3] > .05)
        cy, cx = ys.mean(), xs.mean()
        sc = np.exp(-(((xx - cx) / 900) ** 2 + ((yy - cy) / 420) ** 2))[..., None] * .38 * k
        x *= 1 - sc
        if dy:
            rgb = np.roll(rgb, dy, axis=0)
            al = np.roll(al, dy, axis=0)
        x[:] = x * (1 - al) + rgb * al
    return x


# ---------------------------------------------------------------- frame
def reframe(src, crop, zoom):
    """Affine from source pixels to output pixels: crop, then zoom about its centre."""
    sh, sw = src.shape[:2]
    cx, cy, cw, ch = crop if crop else (0, 0, sw, sh)
    s = W / cw * zoom
    ox, oy = cx + cw / 2, cy + ch / 2
    return np.float32([[s, 0, W / 2 - s * ox], [0, s, H / 2 - s * oy]])


def textures(names, aspects):
    out = []
    for n, a in zip(names, aspects):
        if isinstance(n, tuple):
            _, u0, u1, k = n
            out.append(slide(fit_ui(ui_image(u0), a), fit_ui(ui_image(u1), a), k))
        else:
            out.append(fit_ui(ui_image(n), a))
    return out


def shot_frame(shot, t_local):
    c = clip(shot['clip'])
    t_src = shot['src_in'] + t_local
    i, src = c.at(t_src)
    z0, z1 = shot.get('push', (1, 1))
    k = t_local / shot['dur']
    zoom = z0 + (z1 - z0) * (k * k * (3 - 2 * k))
    M = reframe(src, shot.get('crop'), zoom)
    up = M[0, 0] > 1
    frame = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA,
                           borderMode=cv2.BORDER_REFLECT)
    if c.track is not None and 'ui' in shot:
        q = c.track['corners'][min(i, len(c.track['corners']) - 1)]
        q = q @ M[:, :2].T + M[:, 2]
        frame = composite(frame, q, textures(shot['ui'](t_src), c.aspect))
    if shot.get('graded', True):
        gains, exp = shot['grade']
        return grade(frame, gains, exp)
    return frame.astype(np.float32) / 255


def build_timeline():
    t = 0.0
    spans = []
    for s in SHOTS:
        start = t - s.get('xfade', 0)
        spans.append((start, start + s['dur'], s))
        t = start + s['dur']
    return spans, t


def render_frame(spans, t, n):
    active = [(a, b, s) for a, b, s in spans if a <= t < b]
    x = None
    for a, b, s in active:
        f = shot_frame(s, t - a)
        if x is None:
            x = f
        else:
            k = min((t - a) / s['xfade'], 1)
            x = x * (1 - k) + f * k
    x = apply_supers(x, t)
    x *= min(1.0, t / .5)  # fade up from black
    return finish(x, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'output', 'kinly-launch-spot.mp4'))
    ap.add_argument('--stills', action='store_true')
    args = ap.parse_args()
    spans, total = build_timeline()
    print(f'spot length {total:.2f}s')
    if args.stills:
        d = os.path.join(ROOT, 'output', 'stills')
        os.makedirs(d, exist_ok=True)
        for t in [2.2, 5.2, 8.6, 12.0, 15.6, 19.4, 23.0, 27.0, 32.5]:
            spans_ = spans
            _clips.clear()
            cv2.imwrite(os.path.join(d, f'spot-{t:05.1f}s.jpg'), render_frame(spans_, t, int(t * FPS)),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('stills written to', d)
        return
    import imageio_ffmpeg
    ff = os.environ.get('FFMPEG') or imageio_ffmpeg.get_ffmpeg_exe()
    enc = subprocess.Popen([ff, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-maxrate', '12M', '-bufsize', '24M',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out], stdin=subprocess.PIPE)
    frames = int(round(total * FPS))
    for n in range(frames):
        enc.stdin.write(render_frame(spans, n / FPS, n).tobytes())
        if n % FPS == 0:
            print(f'\r{n / FPS:5.1f}s / {total:.1f}s', end='', flush=True)
    enc.stdin.close()
    enc.wait()
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
