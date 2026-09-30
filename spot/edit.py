"""Cuts the Kinly launch spot from the supplied footage.

Timeline -> per frame: reframe the source (crop + slow push), pin the exact
app UI onto tracked green screens (the chat scrolls with the tracked finger),
grade every shot to one look, lay titles over it, add grain, and pipe the
frames to ffmpeg. The music is then laid under the picture.

Usage: python3 spot/edit.py [--out output/kinly-launch-spot.mp4] [--stills]
Needs: spot/track.py run first, and spot/build/ (supers.mjs, render.mjs --end-only).
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite import composite, fit_ui, load_track, slide, ui_image  # noqa: E402

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


# ---------------------------------------------------------------- chat scroll
# Rows of screens/05-chat.webp: the list scrolls between the header divider and the input bar.
CHAT_TOP, CHAT_BOTTOM, CHAT_END = 278, 1775, 1262  # CHAT_END: bottom of "I'm in. Save me a seat."
CHAT_BG = (31, 15, 20)                              # BGR of the chat background


def chat_strip():
    """The supplied conversation followed by the designed continuation, as one tall list (UI px)."""
    chat = ui_image('05-chat').copy()
    chat[1278:1318, 780:900] = CHAT_BG  # "Read" moves down to the newest message
    ext = cv2.imread(os.path.join(BUILD, 'chat-extension.png'))
    assert ext is not None and ext.shape[1] == chat.shape[1], 'run node spot/supers.mjs first'
    strip = np.concatenate([chat[CHAT_TOP:CHAT_END], ext])
    return strip, strip.shape[0] - (CHAT_BOTTOM - CHAT_TOP)


def chat_scroll_curve(max_off, name='gs-couch-evening'):
    """Scroll offset of the chat (UI px) per source frame, driven by the tracked fingertip.

    The rectified screen is 2000 units tall, the same as the UI, so the list follows the
    finger 1:1 while it drags upward. On release it coasts with iOS-like momentum. Past the
    newest message it rubber-bands and settles back.
    """
    with open(os.path.join(FOOT, 'tracks', name + '-finger.json')) as f:
        data = json.load(f)
    fps = data['fps']
    dt = 1 / fps
    tip = np.array([[np.nan if v is None else v for v in r[1:]] for r in data['tip']], np.float64)
    idx = np.arange(len(tip))
    for j in range(2):
        ok = ~np.isnan(tip[:, j])
        tip[:, j] = np.interp(idx, idx[ok], tip[ok, j])
    k = np.exp(-.5 * (np.arange(-4, 5) / 1.5) ** 2)
    k /= k.sum()
    x = np.convolve(np.pad(tip[:, 0], 4, mode='edge'), k, 'valid')
    y = np.convolve(np.pad(tip[:, 1], 4, mode='edge'), k, 'valid')

    def rubber(d, dim=1500.0):
        return (1 - 1 / (d * .55 / dim + 1)) * dim

    off, vel, out = 0.0, 0.0, [0.0]
    for i in range(1, len(y)):
        v = y[i] - y[i - 1]
        if v < -6 and x[i] < 720:            # finger on the glass, dragging up
            off += -v
            vel = -v / dt
        else:                                 # released: momentum, then settle
            if off > max_off:
                off = max_off + (off - max_off) * np.exp(-dt / .12)
                vel = 0.0
            else:
                off += vel * dt
                vel *= np.exp(-dt / .3)
        out.append(off if off <= max_off else max_off + rubber(off - max_off))
    return fps, np.clip(np.array(out), 0, None)


_chat = {}


def couch_ui(t):
    if not _chat:
        _chat['strip'], _chat['max'] = chat_strip()
        _chat['fps'], _chat['off'] = chat_scroll_curve(_chat['max'])
    off = _chat['off'][min(int(round(t * _chat['fps'])), len(_chat['off']) - 1)]
    return [('chat', off)]


def chat_view(off):
    """The chat screen with its message list scrolled by off UI pixels."""
    view = ui_image('05-chat').copy()
    strip, h = _chat['strip'], CHAT_BOTTOM - CHAT_TOP
    o = int(round(off))
    rows = strip[o:o + h]
    if len(rows) < h:  # overscroll past the newest message shows plain background
        rows = np.concatenate([rows, np.full((h - len(rows), strip.shape[1], 3), CHAT_BG, np.uint8)])
    view[CHAT_TOP:CHAT_BOTTOM] = rows
    return view


# ---------------------------------------------------------------- music
# "Familiar Roads" by Tanner Helland, CC BY 4.0 (see music/CREDITS.md). 115 bpm, and the full
# band enters at bar 4 (8.35 s). The song starts with the spot, and every cut lands on a beat.
MUSIC = os.path.join(ROOT, 'music', 'familiar-roads.mp3')
BEAT = 60 / 115


def beats(n):
    return n * BEAT


# Physical screen aspect (width / height) of the phone in each clip. The tracked quad is
# foreshortened by perspective, so using its on-frame shape would squash the UI.
PHONE_ASPECT = {
    'gs-pov-lap': 750 / 1334,         # iPhone 8
    'gs-park-pair': 1125 / 2436,      # iPhone X / 11 Pro
    'gs-couch-evening': 1179 / 2556,  # iPhone 14 / 15 Pro
    'gs-autumn-hand': 1080 / 2160,    # 18:9 LG
}


# ---------------------------------------------------------------- timeline
# src_in: seconds into the source. crop: (x, y, w, h) in source pixels, 16:9.
# push: zoom at start and end of the shot. grade: (b, g, r) gains, exposure.
SHOTS = [
    dict(clip='broll-friends-hands', src_in=.5, dur=beats(7), push=(1.0, 1.07),
         grade=((1.0, .98, 1.0), 1.0)),
    dict(clip='gs-pov-lap', src_in=3.9, dur=beats(12), push=(1.0, 1.06), ui=pov_ui,
         grade=((1.0, 1.0, 1.0), .97)),
    dict(clip='gs-autumn-hand', src_in=1.0, dur=beats(8), crop=(0, 990, 1080, 608), push=(1.0, 1.1),
         ui=lambda t: ['03-discover'], grade=((1.02, .97, .96), .98)),
    dict(clip='broll-smile', src_in=3.4, dur=beats(5), push=(1.04, 1.1),
         grade=((1.0, .95, 1.0), 1.0)),
    dict(clip='gs-park-pair', src_in=.4, dur=beats(9), push=(1.0, 1.06),
         ui=lambda t: ['04-match', '04-match'], grade=((1.0, .93, 1.0), .97)),
    dict(clip='broll-friends-hands', src_in=6.6, dur=beats(6), push=(1.08, 1.02),
         grade=((1.0, .98, 1.0), 1.0)),
    dict(clip='gs-couch-evening', src_in=1.6, dur=beats(10), crop=(0, 300, 720, 405), push=(1.0, 1.07),
         ui=couch_ui, grade=((1.1, 1.0, .9), 1.06)),
    # end card crossfades in on the bar-14 downbeat
    dict(clip='endcard', src_in=0, dur=4.6, xfade=beats(1), graded=False),
]

# (layer, in, out) in spot seconds; fades are 0.45s with a small rise
SUPERS = [
    ('open-1', beats(1), 3.3), ('open-2', beats(2), 3.3), ('open-3', beats(3), 3.3),
    ('meet', 4.3, 6.9), ('choose', 8.0, 9.8),
    ('match', 17.2, 20.95), ('circle', 21.8, 24.4), ('plans', 25.0, 29.0),
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
            self.aspect = [PHONE_ASPECT[name]] * self.track['screens']

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
        if isinstance(n, tuple) and n[0] == 'chat':
            out.append(fit_ui(chat_view(n[1]), a))
        elif isinstance(n, tuple):
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
        for t in [2.2, 5.2, 8.6, 12.0, 15.6, 19.4, 23.0, 25.4, 27.2, 28.6, 32.5]:
            spans_ = spans
            _clips.clear()
            cv2.imwrite(os.path.join(d, f'spot-{t:05.1f}s.jpg'), render_frame(spans_, t, int(t * FPS)),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('stills written to', d)
        return
    import imageio_ffmpeg
    ff = os.environ.get('FFMPEG') or imageio_ffmpeg.get_ffmpeg_exe()
    silent = os.path.join(BUILD, 'spot-picture.mp4')
    enc = subprocess.Popen([ff, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-maxrate', '12M', '-bufsize', '24M',
                            '-pix_fmt', 'yuv420p', silent], stdin=subprocess.PIPE)
    frames = int(round(total * FPS))
    for n in range(frames):
        enc.stdin.write(render_frame(spans, n / FPS, n).tobytes())
        if n % FPS == 0:
            print(f'\r{n / FPS:5.1f}s / {total:.1f}s', end='', flush=True)
    enc.stdin.close()
    enc.wait()
    # music from the top of the song, faded out under the end card, loudness-normalised for web
    fade = 2.5
    subprocess.run([ff, '-loglevel', 'error', '-y', '-i', silent, '-i', MUSIC, '-filter_complex',
                    f'[1:a]atrim=0:{total:.3f},asetpts=N/SR/TB,afade=t=out:st={total - fade:.3f}:d={fade},'
                    'loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]',
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-shortest', '-movflags', '+faststart', args.out], check=True)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
