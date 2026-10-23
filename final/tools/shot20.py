"""Makes Shot 20 (Ethan's phone, the chat with Marcus) readable, without changing the spot's length.

- A slow, eased push-in toward the phone across the whole shot, ending ZOOM_END closer, with a light
  sharpen so the chat text stays crisp.
- The shot is held HOLD frames longer on the finished chat (the push-in keeps moving through the hold),
  so Marcus's reply stays on screen ~3.8 s instead of ~2.3 s. Those frames come off the start of the next
  shot (the coffee date), so every later frame, and all the audio, stays exactly where it was.

Also lifts the chat text: the writing in Ethan's orange bubble to clean white, Marcus's reply darker.
Reads final/joined_raw.mp4 and writes final/picture.mp4, which mix.py uses as the picture when present.
Usage: python3 final/tools/shot20.py [--preview OUT.mp4]   (--preview: just Shot 20 with a second either side,
       with the master's audio for that stretch)
"""
import os
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 24
FIRST, LAST = 2049, 2147          # Shot 20's frames in the joined cut (1:25.375-1:29.500)
HOLD = 36                         # 1.5 s longer on the finished chat
ZOOM_END = 1.5
PHONE = (650, 470)                # centre of the phone screen in the frame


ROI = (380, 120, 980, 880)          # x0, y0, x1, y1: the phone across the whole shot


def legible(img):
    """Ethan's bubble: deepen the orange a touch and lift its text to clean white. Marcus's reply and the
    small labels: darken the grey text toward black. Only thin text strokes are touched, not the avatar."""
    x0, y0, x1, y1 = ROI
    a = np.asarray(img).astype(np.float32)
    roi = a[y0:y1, x0:x1]
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    mx, mn = roi.max(2), roi.min(2)
    lum = .299 * r + .587 * g + .114 * b
    sat = (mx - mn) / (mx + 1e-6)
    panel = ndi.binary_closing((lum > 200) & (sat < .12), np.ones((15, 15)))
    lab, _ = ndi.label(panel)
    panel = ndi.binary_fill_holes(lab == lab[PHONE[1] + 50 - y0, PHONE[0] - x0])
    orange = (r > 170) & (r - b > 90) & (g > 60) & (g < 190) & panel
    bubble = ndi.binary_fill_holes(ndi.binary_closing(orange, np.ones((9, 9)))) & panel
    out = roi.copy()
    w = np.clip((lum - 150) / 70, 0, 1)[..., None]
    out = np.where(bubble[..., None], roi * np.array([.93, .86, .82]) * (1 - w) + 255 * w, out)
    dark = panel & ~bubble & (lum < 215)
    lab, n = ndi.label(dark)
    if n:
        sizes = ndi.sum(dark, lab, range(1, n + 1))
        big = np.isin(lab, np.flatnonzero(sizes > 600) + 1)     # the avatar photo, not letters
        dark &= ~ndi.binary_dilation(big, iterations=3)
    rest = dark & (sat < .25)
    f = (lum / 215) ** 0.9
    out = np.where(rest[..., None], roi * f[..., None], out)
    a[y0:y1, x0:x1] = out
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def ease(u):
    return u * u * (3 - 2 * u)


def push_in(img, k):
    """Crop toward the phone by zoom k (centre slides from the frame centre to the phone) and scale back up."""
    z = 1 + (ZOOM_END - 1) * ease(k)
    cw, ch = W / z, H / z
    cx = W / 2 + (PHONE[0] - W / 2) * ease(k)
    cy = H / 2 + (PHONE[1] - H / 2) * ease(k)
    cx = min(max(cx, cw / 2), W - cw / 2)
    cy = min(max(cy, ch / 2), H - ch / 2)
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    out = img.resize((W, H), Image.LANCZOS, box=box)
    return out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(40 + 40 * ease(k)), threshold=2))


def main(preview=None):
    src = os.path.join(FINAL, 'joined_raw.mp4')
    out = preview or os.path.join(FINAL, 'picture.mp4')
    lo, hi = (FIRST - FPS, LAST + HOLD + FPS) if preview else (0, 10 ** 9)   # output frames to write
    dec = subprocess.Popen([FF, '-v', 'error', '-i', src, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           stdout=subprocess.PIPE)
    audio = (['-ss', f'{lo / FPS:.4f}', '-t', f'{(hi - lo + 1) / FPS:.4f}',
              '-i', os.path.join(FINAL, 'audio', 'mix_master.wav'), '-c:a', 'aac', '-b:a', '256k', '-shortest']
             if preview else [])
    enc = subprocess.Popen([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', str(FPS), '-i', '-', *audio, '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
                            '-tune', 'film', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                           stdin=subprocess.PIPE)
    span = LAST + HOLD - FIRST                 # frames the push-in runs over
    size = W * H * 3
    n = i = 0                                  # input frame, output frame (same count: the length is unchanged)

    def emit(frame):
        nonlocal i
        if lo <= i <= hi:
            enc.stdin.write(frame if isinstance(frame, bytes) else frame.tobytes())
        i += 1

    while i <= hi:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        if FIRST <= n <= LAST:
            if lo <= i <= hi + HOLD:
                frame = legible(Image.frombytes('RGB', (W, H), buf))
                emit(push_in(frame, (n - FIRST) / span))
                if n == LAST:                  # the hold: the finished chat, still pushing in
                    for h in range(1, HOLD + 1):
                        emit(push_in(frame, (n + h - FIRST) / span))
            else:
                i += 1 + (HOLD if n == LAST else 0)
        elif not LAST < n <= LAST + HOLD:      # the coffee shot's first frames went to the hold
            emit(buf)
        n += 1
    enc.stdin.close()
    enc.wait()
    dec.kill()
    print(f'wrote {out}: Shot 20 now {(LAST - FIRST + 1 + HOLD) / FPS:.2f} s')


if __name__ == '__main__':
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--preview' else None)
