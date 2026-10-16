"""Sound design for the Wagon spot, synthesised so there are no licensing questions.

Events are placed on the joined cut's timeline (seconds), from the screen animations:
- Luma (0-7.7): the swipe, the like tap, the tap into the chat, the tap on the field, the
  keystrokes (one tick per letter, in time with the typing), and the send.
- One shared notification at 69.15 s. It replaces the clips' two separate pings, and both
  Jessica and Ethan pick up their phones on it.
- App beats (76.17-89.5, music-only): only quiet taps where the thumbs press, and a soft
  air transition into the Wagon reveal.
- A soft transition into the end card.
- Room tone where the clips are silent (the opening, the app beats, the montage) and to fill
  the gap left by removing the clips' pings.

Usage: python3 final/tools/sfx.py -> final/audio/sfx.wav and final/audio/roomtone.wav (48 kHz stereo)
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
LENGTH = 113.77
rng = np.random.default_rng(7)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], 'bandpass', fs=SR, output='sos'), x, axis=0)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, 'lowpass', fs=SR, output='sos'), x, axis=0)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, 'highpass', fs=SR, output='sos'), x, axis=0)


def tap(level=1.0, bright=1.0):
    """A finger on glass: a tiny bright tick over a soft low thump."""
    n = int(.06 * SR)
    t = np.arange(n) / SR
    tick = bp(rng.standard_normal(n), 1800 * bright, 6500) * np.exp(-t / .0025)
    thump = np.sin(2 * np.pi * 140 * t) * np.exp(-t / .012) * .5
    x = tick * .6 + thump
    return x / np.abs(x).max() * level


def key_tick(level=1.0):
    n = int(.04 * SR)
    t = np.arange(n) / SR
    x = bp(rng.standard_normal(n), 2500, 8000) * np.exp(-t / .0018) + np.sin(2 * np.pi * 220 * t) * np.exp(-t / .006) * .25
    return x / np.abs(x).max() * level


def whoosh(dur, f0, f1, level=1.0, shape=2.0):
    """Air moving: noise whose band sweeps f0 -> f1 (a crossfaded bank of continuous band filters)."""
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    k = np.linspace(0, 1, n)
    bands = np.geomspace(f0, f1, 8)
    out = np.zeros(n)
    for i, fc in enumerate(bands):
        y = bp(noise, max(60, fc * .7), min(SR / 2 - 100, fc * 1.4))
        out += y * np.exp(-((k - i / 7) / .16) ** 2)
    env = np.sin(np.pi * k ** (1 / shape)) ** 2
    x = out * env
    return x / np.abs(x).max() * level


def chime(level=1.0):
    """The shared notification: a soft, warm two-note mallet chime (E♭6 -> A♭6), in the score's key."""
    out = np.zeros(int(1.6 * SR))
    for k, (f, a) in enumerate([(1244.5, .8), (1661.2, 1.0)]):
        n = len(out) - int(k * .11 * SR)
        t = np.arange(n) / SR
        partials = (np.sin(2 * np.pi * f * t) * np.exp(-t / .42)
                    + .25 * np.sin(2 * np.pi * f * 3.93 * t) * np.exp(-t / .06)   # mallet ping
                    + .12 * np.sin(2 * np.pi * f * .5 * t) * np.exp(-t / .5))      # warmth
        att = np.minimum(1, t / .004)
        out[int(k * .11 * SR):] += partials * att * a
    return out / np.abs(out).max() * level


def room_tone(n, level):
    """Quiet, warm interior air: pink-ish noise, low-passed."""
    w = rng.standard_normal((n, 2))
    b = lp(w, 450, 2) + .04 * lp(w, 1800, 1)   # matched to the clips' quiet room: dark, little hiss
    b = hp(b, 40, 2)
    rms = np.sqrt((b ** 2).mean())
    return b / rms * level


def place(buf, t, x, pan=0.0):
    i = int(round(t * SR))
    if x.ndim == 1:
        p = (pan + 1) / 2
        x = np.stack([x * np.sqrt(1 - p), x * np.sqrt(p)], 1) * 1.414
    j = min(i + len(x), len(buf))
    buf[i:j] += x[:j - i]


def db(v):
    return 10 ** (v / 20)


def main():
    N = int(LENGTH * SR)
    fx = np.zeros((N, 2))

    # Luma, 0-7.71: swipe, like, into chat, field, keys, send (phone sits right of frame)
    place(fx, .05, whoosh(.42, 900, 3500, db(-30)), pan=-.2)
    place(fx, .06, tap(db(-30)), .3)
    place(fx, 2.33, tap(db(-26)), .3)
    place(fx, 2.36, chime_soft(db(-36)), .3)        # a small "like" pop
    place(fx, 2.92, tap(db(-29)), .3)
    place(fx, 3.86, tap(db(-29)), .3)
    for t in (4.25, 4.46, 4.67):
        place(fx, t, key_tick(db(-31)), .3)
    for t in (5.08, 5.21, 5.375, 5.5, 5.67, 5.79, 5.96, 6.08, 6.25, 6.375, 6.54):
        place(fx, t + rng.uniform(-.01, .01), key_tick(db(-27 + rng.uniform(-1.5, 1.5))), .1)
    place(fx, 6.71, tap(db(-25)), .1)
    place(fx, 6.73, whoosh(.32, 1500, 5000, db(-31), shape=1.4), .2)   # message sent

    # Ethan's text from his friend (rev 2): the phone buzzes on the table as it lights up, then the
    # receive sound as the message banner appears. Audible over normal-level music, like 1:22-1:24.
    place(fx, 15.55, vibration(db(-23)), .35)
    place(fx, 17.0, text_received(db(-22)), .3)

    # the shared notification
    place(fx, 69.15, chime(db(-20)), .15)

    # Wagon reveal: soft air into the split screen; app beats are music-only, quiet taps only
    place(fx, 75.2, whoosh(1.2, 300, 2400, db(-28), shape=3.0))
    for t, pan in ((80.25, .5), (80.5, -.5), (83.04, .2), (85.96, -.2)):
        place(fx, t, tap(db(-31)), pan)
    place(fx, 85.98, whoosh(.3, 1500, 5000, db(-34), shape=1.4), -.2)

    # into the end card
    place(fx, 108.55, whoosh(1.0, 400, 3000, db(-29), shape=3.0))

    os.makedirs(os.path.join(FINAL, 'audio'), exist_ok=True)
    sf.write(os.path.join(FINAL, 'audio', 'sfx.wav'), fx.astype(np.float32), SR, subtype='FLOAT')
    sf.write(os.path.join(FINAL, 'audio', 'roomtone.wav'), room_tone(N, db(-46)).astype(np.float32), SR, subtype='FLOAT')
    print('wrote final/audio/sfx.wav and roomtone.wav')


def vibration(level):
    """A phone buzzing on a wooden table: two pulses of a ~175 Hz motor with table rattle."""
    out = np.zeros(int(.95 * SR))
    for start in (0.0, .5):
        n = int(.36 * SR)
        t = np.arange(n) / SR
        motor = np.sign(np.sin(2 * np.pi * 175 * t)) * .6 + np.sin(2 * np.pi * 350 * t) * .3
        rattle = bp(rng.standard_normal(n), 900, 3200) * (.5 + .5 * np.sign(np.sin(2 * np.pi * 175 * t)))
        x = lp(motor, 1400, 2) + .35 * rattle
        env = np.minimum(1, t / .015) * np.minimum(1, (t[-1] - t) / .03)
        i = int(start * SR)
        out[i:i + n] += x * env
    return out / np.abs(out).max() * level


def text_received(level):
    """Incoming message: a short, bright two-note pop (distinct from the Wagon notification chime)."""
    out = np.zeros(int(.6 * SR))
    for k, f in enumerate((1318.5, 1760.0)):   # E6 -> A6
        n = len(out) - int(k * .075 * SR)
        t = np.arange(n) / SR
        x = (np.sin(2 * np.pi * f * t) + .3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t / .07) * np.minimum(1, t / .002)
        out[int(k * .075 * SR):] += x
    return out / np.abs(out).max() * level


def chime_soft(level):
    """The like: a single short, soft mallet blip."""
    n = int(.35 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 1046.5 * t) * np.exp(-t / .08) + .2 * np.sin(2 * np.pi * 4113 * t) * np.exp(-t / .02)
    return x / np.abs(x).max() * level


if __name__ == '__main__':
    main()
