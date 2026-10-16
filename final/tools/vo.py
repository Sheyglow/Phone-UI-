"""Builds the voiceover stem from the Higgsfield narrator lines (voice Holden, Seed Audio 1.0).

Each script line was generated on its own (final/audio/vo_holden/lineNN.wav). For each line the
speech is found inside a search window (lines 1 and 4 open with a second of garbled engine output
before a clear pause, and a few files start with a click, so the window starts after those),
levelled to the same speech level and placed on its shot. A line given as two pieces is joined
with a shorter pause than the read had. Times are on the joined cut's timeline, in seconds.

Usage: python3 final/tools/vo.py -> final/audio/vo.wav (48 kHz stereo, the picture's length)
"""
import os

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
LENGTH = 113.7708
LINES_DIR = os.path.join(FINAL, 'audio', 'vo_holden')
LEVEL_DB = -21.0           # speech RMS of every line (premaster)
PIECE_GAP = .45            # pause between the pieces of a two-piece line

# (line number, search windows in its file, where it starts on the cut, text[, gain dB])
LINES = [
    (1, [(1.6, 4.4)], 0.60, 'Everyone wants meaningful connection.'),                         # shot 1, Jessica on the app
    (2, [(0.0, 3.79)], 36.25, 'But in recovery, finding it can feel different.'),             # her reaction close-up
    (3, [(0.0, 4.51)], 42.35, "The places that once felt familiar don't always feel the same anymore."),  # Ethan at the bar
    (4, [(2.3, 6.90)], 64.35, "But that doesn't mean finding connection in recovery doesn't exist."),     # Ethan back home
    (5, [(0.5, 4.0)], 76.85, "That's why there's Wagon."),                                    # split-screen reveal, after the logo pops
    (6, [(0.3, 5.57)], 78.55, 'A dedicated place for love, dating, friendship and connection in recovery.'),  # two phones
    (7, [(0.6, 2.4), (2.4, 4.4)], 83.65, 'No stigma. No social pressure.'),                  # her match
    (8, [(0.0, 1.66)], 86.90, 'Just people who get it.'),                                     # Ethan and Marcus chat
    (9, [(0.0, 4.63)], 97.80, 'Whatever connection means to you, find people who understand the journey.'),  # payoffs
    (10, [(0.6, 1.88)], 109.95, 'Wagon.', 5.0),                                               # end card, after the logo pops
]


def speech_span(x, rel_db=-40):
    """First and last sample of speech in x (frames within rel_db of the loudest frame)."""
    hop = 240
    e = np.sqrt(np.convolve(x ** 2, np.ones(2 * hop) / (2 * hop), 'same')[::hop] + 1e-12)
    d = 20 * np.log10(e)
    on = np.flatnonzero(d > d.max() + rel_db)
    return on[0] * hop, min(len(x), (on[-1] + 1) * hop)


def speech_rms(x):
    """RMS over the voiced frames only, so pauses inside a line don't pull its level down."""
    hop = 480
    e = np.sqrt((x[:len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    voiced = e[20 * np.log10(e + 1e-12) > 20 * np.log10(e.max()) - 30]
    return np.sqrt((voiced ** 2).mean())


def limit(x, ceiling_db=-4.0, look=.005, release=.08):
    """A soft lookahead peak limiter, so no line's transient sets the master's true peak."""
    c = 10 ** (ceiling_db / 20)
    need = np.minimum(1, c / np.maximum(maximum_filter1d(np.abs(x), int(2 * look * SR)), 1e-9))
    g = np.empty_like(need)
    a = np.exp(-1 / (release * SR))
    g[0] = need[0]
    for i in range(1, len(g)):                     # instant attack (lookahead covers it), smooth release
        g[i] = need[i] if need[i] < g[i - 1] else a * g[i - 1] + (1 - a) * need[i]
    return x * g


def cut_line(take, windows):
    """The speech inside each window, padded and faded, joined with PIECE_GAP between pieces."""
    pre, post = int(.05 * SR), int(.15 * SR)
    pieces = []
    for w0, w1 in windows:
        i0, i1 = int(w0 * SR), min(len(take), int(w1 * SR))
        s, e = speech_span(take[i0:i1])
        a, b = max(i0, i0 + s - pre), min(i1, i0 + e + post)
        x = take[a:b].copy()
        f0, f1 = min(int(.02 * SR), i0 + s - a), int(.08 * SR)
        if f0 > 0:
            x[:f0] *= np.linspace(0, 1, f0)
        x[-f1:] *= np.linspace(1, 0, f1)
        pieces.append((x, i0 + s - a))             # the piece, and where its speech starts in it
    out, lead = pieces[0][0], pieces[0][1]
    for x, _ in pieces[1:]:
        out = np.concatenate([out, np.zeros(int(PIECE_GAP * SR) - int(.20 * SR)), x])
    return out, lead


def main():
    hp = butter(2, 80, 'highpass', fs=SR, output='sos')
    out = np.zeros(int(round(LENGTH * SR)))
    for n, windows, at, text, *gain in LINES:
        take, sr = sf.read(os.path.join(LINES_DIR, f'line{n:02d}.wav'), dtype='float64')
        assert sr == SR
        take = sosfilt(hp, take.mean(1) if take.ndim > 1 else take)   # clear any low rumble
        x, lead = cut_line(take, windows)
        x *= 10 ** ((LEVEL_DB + sum(gain)) / 20) / speech_rms(x)
        j = int(round(at * SR)) - lead             # the speech itself starts on `at`
        out[j:j + len(x)] += x[:len(out) - j]
        print(f'{at:7.2f}-{at + (len(x) - lead - .15 * SR) / SR:7.2f}  {text}')
    out = limit(out)
    os.makedirs(os.path.join(FINAL, 'audio'), exist_ok=True)
    sf.write(os.path.join(FINAL, 'audio', 'vo.wav'), np.stack([out, out], 1).astype(np.float32), SR, subtype='FLOAT')
    print('wrote final/audio/vo.wav, peak', round(20 * np.log10(np.abs(out).max()), 1), 'dBFS')


if __name__ == '__main__':
    main()
