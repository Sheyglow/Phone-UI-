"""Builds the voiceover stem from the Higgsfield narrator lines (Seed Audio 1.0, paid plan).

Each script line was generated on its own, one folder per candidate voice
(final/audio/vo_female/<voice>/lineNN.wav). Lines 1-4 were read a little slower (calm, reflective);
lines 5-10, after the Wagon reveal, at normal pace. For each line the speech is found (clicks and
breaths around it are left out), levelled to the same speech level, given its tone (a touch of
warmth before the reveal, a touch of presence after it) and placed at the client's start time.
The line spans are written to final/audio/vo_spans.json for the music ducking in mix.py.

Usage: python3 final/tools/vo.py [voice] -> final/audio/vo.wav (48 kHz stereo, the picture's length)
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
LENGTH = 113.7708
VOICE = 'human'
LEVEL_DB = -21.0           # speech RMS of every line (premaster)

# (line number, start on the cut from the client's timings, tone, text[, gain dB])
LINES = [
    (1, 1.20, 'intimate', 'Everyone wants meaningful connection.'),
    (2, 36.00, 'intimate', 'But in recovery, finding it can feel different.'),
    (3, 60.00, 'intimate', "The places that once felt familiar don't always feel the same anymore."),
    (4, 70.00, 'intimate', "But that doesn't mean connection in recovery doesn't exist."),
    (5, 77.00, 'bright', "That's why there's Wagon."),
    (6, 80.00, 'bright', 'A dedicated place for love, dating, friendship and connection in recovery.'),
    (7, 90.00, 'bright', 'No stigma. No social pressure.'),
    (8, 94.00, 'bright', 'Just people who get it.'),
    (9, 101.00, 'bright', 'Whatever connection means to you, find people who understand the journey.'),
    (10, 110.00, 'bright', 'Wagon.', 3.0),
]


def speech_span(x):
    """Start and end sample of the line: voiced stretches, leaving out clicks and breaths."""
    hop = 240
    e = np.sqrt(np.convolve(x ** 2, np.ones(2 * hop) / (2 * hop), 'same')[::hop] + 1e-12)
    d = 20 * np.log10(e)
    on = d > d.max() - 30
    runs, i = [], 0
    while i < len(on):                             # runs of activity, merged across gaps < 0.5 s
        if on[i]:
            j = i
            while j < len(on) and (on[j] or np.any(on[j:j + int(.5 * SR / hop)])):
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    peak = d.max()
    keep = [(a, b) for a, b in runs
            if (b - a) * hop / SR >= .12 and np.percentile(d[a:b], 90) > peak - 12]   # not a click or a breath
    keep = keep or runs
    return keep[0][0] * hop, min(len(x), keep[-1][1] * hop)


def speech_rms(x):
    """RMS over the voiced frames only, so pauses inside a line don't pull its level down."""
    hop = 480
    e = np.sqrt((x[:len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    voiced = e[20 * np.log10(e + 1e-12) > 20 * np.log10(e.max()) - 30]
    return np.sqrt((voiced ** 2).mean())


def tone(x, kind):
    """Before the reveal a little low-mid warmth; after it a little presence, for the brighter half."""
    if kind == 'intimate':
        band = sosfilt(butter(2, [150, 400], 'bandpass', fs=SR, output='sos'), x)
        return x + band * (10 ** (1.5 / 20) - 1)
    band = sosfilt(butter(2, [2500, 6000], 'bandpass', fs=SR, output='sos'), x)
    return x + band * (10 ** (2.0 / 20) - 1)


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


def main(voice):
    hp = butter(2, 80, 'highpass', fs=SR, output='sos')
    out = np.zeros(int(round(LENGTH * SR)))
    spans = []
    pre, post = int(.06 * SR), int(.15 * SR)
    for n, at, kind, text, *gain in LINES:
        take, sr = sf.read(os.path.join(FINAL, 'audio', 'vo_female', voice, f'line{n:02d}.wav'), dtype='float64')
        assert sr == SR
        take = sosfilt(hp, take.mean(1) if take.ndim > 1 else take)   # clear any low rumble
        s, e = speech_span(take)
        a, b = max(0, s - pre), min(len(take), e + post)
        x = take[a:b].copy()
        f0, f1 = min(int(.02 * SR), s - a), int(.08 * SR)
        if f0 > 0:
            x[:f0] *= np.linspace(0, 1, f0)
        x[-f1:] *= np.linspace(1, 0, f1)
        x = tone(x, kind)
        x *= 10 ** ((LEVEL_DB + sum(gain)) / 20) / speech_rms(x)
        j = int(round(at * SR)) - (s - a)          # the speech itself starts on `at`
        out[j:j + len(x)] += x[:len(out) - j]
        end = at + (e - s) / SR
        spans.append((at, round(end, 3)))
        print(f'{at:7.2f}-{end:7.2f}  {text}')
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        assert a1 < b0, f'line ending {a1} runs into the next at {b0}'
    out = limit(out)
    os.makedirs(os.path.join(FINAL, 'audio'), exist_ok=True)
    sf.write(os.path.join(FINAL, 'audio', 'vo.wav'), np.stack([out, out], 1).astype(np.float32), SR, subtype='FLOAT')
    with open(os.path.join(FINAL, 'audio', 'vo_spans.json'), 'w') as f:
        json.dump({'voice': voice, 'spans': spans}, f)
    print(f'wrote final/audio/vo.wav ({voice}), peak', round(20 * np.log10(np.abs(out).max()), 1), 'dBFS')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else VOICE)
