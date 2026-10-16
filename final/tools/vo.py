"""Builds the voiceover stem from the ElevenLabs narrator read (voice Darian, eleven_v4).

The script was read in one pass so the tone stays even from line to line. Take C
(final/audio/vo_lines/take_c.mp3) is the one used: its pauses split it into exactly the 10
script lines, and it ends cleanly. Each line is cut from the take, levelled to the same speech
level and placed on its shot (times are on the joined cut's timeline, in seconds).

Usage: python3 final/tools/vo.py -> final/audio/vo.wav (48 kHz stereo, the picture's length)
"""
import os
import subprocess

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
LENGTH = 113.7708
TAKE = os.path.join(FINAL, 'audio', 'vo_lines', 'take_c.mp3')
LEVEL_DB = -21.0           # speech RMS of every line (premaster)

# (where the line sits in the take, where it starts on the cut, text[, gain dB])
LINES = [
    ((0.10, 2.02), 0.60, 'Everyone wants meaningful connection.'),                  # shot 1, Jessica on the app
    ((2.42, 4.51), 36.25, 'But in recovery, finding it can feel different.'),       # her reaction close-up
    ((5.17, 8.19), 42.35, "The places that once felt familiar don't always feel the same anymore."),  # Ethan at the bar
    ((8.73, 11.53), 64.35, "But that doesn't mean finding connection in recovery doesn't exist."),   # Ethan back home
    ((12.20, 13.18), 77.00, "That's why there's Wagon."),                           # split-screen reveal, after the logo pops
    ((13.79, 17.43), 78.90, 'A dedicated place for love, dating, friendship and connection in recovery.'),  # two phones
    ((17.91, 19.76), 83.20, 'No stigma. No social pressure.'),                      # her match
    ((20.15, 21.03), 86.30, 'Just people who get it.'),                             # Ethan and Marcus chat
    ((21.75, 24.68), 98.40, 'Whatever connection means to you, find people who understand the journey.'),  # payoffs
    ((25.30, 25.75), 109.95, 'Wagon.', 5.0),                                       # end card, after the logo pops
]


def load(path):
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


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


def main():
    take = load(TAKE)
    take = sosfilt(butter(2, 80, 'highpass', fs=SR, output='sos'), take)   # clear any low rumble
    out = np.zeros(int(round(LENGTH * SR)))
    pre, post = .06, .18
    for (s, e), at, text, *gain in LINES:
        i0, i1 = int((s - pre) * SR), min(len(take), int((e + post) * SR))
        x = take[i0:i1].copy()
        f0, f1 = int(.02 * SR), int(.08 * SR)
        x[:f0] *= np.linspace(0, 1, f0)
        x[-f1:] *= np.linspace(1, 0, f1)
        x *= 10 ** ((LEVEL_DB + sum(gain)) / 20) / speech_rms(x)
        j = int(round((at - pre) * SR))
        out[j:j + len(x)] += x[:len(out) - j]
        print(f'{at:7.2f}-{at + e - s:7.2f}  {text}')
    out = limit(out)
    os.makedirs(os.path.join(FINAL, 'audio'), exist_ok=True)
    sf.write(os.path.join(FINAL, 'audio', 'vo.wav'), np.stack([out, out], 1).astype(np.float32), SR, subtype='FLOAT')
    print('wrote final/audio/vo.wav, peak', round(20 * np.log10(np.abs(out).max()), 1), 'dBFS')


if __name__ == '__main__':
    main()
