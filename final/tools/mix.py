"""Mixes the Wagon spot and lays it onto the joined picture (parts 1-8).

Stems (48 kHz stereo):
- dialogue: the clips' own audio (dialogue, room and bar ambience). Removed where the brief asks:
  the clips' two separate pings around 69-71.6 s (replaced by the one shared notification
  in the SFX), and the app beats 76.17-89.5 s, which are music-only.
- music: final/audio/music.wav, ducked under speech.
- sfx: final/audio/sfx.wav, plus room tone wherever the dialogue stem is silent or cut.
Master: -14 LUFS integrated, -1.5 dBTP (two-pass loudnorm, linear).

Usage: python3 final/tools/mix.py -> final/Wagon_Full_master.mp4, final/audio/stems/*.flac
"""
import json
import os
import subprocess

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
FF = imageio_ffmpeg.get_ffmpeg_exe()
A = os.path.join(FINAL, 'audio')


def run(*args):
    return subprocess.run([FF, '-hide_banner', *args], check=True, capture_output=True, text=True)


def curve(points, n):
    t = np.arange(n) / SR
    ts, v = zip(*points)
    return np.interp(t, ts, v).astype(np.float32)


def gate(n, cuts, fade):
    """1 everywhere except inside each (start, end) cut, with linear fades."""
    pts = [(0, 1)]
    for s, e in cuts:
        pts += [(s - fade, 1), (s, 0), (e, 0), (e + fade, 1)]
    pts.append((n / SR, 1))
    return curve(pts, n)


def main():
    video = os.path.join(FINAL, 'joined_raw.mp4')
    if not os.path.exists(video):
        run('-y', '-f', 'concat', '-safe', '0', '-i', os.path.join(FINAL, 'list.txt'), '-c', 'copy', video)
    prod_wav = os.path.join(A, 'production.wav')
    run('-y', '-i', video, '-vn', '-ac', '2', '-ar', str(SR), '-c:a', 'pcm_f32le', prod_wav)
    prod, _ = sf.read(prod_wav, dtype='float32')
    music, _ = sf.read(os.path.join(A, 'music.wav'), dtype='float32')
    sfx, _ = sf.read(os.path.join(A, 'sfx.wav'), dtype='float32')
    room, _ = sf.read(os.path.join(A, 'roomtone.wav'), dtype='float32')
    N = len(prod)
    pad = lambda x: np.vstack([x, np.zeros((max(0, N - len(x)), 2), np.float32)])[:N]
    music, sfx, room = pad(music), pad(sfx), pad(room)

    # dialogue stem: drop the clips' two pings and the app beats
    cuts = [(68.95, 71.75), (76.17, 89.5)]
    dlg = prod * gate(N, cuts, .08)[:, None]

    # room tone where the dialogue stem is silent or cut
    rt = curve([(0, 1), (7.6, 1), (7.9, 0), (68.85, 0), (69.0, 1), (71.7, 1), (71.9, 0), (76.05, 0), (76.2, 1),
                (89.45, 1), (89.7, 0), (101.5, 0), (101.8, 1), (109.3, 1), (109.6, 0), (N / SR, 0)], N)
    sfx_stem = sfx + room * rt[:, None]

    # duck the music under speech (speech band of the dialogue stem)
    sos = butter(2, [250, 3500], 'bandpass', fs=SR, output='sos')
    sp = sosfilt(sos, dlg.mean(1))
    hop = 480
    e = np.sqrt(np.convolve(sp ** 2, np.ones(hop) / hop, 'same')[::hop])
    edb = 20 * np.log10(e + 1e-9)
    red = np.clip((edb + 40) * .45, 0, 5.0)          # up to 5 dB of ducking
    sm = np.zeros_like(red)
    for i in range(1, len(red)):                     # fast attack, slow release
        a = .45 if red[i] > sm[i - 1] else .035
        sm[i] = sm[i - 1] + a * (red[i] - sm[i - 1])
    duck = 10 ** (-np.repeat(sm, hop)[:N] / 20)
    duck = np.concatenate([duck, np.ones(N - len(duck))]).astype(np.float32)
    music_stem = music * duck[:, None] * 10 ** (1.5 / 20)

    mix = dlg + music_stem + sfx_stem
    os.makedirs(os.path.join(A, 'stems'), exist_ok=True)
    for name, x in (('dialogue', dlg), ('music', music_stem), ('sfx', sfx_stem)):
        sf.write(os.path.join(A, 'stems', name + '.flac'), np.clip(x, -1, 1), SR, subtype='PCM_24')
    pre = os.path.join(A, 'mix_premaster.wav')
    sf.write(pre, mix, SR, subtype='FLOAT')

    # master: two-pass loudnorm to -14 LUFS / -1 dBTP
    r = subprocess.run([FF, '-hide_banner', '-i', pre, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=20:print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True)
    m = json.loads(r.stderr[r.stderr.rindex('{'):r.stderr.rindex('}') + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=20:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
          f"aresample={SR}")
    master_wav = os.path.join(A, 'mix_master.wav')
    run('-y', '-i', pre, '-af', af, '-c:a', 'pcm_s24le', master_wav)
    out = os.path.join(FINAL, 'Wagon_Full_master.mp4')
    run('-y', '-i', video, '-i', master_wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
        '-shortest', '-movflags', '+faststart', out)
    print('premaster input', m['input_i'], 'LUFS, TP', m['input_tp'])
    print('wrote', out)


if __name__ == '__main__':
    main()
