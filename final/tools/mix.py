"""Mixes the Wagon spot and lays it onto the joined picture (parts 1-8).

Stems (48 kHz stereo):
- dialogue: the clips' own audio, edited to the client's notes (rev 2):
    0:07.6-0:26.1  removed: makeup foley, Ethan's room chatter and the date's small talk. The
                   date's background air carries on under the music from 0:21.3
    0:27.6         Daniel's order replaced with "I'll do an Old Fashion." (ElevenLabs, voice Cole,
                   matched in tone and room to the scene)
    0:41.95-0:47.6 removed: the gibberish after "Hey Ethan!" (bar air carries on)
    1:08.95-1:11.75 removed: the clips' two pings (one shared notification is in the SFX)
    1:16.17-1:29.5 removed: app beats are music-only
    1:29.5-end     removed: the coffee clink, Marcus's greeting and the gibberish after it
- voiceover: final/audio/vo.wav when present (ElevenLabs), else silent.
- music: final/audio/music.wav, ducked under dialogue and voiceover.
- sfx: final/audio/sfx.wav, plus room tone wherever the dialogue stem is cut.
Master: -14 LUFS integrated, -1.5 dBTP. The delivered stems carry the master's gain, so they sum to it.

Usage: python3 final/tools/mix.py -> final/Wagon_Full_master.mp4, master WAVs and final/stems/*.wav
"""
import json
import os
import subprocess

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
LENGTH = 113.7708
FF = imageio_ffmpeg.get_ffmpeg_exe()
A = os.path.join(FINAL, 'audio')
rng = np.random.default_rng(11)

# dialogue removals (client notes, rev 2) and the earlier ones
CUTS = [(7.6, 26.1), (27.5, 29.0), (41.95, 47.6), (68.95, 71.75), (76.17, LENGTH + 1)]
DANIEL_LINE_AT = 27.62      # where the new line's first word starts
DANIEL_TAKE = os.path.join(A, 'vo_dialogue', 'daniel_take1.wav')


def run(*args):
    return subprocess.run([FF, '-hide_banner', *args], check=True, capture_output=True, text=True)


def curve(points, n):
    t = np.arange(n) / SR
    ts, v = zip(*points)
    return np.interp(t, ts, v).astype(np.float32)


def gate(n, cuts, fade):
    """1 everywhere except inside each (start, end) cut, with linear fades."""
    t = np.arange(n) / SR
    g = np.ones(n)
    for s, e in cuts:
        g = np.minimum(g, np.clip(np.maximum((s - t) / fade, (t - e) / fade), 0, 1))
    return g.astype(np.float32)


def bed(src, pieces, length, xfade=.25):
    """Loop speech-free pieces of the clips' own ambience into a bed of `length` seconds."""
    out = np.zeros((int(length * SR), 2), np.float32)
    pos, k, x = 0, 0, int(xfade * SR)
    while pos < len(out):
        s0, s1 = pieces[k % len(pieces)]
        seg = src[int(s0 * SR):int(s1 * SR)].copy()
        r = np.ones(len(seg), np.float32)
        r[:x] = np.linspace(0, 1, x)
        r[-x:] = np.linspace(1, 0, x)
        seg *= r[:, None]
        j = min(pos + len(seg), len(out))
        out[pos:j] += seg[:j - pos]
        pos += len(seg) - x
        k += 1
    return out


def place_bed(dst, b, t0, t1, gain=1.0, fade=.12):
    i0, i1 = int(t0 * SR), int(t1 * SR)
    seg = b[:i1 - i0] * gain
    r = np.ones(len(seg), np.float32)
    f = int(fade * SR)
    r[:f] = np.linspace(0, 1, f)
    r[-f:] = np.linspace(1, 0, f)
    dst[i0:i1] += seg * r[:, None]


def small_room_ir(rt60=.32):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.91 * t / rt60)[:, None]
    ir = sosfilt(butter(1, 3500, 'lowpass', fs=SR, output='sos'), ir, axis=0)
    ir /= np.sqrt((ir ** 2).sum(0))
    return ir


def daniel_line(prod):
    """The new ElevenLabs line, made to sit in the restaurant: tone, room and level matched."""
    x, sr = sf.read(DANIEL_TAKE, dtype='float64')
    x = x.mean(1) if x.ndim > 1 else x
    env = np.sqrt(np.convolve(x ** 2, np.ones(480) / 480, 'same'))
    on = int(np.argmax(20 * np.log10(env + 1e-9) > -40))
    x = x[max(0, on - int(.02 * SR)):]
    # tone: the take is brighter and cleaner than the location audio
    x = sosfilt(butter(2, 3800, 'lowpass', fs=SR, output='sos'), x)
    x = sosfilt(butter(1, 120, 'highpass', fs=SR, output='sos'), x)
    dry = np.stack([x, x], 1)
    wet = np.stack([fftconvolve(x, small_room_ir()[:, c])[:len(x)] for c in range(2)], 1)
    y = dry + wet * .22
    # level: match the original line's speech level
    orig = prod[int(27.68 * SR):int(28.5 * SR)].mean(1)
    sp = lambda s: np.sqrt((sosfilt(butter(2, [250, 3500], 'bandpass', fs=SR, output='sos'), s) ** 2).mean())
    voiced = y.mean(1)[:int(1.0 * SR)]
    y *= sp(orig) / max(sp(voiced), 1e-9)
    return y.astype(np.float32)


def duck_curve(sig, n, max_db=5.0, thresh=-40):
    sos = butter(2, [250, 3500], 'bandpass', fs=SR, output='sos')
    sp = sosfilt(sos, sig)
    hop = 480
    e = np.sqrt(np.convolve(sp ** 2, np.ones(hop) / hop, 'same')[::hop])
    red = np.clip((20 * np.log10(e + 1e-9) - thresh) * .45, 0, max_db)
    sm = np.zeros_like(red)
    for i in range(1, len(red)):                     # fast attack, slow release
        a = .45 if red[i] > sm[i - 1] else .035
        sm[i] = sm[i - 1] + a * (red[i] - sm[i - 1])
    d = 10 ** (-np.repeat(sm, hop)[:n] / 20)
    return np.concatenate([d, np.ones(n - len(d))]).astype(np.float32)


def master_gain(pre, mas, hop=24):
    """The master's (time-varying) gain over the premix, so stems can carry it and sum to the master."""
    N = min(len(pre), len(mas))
    pre, mas = pre[:N].astype(np.float64), mas[:N].astype(np.float64)
    k = N // hop
    pp = (pre[:k * hop] ** 2).reshape(k, hop, 2).sum((1, 2))
    pm = (pre[:k * hop] * mas[:k * hop]).reshape(k, hop, 2).sum((1, 2))
    thr = np.percentile(pp[pp > 0], 5)
    g = np.where(pp > thr * 1e-3, pm / np.maximum(pp, 1e-15), np.nan)
    idx = np.arange(k)
    ok = ~np.isnan(g)
    g = np.clip(np.interp(idx, idx[ok], g[ok]), 1.0, 10.0)
    return np.interp(np.arange(N), (idx + .5) * hop, g)


def main():
    video = os.path.join(FINAL, 'joined_raw.mp4')
    if not os.path.exists(video):
        run('-y', '-f', 'concat', '-safe', '0', '-i', os.path.join(FINAL, 'list.txt'), '-c', 'copy', video)
    prod_wav = os.path.join(A, 'production.wav')
    if not os.path.exists(prod_wav):
        run('-y', '-i', video, '-vn', '-ac', '2', '-ar', str(SR), '-c:a', 'pcm_f32le', prod_wav)
    prod, _ = sf.read(prod_wav, dtype='float32')
    music, _ = sf.read(os.path.join(A, 'music.wav'), dtype='float32')
    sfx, _ = sf.read(os.path.join(A, 'sfx.wav'), dtype='float32')
    room, _ = sf.read(os.path.join(A, 'roomtone.wav'), dtype='float32')
    N = int(round(LENGTH * SR))
    pad = lambda x: np.vstack([x, np.zeros((max(0, N - len(x)), 2), np.float32)])[:N]
    prod, music, sfx, room = pad(prod), pad(music), pad(sfx), pad(room)
    vo_path = os.path.join(A, 'vo.wav')
    vo = pad(sf.read(vo_path, dtype='float32', always_2d=True)[0]) if os.path.exists(vo_path) else np.zeros((N, 2), np.float32)

    # dialogue stem
    dlg = prod * gate(N, CUTS, .06)[:, None]
    # the location's own air under removed stretches (speech-free pieces, from the transcript)
    restaurant = bed(prod, [(30.24, 31.2), (34.85, 35.85)], 8.0)
    place_bed(dlg, restaurant, 21.29, 26.16, .8)
    place_bed(dlg, restaurant[int(2 * SR):], 27.45, 29.06, .9, fade=.06)
    barair = bed(prod, [(47.62, 50.85)], 6.5)
    place_bed(dlg, barair, 41.9, 47.66, 1.0, fade=.15)
    # Daniel's corrected line
    line = daniel_line(prod)
    i = int(DANIEL_LINE_AT * SR)
    dlg[i:i + len(line)] += line[:N - i]
    # the home scene's TV chatter sits low under the voiceover's line 4
    dlg *= curve([(0, 1), (64.0, 1), (64.3, .3), (67.6, .3), (68.3, 1), (LENGTH, 1)], N)[:, None]

    # room tone where the dialogue stem is cut and no location air replaces it
    rt = curve([(0, 1), (21.2, 1), (21.35, 0), (68.85, 0), (69.0, 1), (71.7, 1), (71.9, 0), (76.05, 0), (76.2, 1),
                (LENGTH, 1)], N)
    sfx_stem = sfx + room * rt[:, None]

    # music ducks under dialogue and voiceover
    duck = np.minimum(duck_curve(dlg.mean(1), N), duck_curve(vo.mean(1), N, max_db=9.0, thresh=-42))
    music_stem = music * duck[:, None] * 10 ** (1.5 / 20)
    # a little more room for the closing "Wagon." under the end card's landing chord
    music_stem *= curve([(0, 1), (109.8, 1), (109.95, .71), (110.45, .71), (110.75, 1), (LENGTH, 1)], N)[:, None]

    mix = dlg + vo + music_stem + sfx_stem
    os.makedirs(os.path.join(A, 'stems'), exist_ok=True)
    for name, x in (('dialogue', dlg), ('music', music_stem), ('sfx', sfx_stem), ('voiceover', vo)):
        sf.write(os.path.join(A, 'stems', name + '.flac'), np.clip(x, -1, 1), SR, subtype='PCM_24')
    pre = os.path.join(A, 'mix_premaster.wav')
    sf.write(pre, mix, SR, subtype='FLOAT')

    # master: two-pass loudnorm to -14 LUFS / -1.5 dBTP
    r = subprocess.run([FF, '-hide_banner', '-i', pre, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=20:print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True)
    m = json.loads(r.stderr[r.stderr.rindex('{'):r.stderr.rindex('}') + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=20:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
          f"aresample={SR}")
    master_wav = os.path.join(A, 'mix_master.wav')
    run('-y', '-i', pre, '-af', af, '-t', f'{LENGTH}', '-c:a', 'pcm_s24le', master_wav)
    mas, _ = sf.read(master_wav, dtype='float64')
    mas = np.vstack([mas, np.zeros((max(0, N - len(mas)), 2))])[:N]
    sf.write(os.path.join(FINAL, 'Wagon_Full_mix_24bit_48k.wav'), mas, SR, subtype='PCM_24')
    sf.write(os.path.join(FINAL, 'Wagon_Full_mix_16bit_48k.wav'), mas, SR, subtype='PCM_16')

    # delivery stems with the master's gain, so Music + Dialogue + SFX + Voiceover = master
    g = master_gain(mix, mas)[:, None]
    os.makedirs(os.path.join(FINAL, 'stems'), exist_ok=True)
    total = np.zeros_like(mas)
    for name, x in (('Music', music_stem), ('Dialogue', dlg), ('SFX', sfx_stem), ('Voiceover', vo)):
        y = x.astype(np.float64) * g
        pk = np.abs(y).max()
        if pk > .97:                     # tame a stray overshoot from the gain fit, rather than clip
            y *= np.minimum(1, .97 / np.maximum(np.abs(y).max(1, keepdims=True), 1e-9))
        total += y
        tag = '' if name != 'Voiceover' or np.abs(vo).max() > 0 else '_EMPTY'
        for bits, sub in ((24, 'PCM_24'), (16, 'PCM_16')):
            sf.write(os.path.join(FINAL, 'stems', f'Wagon_Stem_{name}{tag}_{bits}bit_48k.wav'), y, SR, subtype=sub)
    db = lambda s: 20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-12)
    print('stems summed vs master:', round(db(mas - total) - db(mas), 1), 'dB')

    out = os.path.join(FINAL, 'Wagon_Full_master.mp4')
    run('-y', '-i', video, '-i', master_wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
        '-shortest', '-movflags', '+faststart', out)
    print('premaster input', m['input_i'], 'LUFS, TP', m['input_tp'])
    print('wrote', out)


if __name__ == '__main__':
    main()
