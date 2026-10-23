"""Mixes the Wagon spot and lays it onto the joined picture (parts 1-8).

Stems (48 kHz stereo):
- dialogue: the clips' own audio, edited to the client's notes (rev 2):
    0:07.6-0:26.1  removed: makeup foley, Ethan's room chatter and the date's small talk. The
                   date's background air carries on under the music from 0:21.3
    0:28.5         Daniel's own "Old Fashioned" trimmed to "Old Fashion": the over-stressed extra
                   syllable ("fash-n-ed") is cut and the word ends on his own nasal decay
    0:41.95-0:47.6 removed: the gibberish after "Hey Ethan!" (bar air carries on)
    1:08.95-1:11.75 removed: the clips' two pings (one shared notification is in the SFX)
    1:16.17-1:29.5 removed: app beats are music-only
    1:29.5-end     removed: the coffee clink, Marcus's greeting and the gibberish after it
- voiceover: final/audio/vo.wav when present (Higgsfield narrator, built by vo.py), else silent.
- music: final/audio/music.wav, ducked under dialogue, and eased down under each narrator line
  (final/audio/vo_spans.json) then back up, never below MUSIC_FLOOR_DB.
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
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
SR = 48000
LENGTH = 113.7708
FF = imageio_ffmpeg.get_ffmpeg_exe()
A = os.path.join(FINAL, 'audio')

# dialogue removals (client notes, rev 2) and the earlier ones
CUTS = [(7.6, 26.1), (28.66, 29.0), (41.95, 47.6), (68.95, 71.75), (76.17, LENGTH + 1)]
# "Old Fashioned" -> "Old Fashion": keep "fash-n" up to KEEP_TO, skip the extra vowel, and splice on the
# nasal decay that starts at TAIL_FROM (both sides are nasal murmur, so the join is seamless)
VO_MUSIC_DUCK_DB = 9.0     # music under the narrator (at most)
MUSIC_FLOOR_DB = -36.0     # ...but the music stays at least this loud (RMS) under a line
VO_DLG_DIP_DB = 9.0        # location sound under the narrator
KEEP_TO, TAIL_FROM, TAIL_LEN = 28.505, 28.59, .16   # the tail runs on past where the cut closes


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


def trim_old_fashioned(prod):
    """Daniel's own line, with the "-ed" syllable removed. Returns the edited production audio."""
    out = prod.copy()
    k = int(KEEP_TO * SR)
    xf = int(.008 * SR)
    # align the tail to the kept audio's waveform (within one pitch period) so the crossfade doesn't comb
    ref = prod[k:k + xf].mean(1)
    cands = range(int(TAIL_FROM * SR) - 240, int(TAIL_FROM * SR) + 240)
    src = max(cands, key=lambda c: np.dot(prod[c:c + xf].mean(1), ref) / (np.linalg.norm(prod[c:c + xf]) + 1e-9))
    n = int(TAIL_LEN * SR)
    tail = prod[src:src + n].copy()
    r = np.linspace(0, 1, xf)[:, None]
    out[k:k + xf] = prod[k:k + xf] * (1 - r) + tail[:xf] * r
    out[k + xf:k + n] = tail[xf:]
    return out, src / SR


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


def under_lines(spans, n, depth_db, attack=.35, release=.9):
    """Gain that eases down just before each narrator line, holds under it and rises back after.
    depth_db is one depth for every line, or a list with one per line."""
    depths = depth_db if isinstance(depth_db, (list, tuple)) else [depth_db] * len(spans)
    g = np.ones(n, np.float32)
    t = np.arange(n) / SR
    for (a, b), d in zip(spans, depths):
        r = np.clip(np.minimum((t - (a - attack)) / attack, ((b + .15 + release) - t) / release), 0, 1)
        g = np.minimum(g, 1 - (1 - 10 ** (-d / 20)) * r)
    return g


def peak_limit(x, ceiling_db=-1.0, look=.01, release=.15):
    """A lookahead peak limiter on stereo audio: touches only the peaks above the ceiling."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import filtfilt
    c = 10 ** (ceiling_db / 20)
    over = 1 - np.minimum(1, c / np.maximum(np.abs(x).max(1), 1e-9))       # gain reduction each sample needs
    over = maximum_filter1d(over, int(2 * look * SR))                       # start early, hold through the peak
    a = np.exp(-1 / (release * SR))
    over = np.maximum(over, filtfilt([1 - a], [1, -a], over))               # smooth recovery
    y = x * (1 - over)[:, None]
    return np.clip(y, -c, c)


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
    spans_path = os.path.join(A, 'vo_spans.json')
    spans = json.load(open(spans_path))['spans'] if os.path.exists(vo_path) and os.path.exists(spans_path) else []

    # dialogue stem
    edited, tail_src = trim_old_fashioned(prod)
    print(f'Old Fashion trim: tail spliced from {tail_src:.4f} s')
    dlg = edited * gate(N, CUTS, .06)[:, None]
    # the location's own air under removed stretches (speech-free pieces, from the transcript)
    restaurant = bed(prod, [(30.24, 31.2), (34.85, 35.85)], 8.0)
    place_bed(dlg, restaurant, 21.29, 26.16, .8)
    place_bed(dlg, restaurant[int(.94 * SR):], 28.55, 29.06, .9, fade=.06)   # its smoothest stretch; also covers a dropout at 28.86
    barair = bed(prod, [(47.62, 50.85)], 6.5)
    place_bed(dlg, barair, 41.9, 47.66, 1.0, fade=.15)
    dlg_novo = dlg.copy()                # for the no-narration versions
    # location sound under a narrator line (the bar crowd, the home TV) sits low beneath it
    dlg *= under_lines(spans, N, VO_DLG_DIP_DB)[:, None]

    # room tone where the dialogue stem is cut and no location air replaces it
    rt = curve([(0, 1), (21.2, 1), (21.35, 0), (68.85, 0), (69.0, 1), (71.7, 1), (71.9, 0), (76.05, 0), (76.2, 1),
                (LENGTH, 1)], N)
    sfx_stem = sfx + room * rt[:, None]

    # music ducks under the dialogue, and eases down under each whole narrator line, then back up
    # (less where the score is already quiet, so it stays audible underneath: never below MUSIC_FLOOR_DB)
    lvl = lambda a, b: 20 * np.log10(np.sqrt((music[int(a * SR):int(b * SR)] ** 2).mean()) + 1e-12) + 1.5
    depths = [float(np.clip(lvl(a, b) - MUSIC_FLOOR_DB, 3.0, VO_MUSIC_DUCK_DB)) for a, b in spans]
    duck = np.minimum(duck_curve(dlg.mean(1), N), under_lines(spans, N, depths))
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

    # no-narration versions, for auditioning narrators: the score with its full arc and its dips under the
    # on-screen dialogue, but none under the narrator, at the same level as in the commercial (master's gain)
    music_novo = music * duck_curve(dlg_novo.mean(1), N)[:, None] * 10 ** (1.5 / 20) * g
    mix_novo = music_novo + (dlg_novo + sfx_stem) * g
    nv = os.path.join(FINAL, 'no_vo')
    os.makedirs(nv, exist_ok=True)
    for name, x in (('Wagon_Music_Bed_NoVO', music_novo), ('Wagon_Mix_NoVO', mix_novo)):
        x = peak_limit(x)                # without the narrator dips the montage peaks over; only those are held
        for bits, sub in ((24, 'PCM_24'), (16, 'PCM_16')):
            sf.write(os.path.join(nv, f'{name}_{bits}bit_48k.wav'), x, SR, subtype=sub)
    print('no-VO music bed peak', round(20 * np.log10(np.abs(music_novo).max()), 1), 'dBFS')

    out = os.path.join(FINAL, 'Wagon_Full_master.mp4')
    picture = os.path.join(FINAL, 'picture.mp4')        # the cut with Shot 20 made legible (tools/shot20.py)
    run('-y', '-i', picture if os.path.exists(picture) else video, '-i', master_wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
        '-shortest', '-movflags', '+faststart', out)
    print('premaster input', m['input_i'], 'LUFS, TP', m['input_tp'])
    print('wrote', out)


if __name__ == '__main__':
    main()
