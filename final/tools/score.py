"""Scores the Wagon spot to picture from the VS Chamber Orchestra CE samples (CC0).

The cue follows the edit (times are seconds in the joined cut):
  0.00  Jessica on the dating app: light, hopeful piano arpeggios in A♭ (the clips' own
        faint beds are A♭/E♭/F minor, so the score sits in the same key), harp glints.
  7.71  strings pad enters; 13.25 violins join over Ethan.
 21.60  restaurant: sparser piano under the dialogue.
 35.88  Jessica's uncomfortable reaction: the music dips to a held E♭sus2.
 40.33  sports bar: quiet and reflective, single piano notes over low strings.
 61.33  home: solo piano, lonely; from 70.9 it builds (strings swell, timpani roll,
        cymbal swell) into the Wagon reveal.
 76.17  Wagon reveal and app moments (music-only): the lift, with piano 8ths, spiccato strings,
        a low drum pulse and glock sparkles on the logo and the match.
 89.50  payoffs: builds to warm and full, with the violin theme, timpani and cymbal swells into cuts.
109.42  end card: the big A♭ landing (timpani, cymbal, full strings, harp, glock on the logo pop),
        a last piano phrase, then it rings out.

Usage: VSCO=/path/to/VSCO-2-CE python3 final/tools/score.py -> final/audio/music.wav (48 kHz stereo)
"""
import glob
import os
import re

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
VSCO = os.environ.get('VSCO', '/tmp/claude-0/-home-user-Phone-UI-/067023f8-8c94-50e9-b9fa-331e8dde2295/scratchpad/vsco')
SR = 48000
LENGTH = 113.77
NAMES = 'C C# D D# E F F# G G# A A# B'.split()


def midi(name):
    m = re.match(r'([A-G]#?)(-?\d)', name)
    return NAMES.index(m.group(1)) + 12 * (int(m.group(2)) + 1)


def n(s):
    """'Ab3' / 'Eb4' / 'C5' -> MIDI."""
    flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    m = re.match(r'([A-G]b?#?)(-?\d)', s)
    name = flats.get(m.group(1), m.group(1))
    return NAMES.index(name) + 12 * (int(m.group(2)) + 1)


# ------------------------------------------------------------------ sampler
_cache = {}


def load(path):
    if path not in _cache:
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, 1)
        x = x[:, :2]
        if sr != SR:
            x = resample_poly(x, SR, sr, axis=0).astype(np.float32)
        # trim to the onset (some samples carry 0.2 s of lead-in)
        env = np.abs(x).max(1)
        on = int(np.argmax(env > env.max() * .03))
        x = x[max(0, on - int(.004 * SR)):]
        x = x / max(1e-6, np.abs(x).max())   # layers keep their timbre; velocity sets the level
        _cache[path] = x
    return _cache[path]


class Instrument:
    def __init__(self, pattern, regex, octave=0, layers=None, pan=0.0, attack=.004, release=.25):
        self.samples = []  # (midi, layer, path)
        for p in glob.glob(os.path.join(VSCO, pattern)):
            m = re.search(regex, os.path.basename(p))
            if not m:
                continue
            note = midi(m.group('note')) + octave
            layer = m.groupdict().get('layer') or ''
            rank = (layers or {}).get(layer, 0)
            self.samples.append((note, rank, p))
        assert self.samples, pattern
        self.pan, self.attack, self.release = pan, attack, release

    def pick(self, note, vel):
        ranks = sorted(set(r for _, r, _ in self.samples))
        want = ranks[min(len(ranks) - 1, int(vel * len(ranks)))]
        cands = [s for s in self.samples if s[1] == want] or self.samples
        best = min(cands, key=lambda s: (abs(s[0] - note), s[0] < note, s[2]))
        return best

    def render(self, note, vel, dur, attack=None):
        src_note, _, path = self.pick(note, vel)
        x = load(path)
        ratio = 2 ** ((note - src_note) / 12)
        need = int((dur + self.release) * SR * ratio) + 16
        x = x[:need]
        if abs(ratio - 1) > 1e-4:
            idx = np.arange(0, len(x) - 1, ratio)
            x = np.stack([np.interp(idx, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype(np.float32)
        L = len(x)
        env = np.ones(L, np.float32)
        a = int((self.attack if attack is None else attack) * SR)
        if a > 0:
            env[:min(a, L)] = np.linspace(0, 1, min(a, L)) ** 1.5
        d = int(dur * SR)
        r = int(self.release * SR)
        if d < L:
            tail = np.linspace(1, 0, max(1, min(r, L - d))) ** 2
            env[d:d + len(tail)] *= tail
            env[d + len(tail):] = 0
        x = x * env[:, None] * (vel ** 1.6)
        p = (self.pan + 1) / 2
        x[:, 0] *= np.sqrt(1 - p) * 1.414
        x[:, 1] *= np.sqrt(p) * 1.414
        return x


INST = {}


def build_instruments():
    INST['piano'] = Instrument('Keys/Upright Nr1/UR1_*.wav', r'UR1_(?P<note>[A-G]#?\d)_(?P<layer>pp|mf|f)_RR',
                               layers={'pp': 0, 'mf': 1, 'f': 2}, pan=-.05, release=.6)
    INST['vln'] = Instrument('Strings/Violin Section/susVib/*.wav', r'_(?P<note>[A-G]#?\d)_(?P<layer>v\d)',
                             octave=12, layers={'v1': 0, 'v2': 1}, pan=-.35, attack=.35, release=1.2)
    INST['vla'] = Instrument('Strings/Viola Section/susvib/*.wav', r'_(?P<note>[A-G]#?\d)_(?P<layer>v\d)',
                             octave=12, layers={'v1': 0, 'v2': 1}, pan=.2, attack=.4, release=1.2)
    INST['vc'] = Instrument('Strings/Cello Section/susvib/*.wav', r'_(?P<note>[A-G]#?\d)_(?P<layer>v\d)',
                            octave=12, layers={'v1': 0, 'v3': 1}, pan=.35, attack=.4, release=1.3)
    INST['cb'] = Instrument('Strings/Solo Contrabass/SusNV/*.wav', r'_(?P<note>[A-G]#?\d)_(?P<layer>v\d)',
                            octave=12, layers={'v1': 0, 'v3': 1}, pan=.15, attack=.3, release=1.0)
    INST['spic'] = Instrument('Strings/Violin Section/Spic/*.wav', r'_(?P<note>[A-G]#?\d)_(?P<layer>v\d)',
                              octave=12, layers={'v1': 0, 'v2': 1}, pan=-.3, release=.15)
    INST['vcspic'] = Instrument('Strings/Cello Section/spic/*.wav', r'_(?P<note>[A-G]#?\d)_(?P<layer>v\d)',
                                octave=12, layers={'v1': 0, 'v3': 1}, pan=.3, release=.15)
    INST['harp'] = Instrument('Strings/Harp/KSHarp_*.wav', r'KSHarp_(?P<note>[A-G]#?\d)_(?P<layer>\w+)\.wav',
                              layers={'mp': 0, 'mf': 0, 'f': 1}, pan=.25, release=1.5)
    INST['glock'] = Instrument('Percussion/Glock/*.wav', r'_(?P<note>[A-G]#?\d)\.wav', octave=12, pan=-.15, release=2.0)


def one_shot(path, gain=1.0, semis=0.0, pan=0.0):
    x = load(os.path.join(VSCO, path)).copy()
    if semis:
        ratio = 2 ** (semis / 12)
        idx = np.arange(0, len(x) - 1, ratio)
        x = np.stack([np.interp(idx, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype(np.float32)
    p = (pan + 1) / 2
    x[:, 0] *= np.sqrt(1 - p) * 1.414
    x[:, 1] *= np.sqrt(p) * 1.414
    return x * gain


# ------------------------------------------------------------------ score
class Score:
    def __init__(self):
        self.buses = {b: np.zeros((int((LENGTH + 12) * SR), 2), np.float32) for b in ('piano', 'strings', 'perc', 'fx')}

    def put(self, bus, t, x):
        i = int(round(t * SR))
        if i < 0:
            x, i = x[-i:], 0
        j = min(i + len(x), len(self.buses[bus]))
        self.buses[bus][i:j] += x[:j - i]

    def note(self, inst, bus, t, pitch, vel, dur, attack=None):
        self.put(bus, t, INST[inst].render(n(pitch) if isinstance(pitch, str) else pitch, vel, dur, attack))

    def chord(self, inst, bus, t, pitches, vel, dur, attack=None):
        for p in pitches:
            self.note(inst, bus, t, p, vel, dur, attack)


# chord voicings in A♭ major: bass, low strings, upper strings, piano arpeggio tones
CH = {
    'Ab':   dict(bass='Ab2', low=['Ab3', 'Eb4'], high=['C5', 'Eb5'], arp=['Ab3', 'Eb4', 'Ab4', 'C5']),
    'Eb/G': dict(bass='G2', low=['G3', 'Eb4'], high=['Bb4', 'Eb5'], arp=['G3', 'Eb4', 'G4', 'Bb4']),
    'Fm':   dict(bass='F2', low=['F3', 'C4'], high=['Ab4', 'C5'], arp=['F3', 'C4', 'F4', 'Ab4']),
    'Db':   dict(bass='Db2', low=['Ab3', 'F4'], high=['Ab4', 'Db5'], arp=['Db4', 'Ab4', 'Db5', 'F5']),
    'Ab/C': dict(bass='C3', low=['Ab3', 'Eb4'], high=['Ab4', 'C5'], arp=['C4', 'Eb4', 'Ab4', 'C5']),
    'Eb':   dict(bass='Eb2', low=['G3', 'Eb4'], high=['Bb4', 'Eb5'], arp=['Eb4', 'G4', 'Bb4', 'Eb5']),
    'Ebsus': dict(bass='Eb2', low=['Ab3', 'Eb4'], high=['Bb4', 'Eb5'], arp=['Eb4', 'Ab4', 'Bb4', 'Eb5']),
    'Ebsus2': dict(bass='Eb2', low=['Bb3', 'Eb4'], high=['F4', 'Bb4'], arp=['Eb4', 'F4', 'Bb4', 'Eb5']),
}
ARP8 = [0, 1, 2, 3, 2, 1, 2, 1]   # 8th-note pattern over the arp tones (with the top note doubled up an octave)


def arp_bar(S, t, bar, chord, vel, eighths=8, oct_up=False, skip_first=False, vel_shape=None):
    tones = CH[chord]['arp']
    step = bar / eighths
    for k in range(eighths):
        if skip_first and k == 0:
            continue
        p = n(tones[ARP8[k % 8]]) + (12 if oct_up else 0)
        v = vel * (1.0 if k % 4 == 0 else .82) * (vel_shape(k) if vel_shape else 1)
        S.note('piano', 'piano', t + k * step, p, v, step * 2.2)


def pad(S, t, chord, dur, vel, layers=('vla', 'vc'), attack=.9):
    c = CH[chord]
    if 'vc' in layers:
        S.note('vc', 'strings', t, c['low'][0], vel, dur, attack)
    if 'vla' in layers:
        S.note('vla', 'strings', t, c['low'][1], vel * .9, dur, attack)
    if 'vln' in layers:
        for p in c['high']:
            S.note('vln', 'strings', t, p, vel * .85, dur, attack)
    if 'cb' in layers:
        S.note('cb', 'strings', t, c['bass'], vel, dur, attack)


def compose(S):
    # A: 0 - 21.6, light and hopeful (100 bpm, 2.4 s bars)
    prog = ['Ab', 'Eb/G', 'Fm', 'Db'] * 2 + ['Ab']
    for b, ch in enumerate(prog[:9]):
        t = b * 2.4
        if t >= 21.6:
            break
        vel = .32 if t < 7.7 else .38
        arp_bar(S, t, 2.4, ch, vel)
        S.note('piano', 'piano', t, n(CH[ch]['bass']) - 12 + 12, .30, 2.3)   # soft left hand
        if t >= 7.7:
            pad(S, t, ch, 2.5, .34, layers=('vla', 'vc'), attack=1.2 if t < 8 else .6)
        if t >= 13.25 - .2:
            pad(S, t, ch, 2.5, .28, layers=('vln',), attack=1.0)
    for t, p in [(0.02, 'Eb6'), (2.4, 'C6'), (4.8, 'Ab5'), (7.71, 'Eb6'), (9.6, 'C6'), (13.25, 'Ab6')]:
        S.note('harp', 'piano', t, p, .4, 3)

    # B: 21.6 - 35.88, restaurant: sparser under dialogue
    prog = ['Fm', 'Db', 'Ab/C', 'Eb', 'Fm', 'Db']
    for b, ch in enumerate(prog):
        t = 21.6 + b * 2.4
        tones = CH[ch]['arp']
        for k, idx in enumerate([0, 2, 1, 3]):
            S.note('piano', 'piano', t + k * .6, tones[idx], .26, 1.4)
        pad(S, t, ch, 2.5, .3, layers=('vla', 'vc'), attack=.7)

    # C: 35.88 - 40.33, the dip at Jessica's reaction: held E♭sus2, two soft piano notes
    pad(S, 35.88, 'Ebsus2', 4.5, .26, layers=('vla', 'vc', 'vln'), attack=1.4)
    S.note('piano', 'piano', 36.6, 'F5', .22, 2)
    S.note('piano', 'piano', 38.4, 'Bb4', .2, 2)

    # D: 40.33 - 61.33, sports bar: quiet, reflective
    prog = ['Fm', 'Db', 'Ab', 'Eb'] * 3
    mel = ['C5', 'Ab4', 'Eb5', 'Bb4', 'C5', 'Db5', 'C5', 'Bb4', 'Ab4', 'F4', 'Eb4', 'G4']
    for b, ch in enumerate(prog):
        t = 40.33 + b * 2.4
        if t >= 61.0:
            break
        pad(S, t, ch, 2.5, .26, layers=('vc', 'cb'), attack=.8)
        S.note('piano', 'piano', t, mel[b], .24, 2.2)
        S.note('piano', 'piano', t + 1.2, CH[ch]['arp'][1], .16, 1.2)

    # E: 61.33 - 76.17, home: lonely piano, then the build into the reveal
    bar = (76.17 - 61.33) / 6
    prog = ['Fm', 'Db', 'Ab/C', 'Eb', 'Db', 'Ebsus']
    mel = [('C5', 'Ab4'), ('F5', 'Db5'), ('Eb5', 'C5'), ('Bb4', 'G4'), ('Ab4', 'F4'), ('Bb4', 'Eb5')]
    for b, ch in enumerate(prog):
        t = 61.33 + b * bar
        m1, m2 = mel[b]
        S.note('piano', 'piano', t, m1, .28, bar * .55)
        S.note('piano', 'piano', t + bar * .5, m2, .24, bar * .5)
        S.note('piano', 'piano', t, n(CH[ch]['bass']) + 12, .22, bar)
        if b < 4:
            pad(S, t, ch, bar + .1, .2, layers=('vc',), attack=1.0)
        else:
            # the build: strings swell bar by bar, piano 8ths return
            pad(S, t, ch, bar + .1, .34 + .14 * (b - 4), layers=('vla', 'vc', 'vln', 'cb'), attack=bar * .8)
            arp_bar(S, t, bar, ch, .26 + .1 * (b - 4), vel_shape=lambda k: .7 + .3 * k / 7)
    S.put('perc', 76.17 - 7.18, one_shot('Percussion/susCymb1-cresc-Long_v1.wav', .55))      # peak lands on the reveal
    roll = one_shot('Percussion/Timpani/Rolls/Timpani1_Roll_v5_rr1_Sum.wav', .5, semis=2)    # ≈ A♭2
    rl = roll[:int(2.6 * SR)] * np.linspace(.15, 1, int(2.6 * SR))[:, None] ** 2
    S.put('perc', 76.17 - 2.6, rl)

    # F: 76.17 - 89.5, Wagon reveal + app moments (music-only): the lift (90 bpm, 2.667 s bars)
    bar = 2.6667
    prog = ['Db', 'Ab/C', 'Eb', 'Fm', 'Db']
    for b, ch in enumerate(prog):
        t = 76.17 + b * bar
        arp_bar(S, t, bar, ch, .5)
        pad(S, t, ch, bar + .15, .5, layers=('vla', 'vc', 'vln', 'cb'), attack=.25 if b == 0 else .5)
        for k in range(8):   # spiccato ostinato on the chord's top tone
            S.note('spic', 'strings', t + k * bar / 8, CH[ch]['high'][0], .5 if k % 2 == 0 else .38, bar / 8)
        for beat in (0, 2):
            S.put('perc', t + beat * bar / 4, one_shot('Percussion/BDrumNewhit_v5_rr1_Sum.wav', .42 if beat == 0 else .28))
    S.put('perc', 76.17, one_shot('Percussion/cymbal-crash1_mp_rr1.wav', .28, pan=.2))
    for t, ps in [(76.6, ['Ab5', 'Eb6', 'Ab6']), (83.38, ['Eb6', 'Ab6', 'C7'])]:   # logo ring, the match
        for k, p in enumerate(ps):
            S.note('glock', 'piano', t + k * .09, p, .45, 2)
    S.note('harp', 'piano', 76.17, 'Ab4', .45, 2)
    for k, p in enumerate(['Ab4', 'C5', 'Eb5', 'Ab5', 'C6', 'Eb6']):  # harp lift on the reveal
        S.note('harp', 'piano', 76.17 + k * .06, p, .42, 2.5)

    # G: 89.5 - 109.42, payoffs: build to warm and full
    prog = ['Ab', 'Eb/G', 'Fm', 'Db', 'Ab', 'Eb/G', 'Db']
    theme = ['C5', 'Bb4', 'C5', 'Db5', 'Eb5', 'G5', 'F5']
    for b, ch in enumerate(prog):
        t = 89.5 + b * bar
        lift = b / 6
        arp_bar(S, t, bar, ch, .5 + .12 * lift, oct_up=b >= 4)
        pad(S, t, ch, bar + .2, .5 + .2 * lift, layers=('vla', 'vc', 'vln', 'cb'), attack=.4)
        S.note('vln', 'strings', t, theme[b], .62 + .2 * lift, bar * .95, attack=.3)
        S.note('vln', 'strings', t, n(theme[b]) - 12, .45 + .15 * lift, bar * .95, attack=.3)
        for k in range(8):
            S.note('spic', 'strings', t + k * bar / 8, CH[ch]['high'][0], (.5 if k % 2 == 0 else .38) + .1 * lift, bar / 8)
            if b >= 3:
                S.note('vcspic', 'strings', t + k * bar / 8, n(CH[ch]['bass']) + 12, .45 if k % 2 == 0 else .33, bar / 8)
        for beat in (0, 2):
            S.put('perc', t + beat * bar / 4, one_shot('Percussion/BDrumNewhit_v6_rr1_Sum.wav', .5 if beat == 0 else .32))
        if b >= 3:
            S.put('perc', t, one_shot('Percussion/Timpani/Timpani3_Hit_v3_rr1_Sum.wav', .5 + .2 * lift))
    # cymbal swells into the cuts at 93.71 and 101.62
    S.put('perc', 93.71 - 1.52, one_shot('Percussion/susCymb1-cresc-Short_v1.wav', .32))
    S.put('perc', 101.62 - 3.55, one_shot('Percussion/susCymb1-cresc-Median_v1.wav', .4))
    # the last stretch: E♭sus -> E♭ with a timpani roll into the end card
    t0 = 89.5 + 7 * bar   # 108.17
    pad(S, t0, 'Ebsus', 109.42 - t0 + .1, .8, layers=('vla', 'vc', 'vln', 'cb'), attack=.2)
    S.note('vln', 'strings', t0, 'Ab5', .8, .7, attack=.15)
    S.note('vln', 'strings', t0 + .7, 'G5', .85, 109.42 - t0 - .7 + .1, attack=.1)
    for k in range(int((109.42 - t0) / (bar / 8))):
        S.note('spic', 'strings', t0 + k * bar / 8, 'Bb4', .55 + .03 * k, bar / 8)
    S.put('perc', 109.42 - 7.18, one_shot('Percussion/susCymb1-cresc-Long_v1.wav', .6))
    roll = one_shot('Percussion/Timpani/Rolls/Timpani1_Roll_v5_rr1_Sum.wav', .55, semis=-3)   # ≈ E♭2
    rl = roll[:int(1.6 * SR)] * np.linspace(.2, 1, int(1.6 * SR))[:, None] ** 2
    S.put('perc', 109.42 - 1.6, rl)

    # H: 109.42 - end, the landing on the end card
    t = 109.42
    S.put('perc', t, one_shot('Percussion/Timpani/Timpani3_Hit_v3_rr1_Sum.wav', .9))
    S.put('perc', t, one_shot('Percussion/BDrumNewhit_v7_rr1_Sum.wav', .7))
    S.put('perc', t, one_shot('Percussion/cymbal-crash1_mp_rr1.wav', .45, pan=-.1))
    pad(S, t, 'Ab', 5.0, .78, layers=('vla', 'vc', 'vln', 'cb'), attack=.08)
    S.note('vln', 'strings', t, 'Ab5', .75, 5.0, attack=.08)
    S.note('vc', 'strings', t, 'Ab2', .7, 5.0, attack=.08)
    S.chord('piano', 'piano', t, ['Ab2', 'Ab3', 'Eb4', 'Ab4', 'C5', 'Eb5'], .6, 4.5)
    for k, p in enumerate(['Ab4', 'C5', 'Eb5', 'Ab5', 'C6', 'Eb6', 'Ab6']):
        S.note('harp', 'piano', t + k * .05, p, .5, 3.5)
    for k, p in enumerate(['Eb6', 'Ab6', 'C7']):   # the logo pop
        S.note('glock', 'piano', 109.71 + k * .08, p, .5, 2.5)
    for k, p in enumerate(['Eb5', 'C5', 'Ab4', 'Bb4', 'C5']):   # last phrase
        S.note('piano', 'piano', 110.9 + k * .45, p, .3 - .02 * k, 1.6)


# ------------------------------------------------------------------ reverb + mix
def reverb_ir(rt60=2.6, pre=.018, seed=1):
    rng = np.random.default_rng(seed)
    L = int(rt60 * SR)
    t = np.arange(L) / SR
    env = np.exp(-6.91 * t / rt60)
    ir = rng.standard_normal((L, 2)).astype(np.float32) * env[:, None]
    # darker tail: one-pole low-pass that closes over time
    out = np.zeros_like(ir)
    a = 0.0
    y = np.zeros(2, np.float32)
    for i in range(0, L, 256):
        a = .15 + .7 * min(1, i / L)
        blk = ir[i:i + 256]
        for j in range(len(blk)):
            y = (1 - a) * blk[j] + a * y
            out[i + j] = y
    out = np.concatenate([np.zeros((int(pre * SR), 2), np.float32), out])
    return out / np.sqrt((out ** 2).sum(0, keepdims=True))


def level_curve(points, length):
    """dB automation from (time, dB) points, linear in dB between them."""
    t = np.arange(length) / SR
    ts, db = zip(*points)
    return (10 ** (np.interp(t, ts, db) / 20)).astype(np.float32)


def main():
    build_instruments()
    S = Score()
    compose(S)
    ir = reverb_ir()
    sends = {'piano': .32, 'strings': .45, 'perc': .3, 'fx': .4}
    gains = {'piano': .9, 'strings': .8, 'perc': .9, 'fx': 1.0}
    dry = sum(S.buses[b] * gains[b] for b in S.buses)
    wet_in = sum(S.buses[b] * gains[b] * sends[b] for b in S.buses)
    wet = np.stack([fftconvolve(wet_in[:, c], ir[:, c])[:len(dry)] for c in range(2)], 1)
    mix = dry + wet * .55
    N = int(LENGTH * SR)
    mix = mix[:N]
    # the shape of the cue: each section is measured and set to a target level (dB RMS, relative),
    # with ramps at the joins. Light and hopeful, dip at the reaction, reflective through the
    # restaurant and bar, build at home, lift at the reveal, fullest over the payoffs and end card.
    sections = [  # (start, end, target dB, ramp-in seconds)
        # client notes (rev 2): normal level through 0:26 (getting ready, Ethan's text, the date's small
        # talk); it only lowers once the waitress speaks. After "Hey Ethan!" it comes back up over the
        # removed chatter. The payoffs no longer carry clip dialogue, so they stay at full level.
        (0.0, 7.71, -21, 0.0), (7.71, 13.25, -21, 0.0), (13.25, 21.3, -21, 0.0), (21.3, 26.0, -21, 0.0),
        (26.0, 35.88, -27, .25), (35.88, 40.33, -30, .8), (40.33, 41.95, -28, 1.0), (41.95, 47.6, -22, .5),
        (47.6, 61.33, -28, .6), (61.33, 70.9, -27, 1.0), (70.9, 76.17, -21, 2.0), (76.17, 89.5, -16, .05),
        (89.5, 101.62, -16, 1.0), (101.62, 109.42, -13, 1.2), (109.42, LENGTH, -11, .05)]
    mono = mix.mean(1)
    pts = []
    for s0, s1, target, ramp in sections:
        seg = mono[int(s0 * SR):int(s1 * SR)]
        g = target - 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
        pts += [(s0 + (ramp if ramp else 0), g), (s1, g)]
    pts[0] = (0.0, pts[0][1])
    mix *= level_curve(pts, N)[:, None]
    fade = np.ones(N, np.float32)
    k = int(1.8 * SR)
    fade[-k:] = np.linspace(1, 0, k) ** 2
    mix *= fade[:, None]
    mix *= .5 / max(1e-6, np.abs(mix).max())
    os.makedirs(os.path.join(FINAL, 'audio'), exist_ok=True)
    sf.write(os.path.join(FINAL, 'audio', 'music.wav'), mix, SR, subtype='FLOAT')
    print('wrote final/audio/music.wav', mix.shape)


if __name__ == '__main__':
    main()
