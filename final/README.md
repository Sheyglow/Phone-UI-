# Wagon full cut with sound

Parts 1–8 joined in order, with a score, sound design and the clips' own dialogue mixed and mastered.

| File | What it is |
| --- | --- |
| `Wagon_Full_1080p.mp4` | Delivery: 1920×1080, 24 fps, 1:53.8, AAC 256 kbps, −14 LUFS / −1.5 dBTP |
| `Wagon_Full_share_720p.mp4` | Smaller copy for sending |
| `audio/stems/dialogue.flac`, `music.flac`, `sfx.flac` | Mix stems (48 kHz, 24-bit) for anyone re-balancing |

The full-bitrate master (`Wagon_Full_master.mp4`, the joined picture untouched with the master audio) is about 230 MB, so it's rebuilt locally rather than committed.

## The shape of the track

| Time | Picture | Sound |
| --- | --- | --- |
| 0:00 | Jessica on the dating app | Music starts on shot 1. Light, hopeful piano arpeggios in A♭ with harp glints. Taps, a swipe, a key tick per letter and the send |
| 0:07.7 | Jessica on the phone, then Ethan | A strings pad enters under the dialogue, and violins join over Ethan |
| 0:21.3 | The restaurant date | Sparser piano, sitting under the dialogue |
| 0:35.9 | Jessica's uncomfortable reaction | **The music dips** to a held E♭sus2 with two soft piano notes |
| 0:40.3 | Ethan at the sports bar | Quiet and reflective under the bar noise: single piano notes over low strings |
| 0:61.3 | Home, both alone | Solo piano. **One shared notification** (a soft two-note chime) at 1:09.15, which both pick up their phones to. The clips' two separate pings are removed. From 1:10.9 it builds (strings swell, timpani roll, cymbal swell) |
| 1:16.2 | **Wagon reveal** and app moments | **The lift**, music-only: piano 8ths, spiccato strings, low drum pulse, harp run, glock sparkles on the logo and the match, quiet taps only |
| 1:29.5 | Payoffs | Builds warm and full with the violin theme and timpani, under the dialogue, then opens up fully on the montage (1:41.6), with cymbal swells into the cuts |
| 1:49.4 | End card | The landing: a full A♭ chord with timpani, cymbal, harp run and a glock on the logo pop, a last piano phrase, then it rings out |

The clips carry faint music beds of their own in A♭ / E♭ / F minor. The score is written in A♭ major so it sits with them rather than against them.

## Sources and licences

- **Music:** composed for this cut and performed with the *VS Chamber Orchestra: Community Edition* samples (Versilian Studios, **CC0**, public domain): upright piano, string sections, contrabass, harp, glockenspiel, timpani and orchestral percussion. No credit or licence fee is required.
- **Sound design:** synthesised here (taps, key ticks, swipes, the notification chime, transitions, room tone).
- **Dialogue and ambience:** the clips' own audio.
- **No voiceover yet.** None was supplied. The mix leaves room for one, and the dialogue/music/sfx stems make adding it straightforward.

## Rebuilding

```sh
# instrument samples (CC0): https://github.com/sgossner/VSCO-2-CE
VSCO=/path/to/VSCO-2-CE python3 final/tools/score.py   # -> final/audio/music.wav
python3 final/tools/sfx.py                              # -> final/audio/sfx.wav, roomtone.wav
python3 final/tools/mix.py                              # joins parts/part1-8.mp4, mixes, masters -> Wagon_Full_master.mp4
```
