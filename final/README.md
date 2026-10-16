# Wagon full cut with sound

Parts 1–8 joined in order, with a score, sound design, the clips' own dialogue and the narrator's voiceover mixed and mastered. This is revision 2: the client's audio notes and the voiceover are in.

| File | What it is |
| --- | --- |
| `Wagon_Full_1080p.mp4` | Delivery: 1920×1080, 24 fps, 1:53.8, AAC 256 kbps, −14 LUFS / −1.5 dBTP |
| `Wagon_Full_share_720p.mp4` | Smaller copy for sending |
| `Wagon_Full_mix_24bit_48k.wav`, `Wagon_Full_mix_16bit_48k.wav` | The master mix as WAV: 48 kHz stereo, −14 LUFS, exactly the picture's length (1:53.77), starting on frame 1 |
| `stems/Wagon_Stem_*_24bit_48k.wav` | Delivery stems: Music, Dialogue, SFX, Voiceover. They sum to the master (see `stems/README.md`) |
| `audio/stems/*.flac` | The same four stems before mastering (48 kHz, 24-bit), for anyone re-balancing |

The full-bitrate master (`Wagon_Full_master.mp4`, the joined picture untouched with the master audio) is about 230 MB, so it's rebuilt locally rather than committed.

## Voiceover

Narrator: **Higgsfield**, voice **Nora** (female), Seed Audio 1.0, on a paid (Pro) plan, which covers commercial use. The brief asks for a warm, natural, emotionally intelligent and cinematic voice that speaks to the viewer rather than selling: not soft, breathy, dramatic or announcer-like. Each line was generated on its own. Lines 1–4 are read a little slower, calm and reflective, with a touch of low-mid warmth. Lines 5–10, after the Wagon reveal, are at normal pace with a touch of presence, so they sound brighter and more hopeful (`tools/vo.py`).

`audio/vo_female/` holds all 10 lines in each candidate voice (Nora, Quinn, Lucy, plus Naomi, whose takes came out unreliable). `python3 final/tools/vo.py quinn` (or `lucy`) swaps the narrator.

Each line starts on the client's time:

| Start | Line |
| --- | --- |
| 0:01.2 | Everyone wants meaningful connection. |
| 0:36 | But in recovery, finding it can feel different. |
| 1:00 | The places that once felt familiar don't always feel the same anymore. |
| 1:10 | But that doesn't mean connection in recovery doesn't exist. |
| 1:17 | That's why there's Wagon. |
| 1:20 | A dedicated place for love, dating, friendship and connection in recovery. |
| 1:30 | No stigma. No social pressure. |
| 1:34 | Just people who get it. |
| 1:41 | Whatever connection means to you, find people who understand the journey. |
| 1:50 | Wagon. |

The music plays under the whole spot. It eases down just before each line (up to 9 dB), holds under it and rises back after. Where the score is already quiet it dips less, so it never drops below about −36 dB RMS and stays audible. Location sound under a line (the bar crowd at 1:00, the home TV at 1:00–1:07 and 1:10–1:14) dips with it.

## Revision 2 (client's audio notes)

| Time | Change |
| --- | --- |
| whole spot | The voiceover is in, on the client's timings, with the music lowering under every line and rising back after |
| 0:08–0:13 | Music at normal level; the makeup clatter and small foley removed |
| 0:13–0:21 | Music at normal level; Ethan's phone buzz (0:15.6) and the text-received sound (0:17.0) kept audible over it |
| 0:21–0:25 | The small talk and laughter removed; music carries on and sits under once the waitress speaks (0:26.2) |
| 0:28.5 | Daniel now says "I'll do an Old Fashion." in his own voice: the over-stressed extra syllable of "Old Fashioned" ("fash-n-**ed**") is cut out and the word ends on his own nasal decay, with the restaurant's air under the gap |
| 0:42–0:47.6 | The gibberish after "Hey Ethan!" removed; the bar's air carries on and the music comes back up |
| 1:29.5–end | The coffee clink, Marcus's greeting and the gibberish after it removed; music carries it between the voiceover lines |

## The shape of the track

| Time | Picture | Sound |
| --- | --- | --- |
| 0:00 | Jessica on the dating app | Music starts on shot 1. Light, hopeful piano arpeggios in A♭ with harp glints. Taps, a swipe, a key tick per letter and the send |
| 0:07.7 | Jessica on the phone, then Ethan | A strings pad enters under the dialogue, and violins join over Ethan |
| 0:21.3 | The restaurant date | Sparser piano at full level, settling under the dialogue when the waitress arrives |
| 0:35.9 | Jessica's uncomfortable reaction | **The music dips** to a held E♭sus2 with two soft piano notes |
| 0:40.3 | Ethan at the sports bar | Quiet and reflective under the bar noise: single piano notes over low strings |
| 1:01.3 | Home, both alone | Solo piano. **One shared notification** (a soft two-note chime) at 1:09.15, which both pick up their phones to. The clips' two separate pings are removed. From 1:10.9 it builds (strings swell, timpani roll, cymbal swell) |
| 1:16.2 | **Wagon reveal** and app moments | **The lift**, music-only: piano 8ths, spiccato strings, low drum pulse, harp run, glock sparkles on the logo and the match, quiet taps only |
| 1:29.5 | Payoffs | Builds warm and full with the violin theme and timpani, music-only apart from the voiceover, then opens up fully on the montage (1:41.6), with cymbal swells into the cuts |
| 1:49.4 | End card | The landing: a full A♭ chord with timpani, cymbal, harp run and a glock on the logo pop, a last piano phrase, then it rings out |

The clips carry faint music beds of their own in A♭ / E♭ / F minor. The score is written in A♭ major so it sits with them rather than against them.

## Sources and licences

- **Music:** composed for this cut and performed with the *VS Chamber Orchestra: Community Edition* samples (Versilian Studios, **CC0**, public domain): upright piano, string sections, contrabass, harp, glockenspiel, timpani and orchestral percussion. No credit or licence fee is required.
- **Sound design:** synthesised here (taps, key ticks, swipes, the notification chime, transitions, room tone).
- **Dialogue and ambience:** the clips' own audio.
- **Voiceover:** generated with Higgsfield on a paid plan (see Voiceover above). No ElevenLabs audio is used anywhere in the mix.

## Rebuilding

```sh
# instrument samples (CC0): https://github.com/sgossner/VSCO-2-CE
VSCO=/path/to/VSCO-2-CE python3 final/tools/score.py   # -> final/audio/music.wav
python3 final/tools/sfx.py                              # -> final/audio/sfx.wav, roomtone.wav
python3 final/tools/vo.py [nora|quinn|lucy]             # -> final/audio/vo.wav, vo_spans.json (from audio/vo_female/<voice>/)
python3 final/tools/mix.py                              # joins parts/part1-8.mp4, mixes, masters -> Wagon_Full_master.mp4, WAVs, stems
```
