# Wagon stems

Four stems, one per track: 48 kHz stereo WAV, each exactly 1:53.77 and starting on frame 1. Line them all up at 0 on the timeline.

| Stem | Contents |
| --- | --- |
| `Wagon_Stem_Music_*` | The score only |
| `Wagon_Stem_Dialogue_*` | The clips' own production audio: dialogue plus the room and bar ambience recorded with it, which can't be separated from the voices. The clips' faint baked-in music beds are in here too. The two original pings (1:08.9–1:11.8) and the app beats (1:16.2–1:29.5) are removed, as in the mix |
| `Wagon_Stem_SFX_*` | Sound design only: taps, key ticks, swipes, the shared notification, transitions and room tone |
| `Wagon_Stem_Voiceover_EMPTY_*` | **Silent placeholder.** No voiceover was supplied or recorded for this cut |

The stems carry the master's loudness processing, so Music + Dialogue + SFX summed reproduces `../Wagon_Full_mix_24bit_48k.wav` (−14 LUFS) to within −51 dB, which is inaudible.

24-bit versions are committed here; the 16-bit copies were sent directly and are rebuilt locally.
