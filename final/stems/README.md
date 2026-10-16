# Wagon stems

Four stems, one per track: 48 kHz stereo WAV, each exactly 1:53.77 and starting on frame 1. Line them all up at 0 on the timeline.

| Stem | Contents |
| --- | --- |
| `Wagon_Stem_Music_*` | The score only, with its ducking under the dialogue and voiceover |
| `Wagon_Stem_Dialogue_*` | The clips' own production audio, edited to the client's notes: dialogue plus the room and bar ambience recorded with it, which can't be separated from the voices. Removed stretches are filled with each location's own air. Daniel's order is trimmed to "I'll do an Old Fashion." in his own voice. There's no dialogue from the Wagon reveal (1:16.2) to the end |
| `Wagon_Stem_SFX_*` | Sound design only: taps, key ticks, swipes, Ethan's phone buzz and text sound, the shared notification, transitions and room tone |
| `Wagon_Stem_Voiceover_*` | The narrator: the 10 script lines, Higgsfield voice "Holden" |

The stems carry the master's loudness processing, so Music + Dialogue + SFX + Voiceover summed reproduces `../Wagon_Full_mix_24bit_48k.wav` (−14 LUFS) to within −56 dB, which is inaudible.

24-bit versions are committed here; the 16-bit copies were sent directly and are rebuilt locally.
