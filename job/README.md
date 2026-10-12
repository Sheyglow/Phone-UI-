# Wagon commercial: phone UI and motion graphics

The client's edit (`source/main.mp4`, 854×480, 110 s) with every phone screen replaced by the supplied app UI, the split-screen graphics animated, and the Wagon end card added. The output is `output/wagon-commercial-1080p.mp4` (1920×1080, 24 fps, 1:53, with the edit's own audio), plus `output/wagon-commercial-share-720p.mp4` for sending. The full-bitrate master the renderer writes to `output/wagon-commercial.mp4` is too large for GitHub, so it stays local.

## What happens in each shot

Version 2 follows the reference cut (`reference/better.mp4`): the screens carry the story from shot to shot.

| Source frames | Shot | Screen work |
| --- | --- | --- |
| 0–5 | First swipe | **Cut**. The film starts on the second swipe |
| 6–125 | Her on the couch, Luma | **Edward's** card is thrown left on her swipe and **Daniel's** rises from behind. Her tap fills the heart **blue**. Luma then opens the **chat with Daniel** ("Can't wait to meet you."), she taps the field, the **keyboard rises**, and "S… Se… See" starts typing |
| 126–191 | Close-up, typing | **"See you at 8 😊" typed letter by letter**, then **sent**: the keyboard drops and the bubble rises |
| 1860–1935 | Split screen, both smiling | **Ring burst** on the split line, then the **Wagon logo** pops in with its wordmark. **Floating Discover phones** (Jacob, Marcus) slide in with a magenta rim light. The split light is animated |
| 1936–2023 | Split screen, two phones | Dating Discover (**Jacob, 29**) and Friendship Discover (**Marcus, 34**). Each heart is pressed when the thumb reaches it. Split light only, no logo |
| 2024–2093 | Her phone over the shoulder | Her **Discover** screen. She taps Jacob's photo, a **heart pops** under her finger, then **It's a Match!** blooms in ("You and Jacob liked each other."). The match heart pulses |
| 2094–2156 | His phone over the shoulder | **Chat with Marcus, built live**: his message is in the field, he taps send, **his bubble pops in**, then **Marcus is typing…** and **Marcus's reply** appears. All text is from the supplied chat |
| 2618–2726 | End card | Dissolves in over the last shot through a violet wash. The icon pops in with a **ring burst**, the wordmark rises, the tagline and script lines wipe on, the divider grows and **"SOBER. SOCIAL. CONNECTED." types on**. Light and sparkles travel the arcs, the heart pulses and a **shine** sweeps the icon |

Text edits for continuity (`tools/ui_edits.py`): Jacob is 29 on Discover and on the match card, the friendship profile is **Marcus, 34** (the person he then chats with), and the match line reads "You and Jacob liked each other." because the phone is hers. Everything else on screen is the supplied UI.

## How it works

| Script | What it does |
| --- | --- |
| `tools/track_screens.py` | Tracks every phone screen. White screens are keyed and their edges fitted only where they border the bezel, so thumbs on a corner can't tilt the quad. Optical flow fills frames where the screen leaves the frame |
| `tools/thumbs.py` | Finds the thumbs and fingers over each screen, in screen coordinates. Taps, drags and keystrokes are timed from this |
| `tools/ui_edits.py` | Story text edits on the Wagon mockups (names, ages, match line), each fitted to the original text's size |
| `tools/extract_screens.py` | Cuts each screen out of its mockup (edited version when there is one) |
| `tools/ui_luma.py` | Builds the Luma states: Edward's card, the empty and filled heart, and the typing frames with the keyboard |
| `tools/ui_wagon.py` | Prepares the Wagon screens. Fills the rounded corners, erases the mockup's island (the real phone shows its own) and builds Marcus's typed and sent reply |
| `tools/mg_assets.py` | Cuts the logo, wordmark and floating phones from the supplied images. Separates the end card into a clean background plate plus a mask for each element |
| `tools/render_job.py` | Renders the film: upscale, screen composites with thumbs kept in front, interactions, split graphics, end card and audio |

```sh
python3 job/tools/track_screens.py && python3 job/tools/thumbs.py s01-luma s02-luma-chat s10-left s10-right s11-her s12-his
python3 job/tools/ui_edits.py && python3 job/tools/extract_screens.py
python3 job/tools/ui_luma.py && python3 job/tools/ui_wagon.py && python3 job/tools/mg_assets.py
python3 job/tools/render_job.py                      # -> job/output/wagon-commercial.mp4
python3 job/tools/render_job.py --stills 64 150 1980 # check single frames
```

Fonts: Nunito Sans (Luma) and Inter (keyboard and Wagon chat), both under the SIL Open Font License, in `fonts/`.
