# Kinly launch spot: pitch sample

A sample for the *Cinematic App Launch Commercial* brief: a 34-second, 16:9 spot for Kinly. It's cut from the supplied green-screen and b-roll footage, and the five supplied app screens are composited onto every phone.

The brief treats phone-screen text as non-negotiable. For that reason, the app UI in this sample is never generated or redrawn. Each screen is the supplied image, perspective-pinned onto the tracked phone screen in every frame. Fingers and thumbs are keyed back in front of it, tracking markers are filled, and green spill is removed from skin and bezels. Every label and message stays exactly as supplied, including in close-ups.

## Deliverables

| File | What it is |
| --- | --- |
| `output/kinly-launch-spot.mp4` | **The sample.** 1920×1080, 25 fps, H.264, about 34 s |
| `output/kinly-ui-motion.mp4` | A 28 s motion-graphics cut of the same screens on a 3D phone, 30 fps |
| `output/stills/` | Key frames from the spot |
| `PITCH.md` | Draft reply to the brief |

## The cut

| Time | Shot | Screen | Title |
| --- | --- | --- | --- |
| 0.0–3.6 | Friends on their phones (b-roll) | none | Date. Make friends. Or both. |
| 3.6–10.0 | POV, phone on lap; the thumb taps *Get started* | Welcome, then Step 2 of 4 | Dating and friends, in one app. / Dating, friends, or both. |
| 10.0–14.0 | Close-up, phone against autumn leaves | Discover (Amara) | none |
| 14.0–17.0 | Reaction: someone smiling at their phone in a park | none | none |
| 17.0–21.4 | Two people in a park, both phones matched | It's a match (on both phones) | Matched on what you both love. |
| 21.4–24.4 | Friends on their phones (b-roll) | none | Build your circle. |
| 24.4–30.0 | Evening on the couch, a finger on the chat | Chat with Amara | From first message to Friday night. |
| 29.4–33.8 | End card: the rings draw in, wordmark, "Launching soon" | none | none |

Every shot goes through one grade: a plum lift in the shadows, warm highlights, soft contrast, a vignette and film grain. That way stock clips from different cameras cut together as one piece.

## How it's built

| Path | Role |
| --- | --- |
| `footage/` | Supplied clips, renamed by content. `footage/tracks/` holds the per-frame screen corners |
| `screens/` | Supplied app screens, numbered in flow order |
| `spot/track.py` | Keys each green screen, fits straight lines to its four edges (rounded corners, markers and fingers don't pull the corners) and smooths the corners over time |
| `spot/composite.py` | Screen replacement: perspective warp, key-based finger occlusion, marker fill, despill, room-fixed glare |
| `spot/edit.py` | The edit: timeline, reframing and push-ins, UI changes on the thumb press, grade, titles, grain, encode |
| `spot/supers.mjs` | Renders the titles as transparent layers in the app's typefaces |
| `spot/kinly-spot.html` | Motion-graphics composition (the UI-motion cut and the end card). Open it in a browser to watch it loop |
| `spot/render.mjs` | Renders that composition to video |

### Rebuilding

You need Python 3 with `opencv-python-headless`, `numpy` and `imageio-ffmpeg`, plus Node with Playwright and Chromium.

```sh
python3 spot/track.py                                   # screen tracks -> footage/tracks/
node spot/supers.mjs                                    # titles -> spot/build/supers/
node spot/render.mjs --end-only --from 23.6 --to 28 --fps 25 --out spot/build/endcard.mp4
python3 spot/edit.py                                    # -> output/kinly-launch-spot.mp4
python3 spot/edit.py --stills                           # key frames only
node spot/render.mjs                                    # -> output/kinly-ui-motion.mp4
```

## Not in this sample

- **Audio.** The supplied clips have silent audio tracks, so the spot has no sound yet. The full commercial adds voiceover, music, sound design and a final mix.
- **Recurring lead characters.** These stock clips show different people. In the full commercial, the two leads are generated from locked reference boards, and their phones are composited the same way as here.
