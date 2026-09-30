# Kinly launch spot: pitch sample

A sample for the *Cinematic App Launch Commercial* brief: a 34-second, 16:9 spot for Kinly with music. It's cut from the supplied green-screen and b-roll footage, and the five supplied app screens are composited onto every phone.

The brief treats phone-screen text as non-negotiable. For that reason, the app UI in this sample is never generated or redrawn. Each screen is the supplied image, perspective-pinned onto the tracked phone screen in every frame. Fingers and thumbs are keyed back in front of it, tracking markers are filled, and green spill is removed from skin and bezels. Every label and message stays exactly as supplied, including in close-ups.

## Deliverables

| File | What it is |
| --- | --- |
| `output/kinly-launch-spot.mp4` | **The sample.** 1920×1080, 25 fps, H.264 with AAC stereo, 33.8 s (master) |
| `output/kinly-launch-spot-share.mp4` | The same spot at a lower bitrate for sending |
| `output/kinly-ui-motion.mp4` | A 28 s motion-graphics cut of the same screens on a 3D phone, 30 fps, same music |
| `output/stills/` | Key frames from the spot |
| `PITCH.md` | Draft reply to the brief |

## The cut

Cuts land on the beat of the music (115 bpm). The full band comes in at 8.35 s, just after the thumb taps into onboarding, and the end card arrives on the downbeat of bar 14.

| Time | Shot | Screen | Title |
| --- | --- | --- | --- |
| 0.0–3.7 | Friends on their phones (b-roll) | none | Date. Make friends. Or both. (one line per beat) |
| 3.7–9.9 | POV, phone on lap; the thumb taps *Get started* | Welcome, then Step 2 of 4 | Dating and friends, in one app. / Dating, friends, or both. |
| 9.9–14.1 | Close-up, phone against autumn leaves | Discover (Amara) | none |
| 14.1–16.7 | Reaction: someone smiling at their phone in a park | none | none |
| 16.7–21.4 | Two people in a park, both phones matched | It's a match (on both phones) | Matched on what you both love. |
| 21.4–24.5 | Friends on their phones (b-roll) | none | Build your circle. |
| 24.5–29.7 | Evening on the couch; the finger scrolls the chat | Chat with Amara, scrolling | From first message to Friday night. |
| 29.2–33.8 | End card: the rings draw in, wordmark, "Launching soon" | none | none |

**The chat scroll.** The finger in the couch clip swipes up twice. Its fingertip is tracked in screen space, and the message list follows it 1:1, coasts with momentum on release, and rubber-bands at the newest message. The supplied messages stay exactly as supplied. Below them, the conversation continues with new messages designed in the same style (`spot/chat-extension.html`): a film photo, a video clip from The Loft, and a few replies ending on "See you there." The "Read" receipt moves down to the newest message.

Every shot goes through one grade: a plum lift in the shadows, warm highlights, soft contrast, a vignette and film grain. That way stock clips from different cameras cut together as one piece.

## Music

"Familiar Roads" by Tanner Helland, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from [github.com/tannerhelland/free-music](https://github.com/tannerhelland/free-music). The license allows commercial use with credit. Wherever the spot is posted, add: *Music: "Familiar Roads" by Tanner Helland (tannerhelland.com), CC BY 4.0.* Details are in `music/CREDITS.md`.

## How it's built

| Path | Role |
| --- | --- |
| `footage/` | Supplied clips, renamed by content. `footage/tracks/` holds the per-frame screen corners and the couch-clip fingertip path |
| `screens/` | Supplied app screens, numbered in flow order |
| `spot/track.py` | Keys each green screen and fits straight lines to its four edges, so rounded corners, markers and fingers don't pull the corners. Smooths the corners over time and tracks the fingertip that drives the chat scroll |
| `spot/composite.py` | Screen replacement: perspective warp, key-based finger occlusion, marker fill, despill, room-fixed glare |
| `spot/edit.py` | The edit: beat-aligned timeline, reframing and push-ins, UI changes on the thumb press, chat scroll physics, grade, titles, grain, encode and music |
| `spot/supers.mjs` | Renders the titles as transparent layers, plus the chat continuation |
| `spot/chat-extension.html` | The designed chat messages that continue below the supplied screen |
| `spot/kinly-spot.html` | Motion-graphics composition (the UI-motion cut and the end card). Open it in a browser to watch it loop |
| `spot/render.mjs` | Renders that composition to video |
| `spot/fonts/fonts.css` | Fraunces, Figtree and Inter embedded as data URIs (SIL Open Font License), so renders never fall back to system fonts |

### Rebuilding

You need Python 3 with `opencv-python-headless`, `numpy` and `imageio-ffmpeg`, plus Node with Playwright and Chromium. `render.mjs` needs `FFMPEG` set to an ffmpeg that includes libx264, for example `export FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")`.

```sh
python3 spot/track.py                                   # screen tracks + fingertip -> footage/tracks/
node spot/supers.mjs                                    # titles + chat continuation -> spot/build/
node spot/render.mjs --end-only --from 24 --to 28.6 --fps 25 --out spot/build/endcard.mp4
python3 spot/edit.py                                    # -> output/kinly-launch-spot.mp4 (with music)
python3 spot/edit.py --stills                           # key frames only
```

## Not in this sample

- **Voiceover and sound design.** The supplied clips have silent audio tracks, so the spot has music only. The full commercial adds voiceover, sound design and a final mix.
- **Recurring lead characters.** These stock clips show different people. In the full commercial, the two leads are generated from locked reference boards, and their phones are composited the same way as here.
