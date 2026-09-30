# Kinly launch spot: pitch sample

A 28-second, 16:9 motion sample for the *Cinematic App Launch Commercial* brief. It uses the five supplied Kinly app screens.

The brief treats phone-screen text as non-negotiable. For that reason, the screens in this sample are never generated or redrawn. Each one is the supplied image, composited onto a 3D phone with camera moves, glare and grain. Every label and message stays pixel-accurate and correctly spelled, even in close-ups.

## Files

| Path | What it is |
| --- | --- |
| `output/kinly-launch-sample.mp4` | Rendered sample, 1920×1080, 30 fps, H.264 |
| `output/stills/` | Key frames for a quick look or a thumbnail |
| `spot/kinly-spot.html` | The composition. Open it in a browser to watch it loop live |
| `spot/render.mjs` | Renders the composition frame by frame and encodes the MP4 |
| `screens/` | The supplied app screens, numbered in flow order |

## Sequence

| Time | Beat | Screen |
| --- | --- | --- |
| 0.0–3.4 | "Date. Make friends. Or both." over bokeh | none |
| 3.3–6.8 | The phone rises into frame; tap on *Get started* | Welcome |
| 6.7–10.9 | Push in on the *Both* card, then tap *Continue* | Step 2 of 4 |
| 10.9–14.9 | Amara's profile; the heart button presses | Discover |
| 14.9–18.5 | Warm light bloom; tap on *Send a message* | It's a match |
| 18.5–23.5 | Slow tilt down the conversation to the Jazz Night card | Chat |
| 23.5–28.0 | The logo rings draw in; wordmark and "Launching soon" | End card |

## Re-rendering

You need Node with Playwright and Chromium, plus an ffmpeg build that includes libx264.

```sh
node spot/render.mjs            # full MP4 into output/
node spot/render.mjs --stills   # key frames only
```

Set `PLAYWRIGHT_PATH`, `CHROMIUM_PATH` or `FFMPEG` when those tools are not on the default paths. Every frame is a pure function of time (`renderAt(t)` in the composition), so repeated renders come out identical.

## Not in this sample

The sample has no audio yet. The full commercial adds voiceover, music and a final mix. The live-action footage with the two recurring leads will come from AI video generation. These screens will then be tracked onto the phones in that footage using the same compositing approach.
