# Wagon commercial: phone UI and motion graphics

The client's edit (`source/main.mp4`, 854×480, 110 s) with every phone screen replaced by the supplied app UI, the split-screen graphics animated, and the Wagon end card added. The output is `output/wagon-commercial-1080p.mp4` (1920×1080, 24 fps, 1:54, with the edit's own audio), plus `output/wagon-commercial-share-720p.mp4` for sending. The full-bitrate master the renderer writes to `output/wagon-commercial.mp4` is too large for GitHub, so it stays local.

## What happens in each shot

| Source frames | Shot | Screen work |
| --- | --- | --- |
| 0–16 | First swipe | **Cut**, as asked. The film starts on the second swipe |
| 17–125 | Her on the couch, Luma | **Edward** (new profile, using the supplied photo) is swiped off with her thumb. **Daniel** appears with an empty heart, and her thumb tap fills it **blue** with a pop |
| 126–191 | Close-up, typing | Luma chat with Daniel. **"See you at 8 😊" is typed one letter at a time** in time with her thumbs, then **sent**: the keyboard drops and the bubble rises into place |
| 192–1859 | Coffee, park, friends | Unchanged |
| 1860–1935 | Split screen, both smiling | **Wagon logo** pops in on the split. **Animated split light**: energy pulses run up and down the line, the glow breathes and sparks drift. **Floating Discover phones** slide in on each side |
| 1936–2023 | Split screen, two phones | Dating Discover (Jacob) and Friendship Discover (Ethan). Jacob's card follows her left thumb's drag. **Each heart is pressed when the thumb reaches it**: squeeze, pop, glow ring and floating hearts. The logo and split light carry on |
| 2024–2093 | Her phone over the shoulder | **It's a Match!** The centre heart pulses and bursts when her finger reaches it |
| 2094–2156 | His phone over the shoulder | Marcus chat. His reply is in the field, and **his thumb tap sends it** as a new orange bubble |
| 2157–2640 | Café, bar, group | Unchanged |
| +120 frames | End card | The supplied Wagon card, animated element by element. The icon pops, the wordmark rises, the tagline and script lines wipe on, and the heart divider grows. Light runs along the arcs during a slow push-in |

Everything shown on a screen is the supplied UI. Only these were added: Edward's card (laid out exactly like Daniel's), the typing states with a keyboard, the empty heart, and Marcus's new reply. The UI inside the shot images was not used.

## How it works

| Script | What it does |
| --- | --- |
| `tools/track_screens.py` | Tracks every phone screen. White screens are keyed and their edges fitted only where they border the bezel, so thumbs on a corner can't tilt the quad. Optical flow fills frames where the screen leaves the frame |
| `tools/thumbs.py` | Finds the thumbs and fingers over each screen, in screen coordinates. Taps, drags and keystrokes are timed from this |
| `tools/ui_luma.py` | Builds the Luma states: Edward's card, the empty and filled heart, and the typing frames with the keyboard |
| `tools/ui_wagon.py` | Prepares the Wagon screens. Fills the rounded corners, erases the mockup's island (the real phone shows its own) and builds Marcus's typed and sent reply |
| `tools/mg_assets.py` | Cuts the logo, wordmark and floating phones from the supplied images. Separates the end card into a clean background plate plus a mask for each element |
| `tools/render_job.py` | Renders the film: upscale, screen composites with thumbs kept in front, interactions, split graphics, end card and audio |

```sh
python3 job/tools/track_screens.py && python3 job/tools/thumbs.py s01-luma s02-luma-chat s10-left s10-right s11-her s12-his
python3 job/tools/ui_luma.py && python3 job/tools/ui_wagon.py && python3 job/tools/mg_assets.py
python3 job/tools/render_job.py                      # -> job/output/wagon-commercial.mp4
python3 job/tools/render_job.py --stills 64 150 1980 # check single frames
```

Fonts: Nunito Sans (Luma) and Inter (keyboard and Wagon chat), both under the SIL Open Font License, in `fonts/`.
