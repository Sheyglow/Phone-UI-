"""Prepares the supplied Wagon screens for compositing, plus the chat's new message.

- Fills the mockup's rounded corners with the screen's own edge colours (the real
  phone's screen shape clips them), and extends the top and bottom backgrounds so
  each screen has the phones' iPhone proportions without squashing the UI.
- Marcus chat: his reply typed in the message field, then sent as a new bubble in
  the same style as his first message.

Usage: python3 job/tools/ui_wagon.py -> job/ui/states/wagon-*.png and wagon-meta.json
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
UI = os.path.join(JOB, 'ui')
OUT = os.path.join(UI, 'states')
ASPECT = 0.4656  # the phones' screen shape (width / height), same as the Luma screens

# his message, as supplied in the chat's orange bubble
MESSAGE = ['Still pretty new to recovery. Trying', 'to meet more people in it.']


def font(name, size):
    return ImageFont.truetype(os.path.join(JOB, 'fonts', name + '.ttf'), size)


def prep(name):
    a = cv2.imread(os.path.join(UI, 'screens', name + '.png'), cv2.IMREAD_UNCHANGED)
    rgb, alpha = a[..., :3].copy(), a[..., 3]
    hole = (alpha < 250).astype(np.uint8)
    # the mockup's dynamic island: the phones in the footage show their own, so erase this one
    h0, w0 = rgb.shape[:2]
    top_band = np.zeros_like(hole)
    top_band[:90, int(w0 * .25):int(w0 * .75)] = 1
    island = ((rgb.max(2) < 45) & (top_band > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(island)
    if n > 1:
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        hole |= cv2.dilate((lab == k).astype(np.uint8), np.ones((9, 9), np.uint8))
    rgb = cv2.inpaint(rgb, hole, 7, cv2.INPAINT_TELEA)
    # drop the mockup's outermost rows (anti-aliased bezel), then pad with the screen's own
    # background colours, sampled just inside the top and bottom edges
    trim = 6
    rgb = rgb[trim:-trim]
    h, w = rgb.shape[:2]
    extra = int(round(w / ASPECT)) - h
    top = extra // 2
    tcol = np.median(rgb[14:24, 30:90].reshape(-1, 3), 0)
    bcol = np.median(rgb[h - 24:h - 14, 30:90].reshape(-1, 3), 0)
    out = np.vstack([np.tile(tcol, (top, w, 1)), rgb, np.tile(bcol, (extra - top, w, 1))]).astype(np.uint8)
    return out, top - trim


def text_img(img, xy, txt, f, fill):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, txt, font=f, fill=fill)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def chat_states(chat, top):
    """The supplied conversation, built up live (all text as supplied):
    empty chat with his message in the field -> his bubble -> Marcus typing -> Marcus's reply.

    Returns the screens plus the layers and boxes the renderer animates.
    """
    oy = top  # rows added above the supplied screen
    page = np.median(chat[oy + 700:oy + 760, 200:500].reshape(-1, 3), 0).astype(np.uint8)
    ORANGE = (oy + 340, oy + 492, 205, 656)      # his bubble (y0, y1, x0, x1)
    REPLY = (oy + 505, oy + 668, 8, 560)         # Marcus's avatar and reply
    empty = chat.copy()
    for y0, y1, x0, x1 in (ORANGE, REPLY):
        empty[y0:y1, x0:x1] = page
    # draft: the field grows to two lines with his message in it
    draft = empty.copy()
    fx0, fx1, fy1 = 101, 571, oy + 1283
    fy0 = fy1 - 96
    draft[fy0 - 8:oy + 1214, 96:575] = page   # room for the taller field (the "+" stays put)
    pil = Image.fromarray(cv2.cvtColor(draft, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    ss = 4
    fld = Image.new('RGBA', ((fx1 - fx0) * ss, (fy1 - fy0) * ss), (0, 0, 0, 0))
    ImageDraw.Draw(fld).rounded_rectangle([0, 0, (fx1 - fx0) * ss - 1, (fy1 - fy0) * ss - 1], 33 * ss,
                                          fill=(248, 247, 249, 255), outline=(232, 230, 234, 255), width=2 * ss)
    fld = fld.resize((fx1 - fx0, fy1 - fy0), Image.LANCZOS)
    pil = pil.convert('RGBA')
    pil.alpha_composite(fld, (fx0, fy0))
    d = ImageDraw.Draw(pil)
    f = font('Inter-400', 24)
    for i, line in enumerate(MESSAGE):
        d.text((127, fy0 + 17 + i * 32), line, font=f, fill=(24, 24, 28))
    # caret after the last word
    cx = 127 + f.getlength(MESSAGE[-1]) + 3
    d.rectangle([cx, fy0 + 20 + 32, cx + 2, fy0 + 20 + 32 + 26], fill=(52, 116, 252))
    draft = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    # typing indicator: Marcus's avatar and a small grey bubble with three dots (dots drawn per frame)
    typing = empty.copy()
    av = (oy + 560, oy + 652, 8, 112)
    typing[av[0]:av[1], av[2]:av[3]] = chat[av[0]:av[1], av[2]:av[3]]
    bub = (119, oy + 588, 119 + 112, oy + 588 + 62)   # bottom-aligned with the avatar
    pil = Image.fromarray(cv2.cvtColor(typing, cv2.COLOR_BGR2RGB)).convert('RGBA')
    b = Image.new('RGBA', ((bub[2] - bub[0]) * ss, (bub[3] - bub[1]) * ss), (0, 0, 0, 0))
    ImageDraw.Draw(b).rounded_rectangle([0, 0, b.width - 1, b.height - 1], 26 * ss, fill=(242, 242, 242, 255))
    pil.alpha_composite(b.resize((bub[2] - bub[0], bub[3] - bub[1]), Image.LANCZOS), (bub[0], bub[1]))
    typing = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    sent = empty.copy()
    sent[ORANGE[0]:ORANGE[1], ORANGE[2]:ORANGE[3]] = chat[ORANGE[0]:ORANGE[1], ORANGE[2]:ORANGE[3]]
    boxes = {'orange': ORANGE, 'reply': REPLY, 'dots': bub, 'avatar': av}
    return {'empty': empty, 'draft': draft, 'sent': sent, 'typing': typing, 'full': chat}, boxes


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for name, key in [('wagon-discover-dating-jacob', 'wagon-dating'),
                      ('wagon-discover-friendship-ethan', 'wagon-friendship'),
                      ('wagon-match', 'wagon-match'),
                      ('wagon-chat-marcus', 'wagon-chat')]:
        rgb, top = prep(name)
        cv2.imwrite(os.path.join(OUT, key + '.png'), rgb)
        meta[key] = {'size': [rgb.shape[1], rgb.shape[0]], 'top': top}
        if key == 'wagon-chat':
            screens, boxes = chat_states(rgb, top)
            for k, v in screens.items():
                cv2.imwrite(os.path.join(OUT, f'wagon-chat-{k}.png'), v)
            meta[key].update({k: [int(x) for x in v] for k, v in boxes.items()})
    # interaction points, in the prepared textures' pixels (measured on the supplied screens)
    meta['wagon-dating'].update(heart=[409, 1153 + meta['wagon-dating']['top']], ring=58, color=[255, 85, 200],
                                card=[10, 245 + meta['wagon-dating']['top'], 657, 1072 + meta['wagon-dating']['top']])
    meta['wagon-friendship'].update(heart=[395, 1125 + meta['wagon-friendship']['top']], ring=58, color=[40, 140, 255])
    meta['wagon-match'].update(heart=[334, 640 + meta['wagon-match']['top']], ring=48, color=[255, 85, 200])
    meta['wagon-chat'].update(send=[617, 1249 + meta['wagon-chat']['top']])
    json.dump(meta, open(os.path.join(OUT, 'wagon-meta.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
