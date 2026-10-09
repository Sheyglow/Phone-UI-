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

REPLY = "I'm in. What time?"
REPLY_TIME = '6:18 PM'


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
    """Draft (reply typed in the field) and sent (reply as a new bubble)."""
    oy = top  # rows added above the supplied screen
    field = chat[oy + 1230, 400].copy()
    # draft: replace the placeholder with the typed reply
    draft = chat.copy()
    draft[oy + 1236:oy + 1268, 120:560] = field
    f = font('Inter-400', 25)
    draft = text_img(draft, (127, oy + 1236), REPLY, f, (24, 24, 28))
    # caret after the text
    tw = ImageFont.truetype(os.path.join(JOB, 'fonts', 'Inter-400.ttf'), 25).getlength(REPLY)
    cv2.rectangle(draft, (int(127 + tw + 4), oy + 1238), (int(127 + tw + 6), oy + 1266), (52, 116, 252), -1)

    # sent: new bubble under Marcus's reply, right-aligned like the first one (x ends at 650)
    sent = chat.copy()
    tw = f.getlength(REPLY)
    bw = int(max(tw, 140) + 54)
    x1, y0 = 650, oy + 690
    x0, h = x1 - bw, 104
    bub = Image.new('RGBA', (bw * 4, h * 4), (0, 0, 0, 0))
    ImageDraw.Draw(bub).rounded_rectangle([0, 0, bw * 4 - 1, h * 4 - 1], 26 * 4, fill=(252, 116, 52, 255))
    bub = bub.resize((bw, h), Image.LANCZOS)
    pil = Image.fromarray(cv2.cvtColor(sent, cv2.COLOR_BGR2RGB)).convert('RGBA')
    pil.alpha_composite(bub, (x0, y0))
    d = ImageDraw.Draw(pil)
    d.text((x0 + 27, y0 + 22), REPLY, font=f, fill=(255, 255, 255))
    ft = font('Inter-400', 17)
    tt = REPLY_TIME + '  ✓✓'
    d.text((x1 - 28 - ft.getlength(tt), y0 + 70), tt, font=ft, fill=(255, 226, 210))
    sent = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    return draft, sent, (x0, y0, x1, y0 + h)


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
            draft, sent, bubble = chat_states(rgb, top)
            cv2.imwrite(os.path.join(OUT, 'wagon-chat-draft.png'), draft)
            cv2.imwrite(os.path.join(OUT, 'wagon-chat-sent.png'), sent)
            meta[key]['bubble'] = bubble
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
