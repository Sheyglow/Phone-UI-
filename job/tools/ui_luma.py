"""Builds the Luma screen states the shots need, from the supplied Luma screens.

Everything not listed here is the supplied UI, pixel for pixel.
- Edward's profile card (photo, name and details), shown before Daniel's.
- The like button empty (outline heart) and filled (the supplied blue heart).
- The chat while typing: Daniel's message, the message bar above an iOS-style
  keyboard, and the typed text. After sending, the supplied chat screen is used.

Usage: python3 job/tools/ui_luma.py   -> job/ui/states/luma-*.png
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
UI = os.path.join(JOB, 'ui')
OUT = os.path.join(UI, 'states')
FONTS = os.path.join(JOB, 'fonts')


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name + '.ttf'), size)


def load(name):
    a = cv2.imread(os.path.join(UI, 'screens', name + '.png'), cv2.IMREAD_UNCHANGED)
    rgb = a[..., :3].copy()
    rgb[a[..., 3] == 0] = 0  # notch and outside the rounded corners: black, like the bezel
    return rgb


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, name + '.png'), img)


def rounded_mask(w, h, r, ss=4):
    m = Image.new('L', (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, fill=255)
    return np.array(m.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255


# ------------------------------------------------------------------ discover
CARD = (15, 208, 675, 1114)   # x0, y0, x1, y1 of the profile card in the screen texture
CARD_R = 30
HEART_C = (430, 1211)          # like button centre
BLUE = (242, 137, 63)          # BGR of the supplied heart (#3F89F2)
OUTLINE_GREY = (180, 155, 140)  # BGR, the undo/star icon grey


def heart_mask(base):
    """The supplied filled heart's shape, as a soft mask over the screen texture."""
    b = base.astype(int)
    x0, y0 = HEART_C[0] - 45, HEART_C[1] - 45
    reg = b[y0:y0 + 90, x0:x0 + 90]
    blue = np.clip((reg[..., 0] - reg[..., 2] - 40) / 80, 0, 1).astype(np.float32)
    m = np.zeros(base.shape[:2], np.float32)
    m[y0:y0 + 90, x0:x0 + 90] = blue
    return m


def empty_heart(base):
    """Replace the filled heart with an outline heart of the same shape."""
    m = heart_mask(base)
    out = base.astype(np.float32)
    btn = np.array([254, 254, 254], np.float32)
    grow = cv2.dilate(m, np.ones((5, 5), np.uint8))
    out = out * (1 - grow[..., None]) + btn * grow[..., None]
    hard = (m > .5).astype(np.uint8)
    inner = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ring = cv2.GaussianBlur((hard - inner).astype(np.float32), (0, 0), .8)
    out = out * (1 - ring[..., None]) + np.array(OUTLINE_GREY, np.float32) * ring[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def heart_layer(base):
    """The supplied filled heart as an RGBA layer (for the like animation)."""
    m = heart_mask(base)
    rgba = np.dstack([np.full(base.shape[:2] + (3,), BLUE, np.uint8), (m * 255).astype(np.uint8)])
    return rgba


def edward_card(base):
    """Edward's card, same layout as Daniel's, as an RGBA layer the size of the screen."""
    x0, y0, x1, y1 = CARD
    w, h = x1 - x0, y1 - y0
    photo = cv2.imread(os.path.join(UI, 'edward-photo.jpg'))
    ph, pw = photo.shape[:2]
    scale = max(w / pw, h / ph)
    crop_w, crop_h = int(w / scale), int(h / scale)
    cx = (pw - crop_w) // 2
    cy = int((ph - crop_h) * .12)  # keep his face in the upper half, like Daniel's
    card = cv2.resize(photo[cy:cy + crop_h, cx:cx + crop_w], (w, h), interpolation=cv2.INTER_AREA)
    card = card.astype(np.float32)
    # warm the grade slightly towards the supplied photos, then darken the lower half for the text
    card = card * np.array([.94, .98, 1.02], np.float32)
    g = np.clip((np.arange(h) / h - .52) / .48, 0, 1) ** 1.25
    card *= (1 - .82 * g)[:, None, None]
    dan = base[y0:y1, x0:x1].astype(np.float32)
    # copy the dots and the row icons from Daniel's card (white on dark: alpha from brightness)
    for (ax0, ay0, ax1, ay1) in [(583, 0, 660, 52)] + [(28, ty - 2, 64, ty + 31) for ty in (706, 751, 797, 846)]:
        patch = dan[ay0:ay1, ax0:ax1]
        alpha = np.clip((patch.max(2) - 110) / 120, 0, 1)[..., None]
        card[ay0:ay1, ax0:ax1] = card[ay0:ay1, ax0:ax1] * (1 - alpha) + 255 * alpha
    img = Image.fromarray(cv2.cvtColor(np.clip(card, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    # positions and sizes measured from Daniel's card (cap tops at 846 / 918 / 962 / 1008 / 1056)
    d.text((33, 624), 'Edward, 30', font=font('NunitoSans-700', 45), fill=(255, 255, 255))
    rows = ['Architect', '2 miles away', 'Columbia', 'Coffee, hiking, vinyl, design']
    for ty, txt in zip((702, 746, 792, 840), rows):
        d.text((77, ty), txt, font=font('NunitoSans-400', 26), fill=(255, 255, 255))
    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    alpha = (rounded_mask(w, h, CARD_R) * 255).astype(np.uint8)
    layer = np.zeros(base.shape[:2] + (4,), np.uint8)
    layer[y0:y1, x0:x1] = np.dstack([rgb, alpha])
    return layer


def over(base, layer):
    a = layer[..., 3:].astype(np.float32) / 255
    return (base * (1 - a) + layer[..., :3] * a).astype(np.uint8)


# ------------------------------------------------------------------ chat
CHAT_BAR = (1290, 1420)   # rows of the message bar (divider to bottom of the field)
KB_TOP = 860              # keyboard top while typing (matches the keyboard in the footage)
BUBBLE_ROWS = (445, 590)  # "See you at 8" bubble and "Sent", absent while typing
FIELD_TEXT = (128, 1336, 470, 1398)  # placeholder area inside the field
INK = (34, 26, 23)        # BGR of the message text (#171A22)


def keyboard(w, h):
    """An iOS-style light keyboard, w x h pixels (BGR)."""
    ss = 2
    img = Image.new('RGB', (w * ss, h * ss), (209, 212, 217))
    d = ImageDraw.Draw(img)
    S = lambda v: int(round(v * ss))
    unit = w / 10
    pred_h = 74
    d.line([(0, S(pred_h)), (S(w), S(pred_h))], fill=(196, 199, 204), width=S(1.5))
    for i, word in enumerate(['"See"', 'See', 'Seen']):
        tx = S(w * (i + .5) / 3)
        f = font('Inter-400', S(29))
        tw = d.textlength(word, font=f)
        d.text((tx - tw / 2, S(20)), word, font=f, fill=(20, 20, 24))
        if i:
            d.line([(S(w * i / 3), S(18)), (S(w * i / 3), S(56))], fill=(185, 188, 194), width=S(1.5))
    key_h, gap_y, top = 84, 22, pred_h + 14
    pad = 6

    def key(x, y, kw, label, dark=False, fsize=37, color=(20, 20, 24)):
        fill = (172, 177, 187) if dark else (255, 255, 255)
        d.rounded_rectangle([S(x + pad), S(y + 2), S(x + kw - pad), S(y + key_h + 2)], S(9), fill=(137, 140, 145))
        d.rounded_rectangle([S(x + pad), S(y), S(x + kw - pad), S(y + key_h)], S(9), fill=fill)
        if label:
            f = font('Inter-400', S(fsize))
            tw = d.textlength(label, font=f)
            d.text((S(x + kw / 2) - tw / 2, S(y + key_h / 2) - S(fsize * .62)), label, font=f, fill=color)
    rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
    for r, letters in enumerate(rows):
        y = top + r * (key_h + gap_y)
        off = (w - len(letters) * unit) / 2
        for i, ch in enumerate(letters):
            key(off + i * unit, y, unit, ch)
        if r == 2:
            key(0, y, unit * 1.35, '⇧', dark=True, fsize=34)
            key(w - unit * 1.35, y, unit * 1.35, '⌫', dark=True, fsize=30)
    y = top + 3 * (key_h + gap_y)
    key(0, y, unit * 1.25, '123', dark=True, fsize=28)
    key(unit * 1.25, y, unit * 1.25, '', dark=True)
    ex, ey = unit * 1.875, y + key_h / 2  # smiley face drawn as strokes (no emoji font here)
    d.ellipse([S(ex - 16), S(ey - 16), S(ex + 16), S(ey + 16)], outline=(30, 30, 34), width=S(2.6))
    d.ellipse([S(ex - 7), S(ey - 7), S(ex - 3), S(ey - 2)], fill=(30, 30, 34))
    d.ellipse([S(ex + 3), S(ey - 7), S(ex + 7), S(ey - 2)], fill=(30, 30, 34))
    d.arc([S(ex - 9), S(ey - 6), S(ex + 9), S(ey + 10)], 20, 160, fill=(30, 30, 34), width=S(2.4))
    key(unit * 2.5, y, unit * 5.0, 'space', fsize=29)
    key(unit * 7.5, y, unit * 2.5, 'send', fsize=29, color=(255, 255, 255))
    # tint the send key blue like the app's send button
    sx0, sx1 = S(unit * 7.5 + pad), S(w - pad)
    d.rounded_rectangle([sx0, S(y), sx1, S(y + key_h)], S(9), fill=(63, 137, 242))
    f = font('Inter-500', S(29))
    tw = d.textlength('send', font=f)
    d.text(((sx0 + sx1) / 2 - tw / 2, S(y + key_h / 2) - S(29 * .62)), 'send', font=f, fill=(255, 255, 255))
    # globe and mic row
    gy = y + key_h + 26
    d.ellipse([S(34), S(gy + 2), S(70), S(gy + 38)], outline=(90, 92, 98), width=S(2.5))
    d.line([(S(52), S(gy + 2)), (S(52), S(gy + 38))], fill=(90, 92, 98), width=S(2))
    d.line([(S(34), S(gy + 20)), (S(70), S(gy + 20))], fill=(90, 92, 98), width=S(2))
    mx = w - 58
    d.rounded_rectangle([S(mx - 9), S(gy), S(mx + 9), S(gy + 26)], S(9), outline=(90, 92, 98), width=S(2.5))
    d.arc([S(mx - 16), S(gy + 8), S(mx + 16), S(gy + 34)], 0, 180, fill=(90, 92, 98), width=S(2.5))
    d.line([(S(mx), S(gy + 34)), (S(mx), S(gy + 40))], fill=(90, 92, 98), width=S(2.5))
    return cv2.cvtColor(np.array(img.resize((w, h), Image.LANCZOS)), cv2.COLOR_RGB2BGR)


_KB = {}


def chat_typing(final, text, cursor=True, kb_offset=0, down=None):
    """Chat screen while typing `text`. kb_offset slides the keyboard down (0 = fully up).

    down (0..1), when given, places the keyboard between fully up (0) and fully hidden (1),
    with the message bar riding on top of it back to its resting place."""
    h, w = final.shape[:2]
    img = final.copy()
    page = final[700, 300].copy()
    img[BUBBLE_ROWS[0]:BUBBLE_ROWS[1], 20:w - 2] = page          # nothing sent yet
    img[CHAT_BAR[0]:h - 40, 20:w - 2] = page                     # bar moves up with the keyboard
    bar = final[CHAT_BAR[0]:CHAT_BAR[1]].copy()
    bh = CHAT_BAR[1] - CHAT_BAR[0]
    by_up = KB_TOP - bh - 6
    if down is not None:
        by = int(round(by_up + (CHAT_BAR[0] - by_up) * down))
        kb_offset = int(round((h - KB_TOP) * down))
    else:
        by = by_up + kb_offset
    img[by:by + bh, 2:w - 2] = bar[:, 2:w - 2]
    # typed text replaces the placeholder
    fx0, fy0, fx1, fy1 = FIELD_TEXT
    fy0 += by - CHAT_BAR[0]
    fy1 += by - CHAT_BAR[0]
    field = final[1330, 330].copy()  # plain field colour (#F4F5F6)
    if text:
        img[fy0:fy1, fx0:fx1] = field
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = font('NunitoSans-400', 33)
    ink = tuple(int(c) for c in INK[::-1])
    tx, ty = fx0 + 2, fy0 + 9
    plain, emoji = (text[:-1], text[-1]) if text.endswith('😊') else (text, '')
    if text:
        d.text((tx, ty), plain, font=f, fill=ink)
        tx += d.textlength(plain, font=f)
    if emoji:
        tx += 6
        smile = emoji_glyph(34)
        pil.paste(smile, (int(tx), ty + 4), smile)
        tx += smile.width
    if cursor:
        d.rectangle([tx + 3, fy0 + 10, tx + 6, fy1 - 8], fill=(63, 137, 242))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if (w, h) not in _KB:
        _KB[(w, h)] = keyboard(w, h - KB_TOP)
    kb = _KB[(w, h)]
    ky = KB_TOP + kb_offset
    if ky < h:
        img[ky:h] = kb[:h - ky]
    # keep the supplied rounded corners and home indicator area black outside the screen shape
    img[final.sum(2) == 0] = 0
    return img


def emoji_glyph(size):
    """The 😊 from the supplied "See you at 8" bubble, keyed off its blue background."""
    src = cv2.imread(os.path.join(UI, 'screens', 'luma-chat-daniel.png'), cv2.IMREAD_UNCHANGED)[..., :3]
    reg = src[477:522, 598:643].astype(int)  # the emoji measured at x 602-638, y 481-517
    blue = np.clip((reg[..., 0] - reg[..., 2] - 40) / 80, 0, 1)
    face = 1 - blue
    # pull blue fringe on the edge back to the emoji's own yellow
    edge = (face > .05) & (face < .95)
    reg[edge] = (reg[edge] * face[edge][:, None] + np.array([40, 200, 255]) * (1 - face[edge][:, None])).astype(int)
    rgba = np.dstack([cv2.cvtColor(reg.astype(np.uint8), cv2.COLOR_BGR2RGB), (face * 255).astype(np.uint8)])
    im = Image.fromarray(rgba, 'RGBA')
    return im.resize((size, int(size * im.height / im.width)), Image.LANCZOS)


def main():
    daniel = load('luma-discover-daniel')
    save('luma-daniel-liked', daniel)
    save('luma-daniel-empty', empty_heart(daniel))
    cv2.imwrite(os.path.join(OUT, 'luma-heart-layer.png'), heart_layer(daniel))
    card = edward_card(daniel)
    cv2.imwrite(os.path.join(OUT, 'luma-edward-card.png'), card)
    save('luma-edward', over(empty_heart(daniel), card))

    chat = load('luma-chat-daniel')
    save('luma-chat-sent', chat)
    msg = 'See you at 8 😊'
    for n in range(len(msg) + 1):
        save(f'luma-chat-typing-{n:02d}', chat_typing(chat, msg[:n]))
    save('luma-chat-typing-nocursor', chat_typing(chat, msg, cursor=False))
    print('wrote', sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
