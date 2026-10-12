"""Story edits to the supplied Wagon mockups, so the screens connect from shot to shot.

- Dating Discover: "Jacob, 34" -> "Jacob, 29".
- Friendship Discover: "Ethan, 32" -> "Marcus, 34". He likes Marcus here, then chats with Marcus.
- It's a Match: "Jacob, 34" -> "Jacob, 29" on the card, and "You and Jessica liked each other."
  -> "You and Jacob liked each other.", since the phone is hers.

Each edit measures the original text, erases it (inpaint from the surrounding pixels), and
draws the new text in Inter at the size and position that matches the original's cap height.

Usage: python3 job/tools/ui_edits.py -> job/ui/edited/<mockup>.png
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
UI = os.path.join(JOB, 'ui')
OUT = os.path.join(UI, 'edited')
FONTS = os.path.join(JOB, 'fonts')


def font(weight, size):
    return ImageFont.truetype(os.path.join(FONTS, f'Inter-{weight}.ttf'), size)


def bright_mask(img, box, thr):
    x0, y0, x1, y1 = box
    m = np.zeros(img.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = (img[y0:y1, x0:x1].min(2) > thr).astype(np.uint8)
    return m


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def erase(img, mask, grow=3):
    hole = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return cv2.inpaint(img, hole, 6, cv2.INPAINT_TELEA)


def fit_size(text, weight, cap_px, ref_char):
    """Font size whose `ref_char` glyph height equals cap_px."""
    lo, hi = 6, 200
    while lo < hi:
        mid = (lo + hi) // 2
        b = font(weight, mid).getbbox(ref_char)
        if b[3] - b[1] < cap_px:
            lo = mid + 1
        else:
            hi = mid
    return lo


def draw(img, text, weight, size, color, left, top, angle=0.0, anchor='lt'):
    """Draw text whose ink box's top-left lands at (left, top); rotate about that point."""
    f = font(weight, size)
    b = f.getbbox(text)
    w, h = b[2] - b[0] + 8, b[3] - b[1] + 8
    lay = Image.new('RGBA', (w * 2 + 40, h * 2 + 40), (0, 0, 0, 0))
    ox, oy = w + 20 - 4, h + 20 - 4   # ink top-left inside the layer
    ImageDraw.Draw(lay).text((ox - b[0], oy - b[1]), text, font=f, fill=tuple(int(c) for c in color[::-1]) + (255,))
    if angle:
        lay = lay.rotate(-angle, resample=Image.BICUBIC, center=(ox, oy))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    pil.alpha_composite(lay, (int(round(left - ox)), int(round(top - oy))))
    return cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def ink_color(img, mask):
    px = img[mask > 0].astype(np.float32)
    lum = px.mean(1)
    return np.median(px[lum >= np.percentile(lum, 70)], 0)


def baseline_angle(mask):
    """Slope of a text line from the lowest ink pixel in each column (comma excluded by trimming)."""
    ys, xs = np.nonzero(mask)
    cols = {}
    for x, y in zip(xs, ys):
        cols[x] = max(cols.get(x, 0), y)
    X = np.array(sorted(cols))
    Y = np.array([cols[x] for x in X])
    keep = Y < np.percentile(Y, 80) + 2  # drop the comma's tail
    k = np.polyfit(X[keep], Y[keep], 1)[0]
    return float(np.degrees(np.arctan(k)))


def edit_dating(img):
    m = bright_mask(img, (432, 898, 500, 952), 190)
    x0, y0, x1, y1 = bbox(m)
    col = ink_color(img, m)
    img = erase(img, m)
    size = fit_size('3', 700, y1 - y0, '3')
    print('dating 34 box', (x0, y0, x1, y1), 'size', size)
    return draw(img, '29', 700, size, col, x0, y0)


def edit_friendship(img):
    m = bright_mask(img, (236, 862, 494, 925), 190)
    x0, y0, x1, y1 = bbox(m)
    cap = bbox(bright_mask(img, (236, 862, 276, 925), 190))
    col = ink_color(img, m)
    # the orange verified badge right of the name
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bm = np.zeros(img.shape[:2], np.uint8)
    bx = (494, 860, 548, 920)
    bm[bx[1]:bx[3], bx[0]:bx[2]] = ((hsv[bx[1]:bx[3], bx[0]:bx[2], 1] > 120) | (img[bx[1]:bx[3], bx[0]:bx[2]].min(2) > 200)).astype(np.uint8)
    bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    b0, b1, b2, b3 = bbox(bm)
    gap = b0 - x1
    badge = img[b1 - 2:b3 + 2, b0 - 2:b2 + 2].copy()
    balpha = cv2.GaussianBlur(cv2.dilate(bm[b1 - 2:b3 + 2, b0 - 2:b2 + 2], np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), .8)
    img = erase(img, np.maximum(m, bm), 4)
    # the app face is a touch narrower than Inter: size by the original string's width
    size = 10
    while font(700, size + 1).getlength('Ethan, 32') <= (x1 - x0) + 1:
        size += 1
    new = 'Marcus, 34'
    print('friend box', (x0, y0, x1, y1), 'cap', cap[3] - cap[1], 'size', size, 'badge', (b0, b1, b2, b3), 'gap', gap)
    eb = font(700, size).getbbox('E')
    img = draw(img, new, 700, size, col, x0, cap[1] + ((cap[3] - cap[1]) - (eb[3] - eb[1])) / 2)
    nb = font(700, size).getbbox(new)
    nx1 = x0 + (nb[2] - nb[0])
    px = int(nx1 + gap - 2)
    h, w = badge.shape[:2]
    reg = img[b1 - 2:b1 - 2 + h, px:px + w].astype(np.float32)
    a = balpha[..., None]
    img[b1 - 2:b1 - 2 + h, px:px + w] = (reg * (1 - a) + badge * a).astype(np.uint8)
    return img


def edit_match(img):
    # subtitle line
    m = bright_mask(img, (300, 378, 790, 418), 150)
    x0, y0, x1, y1 = bbox(m)
    cap = bbox(bright_mask(img, (300, 378, 330, 418), 150))  # the "Y"
    col = ink_color(img, m)
    img = erase(img, m, 3)
    size = fit_size('Y', 400, cap[3] - cap[1], 'Y')
    new = 'You and Jacob liked each other.'
    nb = font(400, size).getbbox(new)
    cx = (x0 + x1) / 2
    print('match subtitle box', (x0, y0, x1, y1), 'size', size)
    img = draw(img, new, 400, size, col, cx - (nb[2] - nb[0]) / 2, cap[1])
    # "34" on Jacob's tilted card
    name = bright_mask(img, (545, 815, 670, 870), 200)
    ang = baseline_angle(name)
    m2 = bright_mask(img, (622, 826, 668, 868), 200)
    d0, d1, d2, d3 = bbox(m2)
    col2 = ink_color(img, m2)
    img = erase(img, m2, 3)
    # unrotated digit height: the box height shrinks by the slant across the width
    h = (d3 - d1) - abs(np.tan(np.radians(ang))) * (d2 - d0)
    size2 = fit_size('3', 700, h, '3')
    print('match 34 box', (d0, d1, d2, d3), 'angle', round(ang, 2), 'size', size2)
    top = d1 if ang < 0 else d1
    return draw(img, '29', 700, size2, col2, d0, top, angle=ang)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [('wagon-discover-dating-jacob', edit_dating),
                     ('wagon-discover-friendship-ethan', edit_friendship),
                     ('wagon-match', edit_match)]:
        img = cv2.imread(os.path.join(UI, name + '.png'))
        cv2.imwrite(os.path.join(OUT, name + '.png'), fn(img))
    print('wrote', sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
