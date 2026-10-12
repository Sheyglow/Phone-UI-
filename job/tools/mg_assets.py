"""Motion-graphics assets from the supplied Wagon artwork.

- icon.png / wordmark.png: the Wagon app icon and wordmark from the logo image (RGBA).
- phone-dating.png / phone-friendship.png: the Discover mockups cut from their white
  backgrounds, for the floating phones in the split-screen shot (RGBA).
- endcard-plate.png: the end card with its text and icon removed (inpainted sky), and
  endcard-<element>.png alpha masks, so each element can be revealed in turn.

Usage: python3 job/tools/mg_assets.py -> job/mg/
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.path.dirname(HERE)
UI = os.path.join(JOB, 'ui')
OUT = os.path.join(JOB, 'mg')


def rounded_rect_mask(h, w, x0, y0, x1, y1, r, ss=4):
    m = np.zeros((h * ss, w * ss), np.uint8)
    x0, y0, x1, y1, r = [int(v * ss) for v in (x0, y0, x1, y1, r)]
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)


def logo_assets():
    img = cv2.imread(os.path.join(UI, 'wagon-logo.png'))
    h, w = img.shape[:2]
    # icon: the rounded square at x 538-993, y 164-606 (its soft outer glow is left out)
    x0, y0, x1, y1 = 541, 167, 990, 603
    m = rounded_rect_mask(h, w, x0, y0, x1, y1, 104)
    icon = np.dstack([img, m])[y0 - 4:y1 + 5, x0 - 4:x1 + 5]
    cv2.imwrite(os.path.join(OUT, 'icon.png'), icon)
    # wordmark: white letters on dark purple; alpha from brightness, colour pure white
    reg = img[640:775, 560:965].astype(np.float32)
    lum = reg.min(2)
    a = np.clip((lum - 95) / 140, 0, 1)
    word = np.dstack([np.full(reg.shape, 255, np.uint8), (a * 255).astype(np.uint8)])
    cv2.imwrite(os.path.join(OUT, 'wordmark.png'), word)


def phone_cutout(name, out):
    edited = os.path.join(UI, 'edited', name + '.png')
    img = cv2.imread(edited if os.path.exists(edited) else os.path.join(UI, name + '.png'))
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = img.copy()
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(flood, mask, seed, (0, 0, 0), (6, 6, 6), (6, 6, 6), cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    bg = mask[1:-1, 1:-1] > 0
    alpha = (~bg).astype(np.float32)
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
    ys, xs = np.nonzero(alpha > .5)
    crop = np.dstack([img, (alpha * 255).astype(np.uint8)])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cv2.imwrite(os.path.join(OUT, out), crop)


# end card elements: name -> bbox (x0, y0, x1, y1) in the 1672x941 card, and how to mask it
ENDCARD = {
    'icon': ((600, 40, 1072, 500), 'icon'),
    'wordmark': ((650, 410, 1025, 532), 'bright'),
    'tagline': ((245, 540, 1430, 636), 'text'),
    'divider': ((590, 645, 1085, 697), 'text'),
    'sober': ((555, 700, 1120, 748), 'text'),
    'script-real': ((70, 75, 300, 220), 'text'),
    'script-brighter': ((170, 300, 380, 425), 'text'),
    'script-stronger': ((1200, 245, 1490, 400), 'text'),
}


def smooth_fill(img, hole):
    """Fill holes with a smooth blend of the surrounding sky (normalised blur, coarse to fine)."""
    valid = (hole == 0).astype(np.float32)
    src = img.astype(np.float32)
    out = src.copy()
    todo = hole > 0
    for sigma in (4, 8, 16, 32, 64, 128, 256, 400):
        num = cv2.GaussianBlur(src * valid[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(valid, (0, 0), sigma)
        ok = todo & (den > (.25 if sigma < 400 else 1e-6))
        out[ok] = num[ok] / den[ok][:, None]
        todo &= ~ok
    # a touch of grain so the filled sky matches the image's texture
    rng = np.random.default_rng(5)
    out[hole > 0] += rng.normal(0, 2.0, (int((hole > 0).sum()), 3))
    return np.clip(out, 0, 255).astype(np.uint8)


def endcard_assets():
    img = cv2.imread(os.path.join(UI, 'wagon-endcard.png'))
    h, w = img.shape[:2]
    hole = np.zeros((h, w), np.uint8)
    smooth_bg = cv2.medianBlur(img, 31).astype(np.float32)  # the sky without thin strokes
    for name, ((x0, y0, x1, y1), kind) in ENDCARD.items():
        reg = img[y0:y1, x0:x1].astype(np.float32)
        if kind == 'icon':
            m = rounded_rect_mask(h, w, 697, 132, 974, 397, 64)[y0:y1, x0:x1].astype(np.float32) / 255
            # the icon plus its shadow and glow, which reach about 40 px past the square
            soft = cv2.GaussianBlur(cv2.dilate(m, np.ones((121, 121), np.uint8)), (0, 0), 14)
            a = np.maximum(m, soft)
        elif kind == 'bright':  # thick white letters: brightness, not difference from a blur
            a = np.clip((reg.min(2) - 110) / 90, 0, 1)
            a = cv2.GaussianBlur(cv2.dilate(a, np.ones((5, 5), np.uint8)), (0, 0), 1)
        else:
            diff = np.abs(reg - smooth_bg[y0:y1, x0:x1]).max(2)
            a = np.clip((diff - 18) / 50, 0, 1)
            a = cv2.GaussianBlur(cv2.dilate(a, np.ones((3, 3), np.uint8)), (0, 0), .8)
        full = np.zeros((h, w), np.float32)
        full[y0:y1, x0:x1] = a
        cv2.imwrite(os.path.join(OUT, f'endcard-{name}.png'), (full * 255).astype(np.uint8))
        hole[y0:y1, x0:x1] |= (cv2.dilate(a, np.ones((11, 11), np.uint8)) > .04).astype(np.uint8)
    plate = smooth_fill(img, hole)
    cv2.imwrite(os.path.join(OUT, 'endcard-plate.png'), plate)
    cv2.imwrite(os.path.join(OUT, 'endcard-full.png'), img)
    # the glowing arcs (for the travelling shimmer): bright saturated magenta strokes in the sky
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    arcs = ((hsv[..., 1] > 90) & (hsv[..., 2] > 150) & (np.arange(h)[:, None] < 520)).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT, 'endcard-arcs.png'), cv2.GaussianBlur(arcs, (0, 0), 1.2))


def main():
    os.makedirs(OUT, exist_ok=True)
    logo_assets()
    phone_cutout('wagon-discover-dating-jacob', 'phone-dating.png')
    phone_cutout('wagon-discover-friendship-ethan', 'phone-friendship.png')
    endcard_assets()
    print(sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
