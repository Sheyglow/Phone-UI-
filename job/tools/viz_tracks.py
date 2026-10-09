"""Draws tracked screen quads over sample frames: job/analysis/track-<screen>.jpg"""
import json, os, sys
import cv2, numpy as np
JOB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cap = cv2.VideoCapture(os.path.join(JOB, 'source', 'main.mp4'))
for name in sys.argv[1:]:
    t = json.load(open(os.path.join(JOB, 'tracks', name + '.json')))
    q = np.array(t['corners']); n = len(q); tiles = []
    for i in np.linspace(0, n - 1, 12).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t['first'] + i); ok, fr = cap.read()
        cv2.polylines(fr, [np.round(q[i] * 4).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA, shift=2)
        cv2.putText(fr, f"{t['first'] + i} {t['status'][i]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        tiles.append(cv2.resize(fr, (427, 240)))
    cv2.imwrite(os.path.join(JOB, 'analysis', f'track-{name}.jpg'),
                np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, 12, 4)], 0))
