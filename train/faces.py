"""Detect faces with YuNet and write <dataset>/faces.json.

Boxes are head boxes in 0..1 image coordinates ([x0, y0, x1, y1, score]):
the detected face widened a little and raised to include the hair, since a
crop through the forehead cuts the head as badly as one through the eyes.
YuNet (OpenCV Zoo, MIT licence) runs on a 640 px copy of each image.

Usage: python faces.py <dataset-dir>...
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

MODEL = str(Path(__file__).parent.parent / "eval" / ".cache" / "models" / "yunet.onnx")
SIZE = 640
detector = None


def detect(path):
    global detector
    img = cv2.imread(str(path))
    if img is None:
        return path.stem, []
    h, w = img.shape[:2]
    scale = min(1, SIZE / max(w, h))
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    if detector is None:
        detector = cv2.FaceDetectorYN.create(MODEL, "", (sw, sh), score_threshold=0.75)
    detector.setInputSize((sw, sh))
    _, faces = detector.detect(small)
    heads = []
    for f in faces if faces is not None else []:
        x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[-1]
        # Face to head: 15% wider on each side, 35% higher, 5% lower
        x0, x1 = (x - 0.15 * fw) / sw, (x + 1.15 * fw) / sw
        y0, y1 = (y - 0.35 * fh) / sh, (y + 1.05 * fh) / sh
        heads.append([round(float(v), 4) for v in (max(0, x0), max(0, y0), min(1, x1), min(1, y1))] + [round(float(score), 3)])
    return path.stem, heads


if __name__ == "__main__":
    for d in sys.argv[1:]:
        d = Path(d)
        paths = sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        with Pool() as pool:
            found = dict(pool.map(detect, paths, chunksize=64))
        found = {k: v for k, v in found.items() if v}
        (d / "faces.json").write_text(json.dumps(found, separators=(",", ":")))
        print(f"{d.name}: faces in {len(found)} of {len(paths)} images")
