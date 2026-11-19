"""VisDrone2019-DET -> a single-class ("vehicle") YOLO dataset.

Images are hard-linked rather than copied, so the YOLO folder costs no extra
disk space (falls back to copying where links are not possible).

Usage: python scripts/prepare_det.py   (reads data/raw/det_*.zip, writes data/yolo/)
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavtrack.visdrone import det_to_yolo  # noqa: E402

SPLITS = {"train": "det_train.zip", "val": "det_val.zip"}


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    raw, out = ROOT / "data/raw", ROOT / "data/yolo"
    for split, zname in SPLITS.items():
        unpacked = raw / "det" / split
        if not unpacked.exists():
            with zipfile.ZipFile(raw / zname) as z:
                z.extractall(unpacked)
        src = next(unpacked.rglob("images")).parent
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for img in sorted((src / "images").glob("*.jpg")):
            w, h = Image.open(img).size
            lines = det_to_yolo(src / "annotations" / f"{img.stem}.txt", w, h)
            link(img, out / split / "images" / img.name)
            (out / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_box += len(lines)
        print(f"{split}: {n_img} images, {n_box} vehicle boxes")
    (out / "vehicle.yaml").write_text(f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: vehicle\n")


if __name__ == "__main__":
    main()
