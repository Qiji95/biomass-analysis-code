from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pywt
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import remove_small_objects


def normalize(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    return ((values - low) / max(high - low, 1e-6)).astype(np.float32)


def clean_mask(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return remove_small_objects(result.astype(bool), min_size=min_size)


def exg_response(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    red, green, blue = cv2.split(image_rgb)
    response = normalize(cv2.GaussianBlur(2 * green - red - blue, (0, 0), sigmaX=1.0))
    try:
        threshold = float(threshold_multiotsu(response, classes=3)[0])
    except ValueError:
        threshold = float(threshold_otsu(response))
    return response, threshold


def wavelet_texture(image_bgr: np.ndarray) -> np.ndarray:
    if min(image_bgr.shape[:2]) < 60:
        raise ValueError("Image dimensions must be at least 60 pixels for a two-level db8 transform")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    red, green, blue = cv2.split(image_rgb)
    gray = 0.2125 * red + 0.7154 * green + 0.0721 * blue
    gray = cv2.bilateralFilter(gray.astype(np.uint8), 9, 75, 75).astype(np.float32) / 255.0
    _, (lh2, hl2, hh2), (lh1, hl1, hh1) = pywt.wavedec2(gray, "db8", level=2)
    height, width = gray.shape
    bands = [
        cv2.resize(np.abs(band), (width, height), interpolation=cv2.INTER_CUBIC)
        for band in (lh1, hl1, hh1, lh2, hl2, hh2)
    ]
    lh1, hl1, hh1, lh2, hl2, hh2 = bands
    texture = (
        0.6 * (0.4 * lh1 + 0.6 * lh2)
        + 0.2 * (0.4 * hl1 + 0.6 * hl2)
        + 0.2 * (0.4 * hh1 + 0.6 * hh2)
    )
    p5, p95 = np.percentile(texture, [5, 95])
    return np.clip((texture - p5) / max(float(p95 - p5), 1e-6), 0, 1).astype(np.float32)


def segment(
    image_bgr: np.ndarray,
    delta: float = 0.08,
    texture_cut: float = 0.20,
    diameter: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("Expected an 8-bit, three-channel BGR image")
    if not 0 <= delta <= 1 or not 0 <= texture_cut <= 1 or diameter < 1 or diameter % 2 != 1:
        raise ValueError("delta and texture_cut must be within [0, 1]; diameter must be positive and odd")
    exg, exg_cut = exg_response(image_bgr)
    base = clean_mask(exg > exg_cut)
    texture = wavelet_texture(image_bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    nearby = cv2.dilate(base.astype(np.uint8), kernel).astype(bool)
    additions = nearby & ~base & (exg > exg_cut - delta) & (texture > texture_cut)
    refined = base | clean_mask(base | additions)
    return base, refined


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_mask(path: Path, mask: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not success:
        raise OSError(f"Cannot encode mask: {path}")
    encoded.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.is_file():
        images = [args.input]
    elif args.input.is_dir():
        images = sorted(
            path for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
    else:
        parser.error("--input must be an image or a directory of images")
    if not images:
        parser.error("No supported images found")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in images:
        base, refined = segment(read_image(path))
        write_mask(args.output / f"{path.stem}_exg.png", base)
        write_mask(args.output / f"{path.stem}_exg_wavelet.png", refined)
        print(path.name)


if __name__ == "__main__":
    main()
