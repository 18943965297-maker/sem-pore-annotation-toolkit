"""Reusable OpenCV preprocessing and classical segmentation helpers."""

from __future__ import annotations

import cv2
import numpy as np


def _u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if arr.dtype == np.uint8:
        return arr
    lo, hi = np.percentile(arr.astype(np.float32), [1, 99])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def crop_image(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Crop with coordinates clamped to the image boundary."""
    if image is None or np.asarray(image).ndim < 2:
        raise ValueError("image must have at least two dimensions")
    h, w = image.shape[:2]
    left, right = sorted((int(x0), int(x1)))
    top, bottom = sorted((int(y0), int(y1)))
    left, right = max(0, left), min(w, right)
    top, bottom = max(0, top), min(h, bottom)
    if right <= left or bottom <= top:
        raise ValueError("crop does not intersect the image")
    return np.ascontiguousarray(image[top:bottom, left:right])


def to_grayscale(image: np.ndarray, color_order: str = "bgr") -> np.ndarray:
    """Convert a grayscale, BGR/RGB or BGRA/RGBA image to uint8 grayscale."""
    arr = _u8(image)
    if arr.ndim == 2:
        return arr.copy()
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape: {arr.shape}")
    order = color_order.lower()
    if order not in {"bgr", "rgb"}:
        raise ValueError("color_order must be 'bgr' or 'rgb'")
    if arr.shape[2] == 3:
        code = cv2.COLOR_BGR2GRAY if order == "bgr" else cv2.COLOR_RGB2GRAY
    else:
        code = cv2.COLOR_BGRA2GRAY if order == "bgr" else cv2.COLOR_RGBA2GRAY
    return cv2.cvtColor(arr, code)


def denoise_image(image: np.ndarray, method: str = "median", kernel_size: int = 5) -> np.ndarray:
    """Denoise a grayscale image with median, Gaussian or bilateral filtering."""
    gray = to_grayscale(image)
    k = max(3, int(kernel_size) | 1)
    if method == "median":
        return cv2.medianBlur(gray, k)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (k, k), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, k, 35, 35)
    raise ValueError("method must be median, gaussian or bilateral")


def enhance_image(image: np.ndarray, amount: float = 1.0, sigma: float = 1.2) -> np.ndarray:
    """Enhance local detail with a controlled unsharp mask."""
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), max(0.1, float(sigma)))
    return cv2.addWeighted(gray, 1.0 + float(amount), blurred, -float(amount), 0)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization."""
    gray = to_grayscale(image)
    grid = tuple(max(1, int(v)) for v in tile_grid_size)
    return cv2.createCLAHE(clipLimit=max(0.01, float(clip_limit)), tileGridSize=grid).apply(gray)


def otsu_segment(image: np.ndarray, invert: bool = False) -> tuple[float, np.ndarray]:
    """Return the Otsu threshold and a 0/255 binary image."""
    gray = to_grayscale(image)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold, binary = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    return float(threshold), binary


def canny_edges(
    image: np.ndarray, lower: int | None = None, upper: int | None = None, sigma: float = 0.33
) -> tuple[int, int, np.ndarray]:
    """Detect edges; omitted thresholds are derived robustly from the median intensity."""
    gray = to_grayscale(image)
    median = float(np.median(gray))
    if lower is None:
        lower = int(max(0, (1.0 - sigma) * median))
    if upper is None:
        upper = int(min(255, (1.0 + sigma) * median))
    lower, upper = max(0, int(lower)), min(255, int(upper))
    if upper <= lower:
        upper = min(255, lower + 1)
    return lower, upper, cv2.Canny(gray, lower, upper, L2gradient=True)


def morphology_open_close(
    binary: np.ndarray,
    open_size: int = 3,
    close_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """Remove small specks and close narrow gaps in a binary image."""
    mask = (np.asarray(binary) > 0).astype(np.uint8) * 255
    iterations = max(1, int(iterations))
    if open_size > 1:
        k = max(3, int(open_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if close_size > 1:
        k = max(3, int(close_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return mask


def connected_components_analysis(
    binary: np.ndarray, min_area: int = 1, connectivity: int = 8
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Label components, filter by area, and return serializable measurements."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = (np.asarray(binary) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    components: list[dict[str, object]] = []
    min_area = max(1, int(min_area))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        filtered[labels == label] = 255
        components.append({
            "label": label,
            "area": area,
            "bbox": (x, y, w, h),
            "centroid": (float(centroids[label, 0]), float(centroids[label, 1])),
        })
    components.sort(key=lambda item: int(item["area"]), reverse=True)
    return filtered, labels, components


def process_roi(
    image: np.ndarray,
    *,
    dark_foreground: bool = True,
    min_component_area: int = 20,
) -> dict[str, object]:
    """Run the complete grayscale-to-components processing pipeline."""
    gray = to_grayscale(image)
    denoised = denoise_image(gray, method="median", kernel_size=5)
    enhanced = enhance_image(denoised)
    clahe = apply_clahe(enhanced)
    threshold, otsu = otsu_segment(clahe, invert=dark_foreground)
    low, high, edges = canny_edges(clahe)
    morphology = morphology_open_close(otsu)
    components_mask, labels, components = connected_components_analysis(
        morphology, min_area=min_component_area
    )
    return {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "clahe": clahe,
        "otsu": otsu,
        "otsu_threshold": threshold,
        "canny": edges,
        "canny_thresholds": (low, high),
        "morphology": morphology,
        "components_mask": components_mask,
        "labels": labels,
        "components": components,
    }
