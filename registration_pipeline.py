"""Magnification-aware registration for paired SEM images.

The pipeline never overwrites source images.  It removes a likely SEM footer,
uses the magnification ratio as a scale prior, estimates a robust local affine
transform, refines it with ECC, and reports geometric and image-level quality.
"""

from __future__ import annotations

import json
import math
import re
import configparser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class RegistrationError(RuntimeError):
    """Raised when a pair cannot be registered with enough evidence."""


@dataclass
class RegistrationMetrics:
    detector: str
    before_magnification: Optional[float]
    after_magnification: Optional[float]
    target_magnification: Optional[float]
    before_pixel_size_um: Optional[float]
    after_pixel_size_um: Optional[float]
    target_pixel_size_um: Optional[float]
    scale_source: str
    scale_prior: float
    before_scale_factor: float
    after_scale_factor: float
    keypoints_before: int
    keypoints_after: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    coverage_ratio: float
    reprojection_rmse_px: float
    reprojection_median_px: float
    reprojection_p95_px: float
    overlap_ratio: float
    ncc_before: float
    ncc_after: float
    ecc_used: bool
    ecc_correlation: Optional[float]
    transform_scale: float
    rotation_deg: float
    quality: str
    warning: str


@dataclass
class RegistrationResult:
    before_crop: np.ndarray
    after_crop: np.ndarray
    after_registered: np.ndarray
    overlap_mask: np.ndarray
    before_core: np.ndarray
    after_core: np.ndarray
    overlap_core: np.ndarray
    diff_core: np.ndarray
    transform_after_to_before: np.ndarray
    before_crop_bounds_xyxy: tuple[int, int, int, int]
    after_crop_bounds_xyxy: tuple[int, int, int, int]
    core_bounds_xyxy: tuple[int, int, int, int]
    metrics: RegistrationMetrics


def read_image_unicode(path: str | Path, grayscale: bool = True) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    # Do not use IMREAD_GRAYSCALE here: OpenCV converts 16-bit SEM TIFFs to
    # uint8 with that flag.  The source TIFF stays untouched, but that would
    # silently throw away the intensity precision used by downstream outputs.
    flag = cv2.IMREAD_UNCHANGED if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flag)
    if image is None:
        raise FileNotFoundError(path)
    if grayscale and image.ndim == 3:
        # Keep the original channel depth while reducing unusual RGB/RGBA
        # TIFF exports to a single channel.
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    return image


def write_png_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"无法写入图像：{path}")
    encoded.tofile(str(path))


def infer_magnification(value: str | Path) -> Optional[float]:
    """Infer SEM magnification from strings such as 500x, 2kx, x2000 or 2000倍."""
    text = str(value).lower().replace("×", "x").replace("，", ",")
    patterns = (
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*k\s*x(?![a-z0-9])",
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*k(?![a-z0-9])",
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*x(?![a-z0-9])",
        r"(?<![a-z0-9])x\s*(\d+(?:\.\d+)?)(?![\d.])",
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*倍",
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if index < 2:
                value *= 1000.0
            return value if value > 0 else None
    return None


def parse_magnification(value: object, path: str | Path = "") -> Optional[float]:
    if value is None:
        return infer_magnification(path)
    text = str(value).strip().lower()
    if not text or text in {"auto", "自动", "none", "unknown", "?"}:
        return infer_magnification(path)
    inferred = infer_magnification(text)
    if inferred is not None:
        return inferred
    try:
        number = float(text.replace(",", ""))
    except ValueError as exc:
        raise RegistrationError(f"无法识别倍率：{value}") from exc
    if number <= 0:
        raise RegistrationError("倍率必须大于0")
    return number


def find_sem_header(image_path: str | Path) -> Optional[Path]:
    """Find the TESCAN header stored beside an image without scanning other folders."""
    image_path = Path(image_path)
    candidates = [
        image_path.with_name(f"{image_path.stem}-tif.hdr"),
        image_path.with_suffix(".hdr"),
        image_path.with_name(f"{image_path.name}.hdr"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    lower_names = {item.name.lower(): item for item in image_path.parent.glob("*.hdr")}
    for candidate in candidates:
        found = lower_names.get(candidate.name.lower())
        if found is not None:
            return found
    return None


def _sem_values_from_text(value: object) -> dict:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    elif isinstance(value, (tuple, list)):
        text = "\n".join(
            item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else str(item)
            for item in value
        )
    else:
        text = str(value)

    def number(name: str) -> Optional[float]:
        match = re.search(
            rf"(?:^|[\r\n\x00]){re.escape(name)}\s*=\s*"
            rf"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            text,
        )
        return float(match.group(1)) if match else None

    return {
        "magnification": number("Magnification"),
        "pixel_size_x_m": number("PixelSizeX"),
        "pixel_size_y_m": number("PixelSizeY"),
    }


def read_tiff_embedded_sem_metadata(image_path: str | Path) -> dict:
    """Read TESCAN key/value metadata embedded in TIFF tag 50431/description."""
    empty = {"magnification": None, "pixel_size_x_um": None, "pixel_size_y_um": None}
    path = Path(image_path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return empty
    try:
        with Image.open(path) as image:
            values = []
            for tag in (50431, 270):
                value = image.tag_v2.get(tag)
                if value is not None:
                    values.append(value)
    except (OSError, ValueError):
        return empty
    parsed = _sem_values_from_text(values)
    pixel_x_m = parsed["pixel_size_x_m"]
    pixel_y_m = parsed["pixel_size_y_m"]
    return {
        "magnification": parsed["magnification"],
        "pixel_size_x_um": pixel_x_m * 1_000_000.0 if pixel_x_m else None,
        "pixel_size_y_um": pixel_y_m * 1_000_000.0 if pixel_y_m else None,
    }


def read_sem_metadata(image_path: str | Path) -> dict:
    """Read SEM scale from embedded TIFF metadata, then fill gaps from sibling .hdr."""
    header = find_sem_header(image_path)
    embedded = read_tiff_embedded_sem_metadata(image_path)
    result = {
        "header_path": str(header) if header else None,
        "metadata_source": "TIFF内嵌" if any(value is not None for value in embedded.values()) else None,
        **embedded,
    }
    if header is None:
        return result
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(header, encoding="utf-8-sig")
        main = parser["MAIN"]
        header_magnification = main.getfloat("Magnification", fallback=None)
        pixel_x_m = main.getfloat("PixelSizeX", fallback=None)
        pixel_y_m = main.getfloat("PixelSizeY", fallback=None)
        header_values = {
            "magnification": header_magnification,
            "pixel_size_x_um": pixel_x_m * 1_000_000.0 if pixel_x_m else None,
            "pixel_size_y_um": pixel_y_m * 1_000_000.0 if pixel_y_m else None,
        }
        filled = False
        for key, value in header_values.items():
            if result[key] is None and value is not None:
                result[key] = value
                filled = True
        if result["metadata_source"] is None and any(value is not None for value in header_values.values()):
            result["metadata_source"] = ".hdr"
        elif filled:
            result["metadata_source"] = "TIFF内嵌 + .hdr"
    except (OSError, configparser.Error, ValueError):
        pass
    return result


def crop_sem_content(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Remove the common bottom SEM information band without resizing pixels.

    Portrait/square SEM exports commonly place metadata below a square field.
    For landscape images the full frame is retained because the location of a
    footer cannot be inferred safely from geometry alone.
    """
    if image is None or image.size == 0:
        raise RegistrationError("输入图像为空")
    h, w = image.shape[:2]
    if h > w:
        return image[:w, :w].copy(), (0, 0, w, w)
    return image.copy(), (0, 0, w, h)


def normalize_registration_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)


def prepare_sam_roi(image: np.ndarray) -> np.ndarray:
    """Contrast-normalize a local SEM crop for promptable segmentation."""
    return normalize_registration_image(image)


def _detect_and_match(reference: np.ndarray, moving: np.ndarray):
    ref = normalize_registration_image(reference)
    mov = normalize_registration_image(moving)
    if hasattr(cv2, "SIFT_create"):
        detector_name = "SIFT"
        detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015, edgeThreshold=15)
        norm = cv2.NORM_L2
        ratio = 0.76
    else:
        detector_name = "ORB"
        detector = cv2.ORB_create(nfeatures=10000, fastThreshold=8)
        norm = cv2.NORM_HAMMING
        ratio = 0.82
    kp_ref, des_ref = detector.detectAndCompute(ref, None)
    kp_mov, des_mov = detector.detectAndCompute(mov, None)
    if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
        raise RegistrationError("可重复特征太少，无法配准")
    pairs = cv2.BFMatcher(norm).knnMatch(des_mov, des_ref, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    # A reference keypoint can only support one independent correspondence.
    seen = set()
    good = [m for m in sorted(good, key=lambda m: m.distance)
            if not (m.trainIdx in seen or seen.add(m.trainIdx))]
    return detector_name, kp_ref, kp_mov, good


def _grid_coverage(points: np.ndarray, shape: tuple[int, int], grid: int = 4) -> float:
    if len(points) == 0:
        return 0.0
    h, w = shape
    gx = np.clip((points[:, 0] / max(w, 1) * grid).astype(int), 0, grid - 1)
    gy = np.clip((points[:, 1] / max(h, 1) * grid).astype(int), 0, grid - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(grid * grid)


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if int(valid.sum()) < 64:
        return -1.0
    av = normalize_registration_image(a)[valid].astype(np.float32)
    bv = normalize_registration_image(b)[valid].astype(np.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 1e-9 else 0.0


def _warp(image: np.ndarray, matrix: np.ndarray, shape: tuple[int, int], interpolation=cv2.INTER_LINEAR):
    h, w = shape
    return cv2.warpAffine(
        image, matrix[:2].astype(np.float32), (w, h), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest all-true axis-aligned rectangle as x0,y0,x1,y1."""
    binary = mask.astype(bool)
    h, w = binary.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)
    for y in range(h):
        heights = np.where(binary[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(w + 1):
            height = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > height:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, y - sh + 1, x, y + 1)
                start = sx
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


def _phase_translation(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    if reference.shape != moving.shape:
        raise RegistrationError("特征配准失败，且不同尺寸图像不能安全使用平移回退")
    ref = normalize_registration_image(reference).astype(np.float32)
    mov = normalize_registration_image(moving).astype(np.float32)
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(mov, ref, window)
    if not np.isfinite([dx, dy, response]).all() or response < 0.04:
        raise RegistrationError("特征不足且相位相关置信度过低")
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def register_arrays(
    before: np.ndarray,
    after: np.ndarray,
    before_magnification: Optional[float] = None,
    after_magnification: Optional[float] = None,
    before_pixel_size_um: Optional[float] = None,
    after_pixel_size_um: Optional[float] = None,
    use_ecc: bool = True,
    matcher: str = 'classic',
    manual_points=None,
    crop_footer: bool = True,
    before_content_bounds=None,
    after_content_bounds=None,
) -> RegistrationResult:
    if crop_footer:
        before_original, before_crop_bounds = crop_sem_content(before)
        after_crop, after_crop_bounds = crop_sem_content(after)
    else:
        before_original, before_crop_bounds = before.copy(), (0,0,before.shape[1],before.shape[0])
        after_crop, after_crop_bounds = after.copy(), (0,0,after.shape[1],after.shape[0])
    for side, bounds in enumerate((before_content_bounds, after_content_bounds)):
        if bounds is not None:
            raw = before if side == 0 else after
            x0, y0, x1, y1 = map(int, bounds)
            if not (0 <= x0 < x1 <= raw.shape[1] and 0 <= y0 < y1 <= raw.shape[0]):
                raise RegistrationError('成像区裁剪范围超出原图')
            cropped = raw[y0:y1, x0:x1].copy()
            if side == 0:
                before_original, before_crop_bounds = cropped, (x0,y0,x1,y1)
            else:
                after_crop, after_crop_bounds = cropped, (x0,y0,x1,y1)
    if before_original.ndim == 3:
        before_original = cv2.cvtColor(before_original, cv2.COLOR_BGR2GRAY)
    if after_crop.ndim == 3:
        after_crop = cv2.cvtColor(after_crop, cv2.COLOR_BGR2GRAY)
    before_original = np.ascontiguousarray(before_original)
    after_crop = np.ascontiguousarray(after_crop)
    if min(before_original.shape[:2] + after_crop.shape[:2]) < 64:
        raise RegistrationError("图像尺寸太小，无法可靠配准")

    if before_pixel_size_um and after_pixel_size_um:
        if before_pixel_size_um <= 0 or after_pixel_size_um <= 0:
            raise RegistrationError("物理像素尺寸必须大于0")
        scale_source = "PixelSizeX"
        target_pixel_size_um = max(float(before_pixel_size_um), float(after_pixel_size_um))
        before_factor = float(before_pixel_size_um) / target_pixel_size_um
        after_factor = float(after_pixel_size_um) / target_pixel_size_um
        scale_prior = float(after_pixel_size_um) / float(before_pixel_size_um)
        target_magnification = (
            min(float(before_magnification), float(after_magnification))
            if before_magnification and after_magnification else None
        )
    elif before_magnification and after_magnification:
        scale_source = "Magnification"
        target_pixel_size_um = None
        scale_prior = float(before_magnification) / float(after_magnification)
        if not 0.05 <= scale_prior <= 20.0:
            raise RegistrationError(f"倍率差过大（尺度比={scale_prior:.4g}），请核对输入")
        # Work at the lower physical sampling density.  This avoids inventing
        # detail by upsampling a 500x image to a 2000x reference grid.
        target_magnification = min(float(before_magnification), float(after_magnification))
        before_factor = target_magnification / float(before_magnification)
        after_factor = target_magnification / float(after_magnification)
    else:
        scale_source = "none"
        target_pixel_size_um = None
        scale_prior = 1.0
        target_magnification = None
        before_factor = after_factor = 1.0
    before_crop = cv2.resize(
        before_original, None, fx=before_factor, fy=before_factor, interpolation=cv2.INTER_AREA
    ) if before_factor < 1.0 - 1e-6 else before_original
    mov_scaled = cv2.resize(
        after_crop, None, fx=after_factor, fy=after_factor, interpolation=cv2.INTER_AREA
    ) if after_factor < 1.0 - 1e-6 else after_crop

    # OpenCV resize uses centre-based sampling; record the corresponding affine
    # mapping, including the half-pixel offset, rather than only a scale scalar.
    def scale_matrix(factor):
        return np.array([[factor, 0, (factor-1)/2], [0, factor, (factor-1)/2], [0,0,1]], np.float64)
    before_sampling = scale_matrix(before_factor)
    after_sampling = scale_matrix(after_factor)

    detector_name = "phase-correlation"
    kp_ref = kp_mov = []
    good = []
    inlier_flags = np.zeros(0, dtype=bool)
    src_scaled = dst = np.empty((0, 2), dtype=np.float32)
    try:
        if matcher == 'manual':
            if manual_points is None or len(manual_points.get('fit', [])) < 6 or len(manual_points.get('check', [])) < 3:
                raise RegistrationError('人工辅助需要至少6对拟合点和3对独立检查点')
            pairs = np.asarray(manual_points['fit'], np.float32)
            if pairs.shape[1:] != (2, 2) or not np.isfinite(pairs).all():
                raise RegistrationError('人工点坐标无效')
            checks_raw = np.asarray(manual_points['check'], np.float64)
            if checks_raw.ndim != 3 or checks_raw.shape[1:] != (2,2) or not np.isfinite(checks_raw).all():
                raise RegistrationError('独立检查点无效')
            for points in (pairs, checks_raw):
                for side, image in enumerate((before_original, after_crop)):
                    xy = points[:,side]
                    if np.any(xy < 0) or np.any(xy[:,0] >= image.shape[1]) or np.any(xy[:,1] >= image.shape[0]):
                        raise RegistrationError('人工点超出裁剪图像范围')
            if np.any(np.linalg.norm(checks_raw[:,None,0]-pairs[None,:,0],axis=2)<1):
                raise RegistrationError('检查点不能重复使用拟合点，请选择独立位置')
            ref_points = cv2.transform(pairs[:,0,None], before_sampling[:2])[:,0]
            mov_points = cv2.transform(pairs[:,1,None], after_sampling[:2])[:,0]
            if _grid_coverage(ref_points, before_crop.shape) < .125 or np.linalg.matrix_rank(ref_points-ref_points.mean(0)) < 2:
                raise RegistrationError('人工点过于集中或共线，请分散选择稳定结构')
            detector_name = 'manual-landmarks'
            kp_ref = [cv2.KeyPoint(float(x),float(y),1) for x,y in ref_points]
            kp_mov = [cv2.KeyPoint(float(x),float(y),1) for x,y in mov_points]
            good = [cv2.DMatch(i,i,0) for i in range(len(pairs))]
        elif matcher == 'lightglue':
            from enhanced_matching import detect_lightglue
            detector_name, kp_ref, kp_mov, good = detect_lightglue(before_crop, mov_scaled)
        else:
            detector_name, kp_ref, kp_mov, good = _detect_and_match(before_crop, mov_scaled)
        if len(good) < 6:
            raise RegistrationError(f"可靠特征匹配不足：{len(good)} < 6")
        src_scaled = np.float32([kp_mov[m.queryIdx].pt for m in good])
        dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
        ransac_px = float(np.clip(min(before_crop.shape[:2]) * 0.003, 1.5, 5.0))
        affine, inliers = cv2.estimateAffinePartial2D(
            src_scaled, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
            maxIters=10000, confidence=0.999, refineIters=25,
        )
        if affine is None or inliers is None:
            raise RegistrationError("RANSAC未找到稳定变换")
        inlier_flags = inliers.ravel().astype(bool)
        if int(inlier_flags.sum()) < 6:
            raise RegistrationError(f"RANSAC内点不足：{int(inlier_flags.sum())} < 6")
        affine_h = np.vstack([affine, [0.0, 0.0, 1.0]])
        total = affine_h @ after_sampling
    except RegistrationError:
        if matcher != 'classic':
            raise
        same_scale = abs(before_factor - 1.0) < 1e-6 and abs(after_factor - 1.0) < 1e-6
        if not same_scale or before_crop.shape != after_crop.shape:
            raise
        total = _phase_translation(before_crop, after_crop)
        # Failed feature evidence does not validate the phase-only transform.
        detector_name = 'phase-correlation'
        kp_ref = kp_mov = []
        good = []
        inlier_flags = np.zeros(0, bool)
        src_scaled = dst = np.empty((0,2), np.float32)

    ref_shape = before_crop.shape[:2]
    registered = _warp(after_crop, total, ref_shape)
    overlap = _warp(np.ones(after_crop.shape[:2], np.uint8), total, ref_shape, cv2.INTER_NEAREST) > 0
    ncc_initial = _ncc(before_crop, registered, overlap)
    ecc_used = False
    ecc_corr: Optional[float] = None
    ecc_validation = {'attempted': False, 'accepted': False}
    def point_errors(matrix):
        if not len(src_scaled) or not inlier_flags.any():
            return np.empty(0)
        original = cv2.transform(src_scaled[:,None], np.linalg.inv(after_sampling)[:2])[:,0]
        predicted = cv2.transform(original[:,None], matrix[:2])[:,0]
        return np.linalg.norm(predicted-dst, axis=1)[inlier_flags]
    if use_ecc and int(overlap.sum()) >= 4096:
        ecc_validation['attempted'] = True
        ref_norm = normalize_registration_image(before_crop).astype(np.float32) / 255.0
        reg_norm = normalize_registration_image(registered).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
        try:
            corr, warp = cv2.findTransformECC(
                ref_norm, reg_norm, warp, cv2.MOTION_EUCLIDEAN, criteria,
                inputMask=(overlap.astype(np.uint8) * 255), gaussFiltSize=5,
            )
            warp_h = np.vstack([warp, [0.0, 0.0, 1.0]]).astype(np.float64)
            refined_total = np.linalg.inv(warp_h) @ total
            refined = _warp(after_crop, refined_total, ref_shape)
            refined_overlap = _warp(
                np.ones(after_crop.shape[:2], np.uint8), refined_total, ref_shape, cv2.INTER_NEAREST
            ) > 0
            refined_ncc = _ncc(before_crop, refined, refined_overlap)
            old_error, new_error = point_errors(total), point_errors(refined_total)
            geometric_ok = True
            if len(old_error):
                old_m, old_p = np.median(old_error), np.percentile(old_error,95)
                new_m, new_p = np.median(new_error), np.percentile(new_error,95)
                geometric_ok = new_m <= old_m + .25 and new_p <= old_p + .5
                ecc_validation.update(coarse_median_px=float(old_m), refined_median_px=float(new_m),
                                      coarse_p95_px=float(old_p), refined_p95_px=float(new_p))
            overlap_ok = refined_overlap.mean() >= overlap.mean() - .02
            ecc_validation.update(geometry_ok=bool(geometric_ok), overlap_ok=bool(overlap_ok))
            if geometric_ok and overlap_ok and refined_ncc >= ncc_initial - 0.005:
                total, registered, overlap = refined_total, refined, refined_overlap
                ecc_used, ecc_corr = True, float(corr)
                ecc_validation['accepted'] = True
        except (cv2.error, np.linalg.LinAlgError) as exc:
            ecc_validation['error'] = str(exc)

    # Geometric residuals use the feature correspondences, if available.
    if len(src_scaled):
        src_original = cv2.transform(src_scaled[:,None], np.linalg.inv(after_sampling)[:2])[:,0]
        predicted = cv2.transform(src_original.reshape(-1, 1, 2), total[:2].astype(np.float64)).reshape(-1, 2)
        errors = np.linalg.norm(predicted - dst, axis=1)[inlier_flags]
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        median = float(np.median(errors))
        p95 = float(np.percentile(errors, 95))
        coverage = _grid_coverage(dst[inlier_flags], ref_shape)
    else:
        rmse = median = p95 = float("nan")
        coverage = 0.0

    # Remove a thin interpolation boundary before selecting an all-valid core.
    eroded = cv2.erode(overlap.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    bounds = largest_valid_rectangle(eroded)
    x0, y0, x1, y1 = bounds
    if (x1 - x0) * (y1 - y0) < 4096:
        raise RegistrationError("有效重叠区域太小，不能生成可靠的变化图")
    before_core = before_crop[y0:y1, x0:x1].copy()
    after_core = registered[y0:y1, x0:x1].copy()
    overlap_core = overlap[y0:y1, x0:x1].copy()
    diff_core = cv2.absdiff(normalize_registration_image(before_core), normalize_registration_image(after_core))
    ncc_after = _ncc(before_crop, registered, overlap)
    overlap_ratio = float(overlap.mean())
    a, b = float(total[0, 0]), float(total[1, 0])
    transform_scale = math.hypot(a, b)
    rotation_deg = math.degrees(math.atan2(b, a))
    inlier_count = int(inlier_flags.sum())
    inlier_ratio = inlier_count / max(len(good), 1)

    if detector_name == "phase-correlation":
        quality = "需复核"
        warning = "仅使用平移回退；没有可报告的独立特征重投影误差"
    elif inlier_count >= 20 and median <= 1.0 and p95 <= 3.0 and coverage >= 0.25:
        quality = "优秀"
        warning = ""
    elif inlier_count >= 10 and median <= 2.0 and p95 <= 5.0 and coverage >= 0.125:
        quality = "良好"
        warning = ""
    else:
        quality = "需复核"
        warning = "特征误差或空间覆盖未达到自动通过阈值"
    if ncc_after < 0.15:
        quality = "需复核"
        warning = (warning + "；" if warning else "") + "配准后纹理相关性偏低"
    if before_magnification and after_magnification:
        magnification_ratio = max(before_magnification, after_magnification) / min(before_magnification, after_magnification)
        if magnification_ratio > 1.08:
            warning = (warning + "；" if warning else "") + "跨倍率结果只适合共同视野定位，不宜直接定量像素变化"
        elif before_pixel_size_um and after_pixel_size_um:
            pixel_ratio = max(before_pixel_size_um, after_pixel_size_um) / min(before_pixel_size_um, after_pixel_size_um)
            if pixel_ratio > 1.02:
                warning = (warning + "；" if warning else "") + "同倍率但物理像素尺寸不同，已统一到较粗像素网格"

    validation = None
    if matcher == 'manual':
        checks = np.asarray(manual_points['check'], np.float64)
        if checks.shape[1:] != (2,2) or not np.isfinite(checks).all():
            raise RegistrationError('独立检查点无效')
        predicted = cv2.transform(checks[:,1].reshape(-1,1,2), total[:2]).reshape(-1,2)
        reference_checks = cv2.transform(checks[:,0,None], before_sampling[:2])[:,0]
        check_errors = np.linalg.norm(predicted-reference_checks, axis=1)
        validation = {'points_cropped_original_xy': manual_points, 'check_errors_px': check_errors.tolist(),
                      'check_median_px': float(np.median(check_errors)), 'check_p95_px': float(np.percentile(check_errors,95))}
        passed = (inlier_count >= 6 and coverage >= .125 and median <= 2 and p95 <= 5 and
                  validation['check_median_px'] <= 2 and validation['check_p95_px'] <= 5 and ncc_after >= .15)
        quality = '良好' if passed else '需复核'
        warning = '人工辅助：包含独立检查点验证；仍需检查棋盘格及目标附近边界' if passed else '人工辅助拟合/独立检查误差或覆盖未通过'

    metrics = RegistrationMetrics(
        detector=detector_name,
        before_magnification=before_magnification,
        after_magnification=after_magnification,
        target_magnification=target_magnification,
        before_pixel_size_um=before_pixel_size_um,
        after_pixel_size_um=after_pixel_size_um,
        target_pixel_size_um=target_pixel_size_um,
        scale_source=scale_source,
        scale_prior=scale_prior,
        before_scale_factor=before_factor,
        after_scale_factor=after_factor,
        keypoints_before=len(kp_ref),
        keypoints_after=len(kp_mov),
        good_matches=len(good),
        inliers=inlier_count,
        inlier_ratio=float(inlier_ratio),
        coverage_ratio=float(coverage),
        reprojection_rmse_px=rmse,
        reprojection_median_px=median,
        reprojection_p95_px=p95,
        overlap_ratio=overlap_ratio,
        ncc_before=float(ncc_initial),
        ncc_after=float(ncc_after),
        ecc_used=ecc_used,
        ecc_correlation=ecc_corr,
        transform_scale=transform_scale,
        rotation_deg=rotation_deg,
        quality=quality,
        warning=warning,
    )
    result = RegistrationResult(
        before_crop=before_crop,
        after_crop=after_crop,
        after_registered=registered,
        overlap_mask=overlap,
        before_core=before_core,
        after_core=after_core,
        overlap_core=overlap_core,
        diff_core=diff_core,
        transform_after_to_before=total,
        before_crop_bounds_xyxy=before_crop_bounds,
        after_crop_bounds_xyxy=after_crop_bounds,
        core_bounds_xyxy=bounds,
        metrics=metrics,
    )
    result.manual_validation = validation
    result.ecc_validation = ecc_validation
    result.before_sampling = before_sampling
    result.after_sampling = after_sampling
    return result


def register_pair(
    before_path: str | Path,
    after_path: str | Path,
    before_magnification: object = None,
    after_magnification: object = None,
    use_ecc: bool = True,
    matcher: str = 'classic',
    manual_points=None,
) -> RegistrationResult:
    before_meta = read_sem_metadata(before_path)
    after_meta = read_sem_metadata(after_path)
    before_auto = before_magnification is None or str(before_magnification).strip().lower() in {"", "auto", "自动", "none", "unknown", "?"}
    after_auto = after_magnification is None or str(after_magnification).strip().lower() in {"", "auto", "自动", "none", "unknown", "?"}
    before_mag = before_meta["magnification"] if before_auto else parse_magnification(before_magnification, before_path)
    after_mag = after_meta["magnification"] if after_auto else parse_magnification(after_magnification, after_path)
    if before_mag is None:
        before_mag = infer_magnification(before_path)
    if after_mag is None:
        after_mag = infer_magnification(after_path)
    return register_arrays(
        read_image_unicode(before_path), read_image_unicode(after_path),
        before_mag, after_mag,
        before_pixel_size_um=before_meta["pixel_size_x_um"],
        after_pixel_size_um=after_meta["pixel_size_x_um"],
        use_ecc=use_ecc,
        matcher=matcher, manual_points=manual_points,
    )


def save_registration_result(result: RegistrationResult, output_dir: str | Path, filenames=None) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "before": output_dir / "before_core.png",
        "after": output_dir / "after_registered_core.png",
        "diff": output_dir / "difference_core.png",
        "mask": output_dir / "change_mask_initial.png",
        "overlap": output_dir / "overlap_mask.png",
        "checkerboard": output_dir / "registration_checkerboard.png",
        "overlay": output_dir / "registration_overlay.png",
        "metrics": output_dir / "registration_metrics.json",
        "transform": output_dir / "transform_after_to_before.txt",
    }
    if filenames:
        if set(filenames) - set(files):
            raise ValueError('未知的配准输出文件类别')
        for key, name in filenames.items():
            if Path(name).name != name:
                raise ValueError('文件名不能包含目录')
            files[key] = output_dir / name
    checker = normalize_registration_image(result.before_core)
    checker_after = normalize_registration_image(result.after_core)
    tile = 64
    yy, xx = np.indices(checker.shape)
    choose_after = ((xx // tile + yy // tile) % 2) == 1
    checker[choose_after] = checker_after[choose_after]
    overlay = cv2.addWeighted(
        normalize_registration_image(result.before_core), 0.5,
        normalize_registration_image(result.after_core), 0.5, 0,
    )
    write_png_unicode(files["before"], result.before_core)
    write_png_unicode(files["after"], result.after_core)
    write_png_unicode(files["diff"], result.diff_core)
    write_png_unicode(files["mask"], np.zeros(result.before_core.shape, np.uint8))
    write_png_unicode(files["overlap"], result.overlap_core.astype(np.uint8) * 255)
    write_png_unicode(files["checkerboard"], checker)
    write_png_unicode(files["overlay"], overlay)
    metrics = asdict(result.metrics)
    metrics = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metrics.items()
    }
    metrics["before_crop_bounds_xyxy"] = list(result.before_crop_bounds_xyxy)
    metrics["after_crop_bounds_xyxy"] = list(result.after_crop_bounds_xyxy)
    metrics["core_bounds_xyxy"] = list(result.core_bounds_xyxy)
    metrics["core_shape_hw"] = list(result.before_core.shape[:2])
    metrics["before_core_dtype"] = str(result.before_core.dtype)
    metrics["after_core_dtype"] = str(result.after_core.dtype)
    metrics["source_precision_note"] = (
        "before_core/after_registered_core保留输入数组位深；"
        "registration_overlay/difference_core为8位显示预览；原始TIF由source_pair.json引用且未改写"
    )
    metrics['manual_validation'] = getattr(result, 'manual_validation', None)
    metrics['ecc_validation'] = getattr(result, 'ecc_validation', None)
    metrics['before_sampling_matrix'] = getattr(result, 'before_sampling', np.eye(3)).tolist()
    metrics['after_sampling_matrix'] = getattr(result, 'after_sampling', np.eye(3)).tolist()
    metrics['error_basis'] = '拟合匹配点在最终共同网格上的残差；不是独立人工真值误差'
    files["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savetxt(files["transform"], result.transform_after_to_before, fmt="%.12g")
    return {key: str(value) for key, value in files.items()}
