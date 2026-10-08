import csv
import datetime as dt
import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from image_processing import crop_image, process_roi
from registration_pipeline import (
    RegistrationError,
    parse_magnification,
    prepare_sam_roi,
    read_sem_metadata,
    register_pair,
    save_registration_result,
)
from sam_client import SamApiError, SamClient
from annotation_modes import AnnotationLayers, MODES, CLASSES, pore_colors, label_colors, binary_overlay_colors
from dataset_paths import resolve_row, portable_row


DEFAULT_INDEX = Path(__file__).resolve().parent / 'annotation_index.csv'
DEFAULT_MANUAL_DIR = Path(__file__).resolve().parent / 'manual_masks'
ANNOTATION_LOG = DEFAULT_MANUAL_DIR / "manual_annotation_log.csv"
REGISTRATION_RUNS_DIR = Path(__file__).resolve().parent / "registration_runs"


def read_image(path, grayscale=True):
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    # Preserve uint16 SEM TIFF intensities; normalize_u8() is applied only for
    # display/SAM input, never by the source reader itself.
    flag = cv2.IMREAD_UNCHANGED if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flag)
    if image is None:
        raise FileNotFoundError(path)
    if grayscale and image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")
    data.tofile(str(path))


def normalize_u8(image):
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def align_shape(*images):
    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)
    return [img[:h, :w] for img in images]


def make_mask_overlay(gray, mask, alpha=0.45):
    base = cv2.cvtColor(normalize_u8(gray), cv2.COLOR_GRAY2RGB)
    layer = base.copy()
    layer[mask] = (255, 0, 0)
    return cv2.addWeighted(base, 1.0 - alpha, layer, alpha, 0)


def compare_masks(auto_mask, manual_mask):
    auto = auto_mask.astype(bool)
    manual = manual_mask.astype(bool)
    tp = int(np.logical_and(auto, manual).sum())
    fp = int(np.logical_and(~auto, manual).sum())
    fn = int(np.logical_and(auto, ~manual).sum())
    auto_area = int(auto.sum())
    manual_area = int(manual.sum())
    eps = 1e-9
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "auto_area_px": auto_area,
        "manual_area_px": manual_area,
        "added_area_px": fp,
        "removed_area_px": fn,
        "area_delta_px": manual_area - auto_area,
        "area_delta_percent_of_auto": (manual_area - auto_area) / (auto_area + eps) * 100.0,
        "manual_vs_auto_iou": iou,
        "manual_vs_auto_dice": dice,
    }


def make_mask_view(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = (255, 255, 255)
    return out


def encode_png(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("图像编码失败")
    return data.tobytes()


def geojson_to_mask(features, shape):
    """Sample polygons at integer pixel centres; never round contour vertices.

    SAM's marching-squares boundary lies between these centres. Scanline
    intervals use a half-open convention for centres exactly on a boundary.
    """
    mask = np.zeros(shape, dtype=bool)

    def raster_ring(coordinates):
        result = np.zeros(shape, dtype=bool)
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
            return result
        a, b = points, np.roll(points, -1, axis=0)
        low = max(0, int(np.ceil(points[:, 1].min())))
        high = min(shape[0], int(np.ceil(points[:, 1].max())))
        for y in range(low, high):
            crossing = (a[:, 1] > y) != (b[:, 1] > y)
            p, q = a[crossing], b[crossing]
            xs = np.sort(p[:, 0] + (y-p[:, 1])*(q[:, 0]-p[:, 0])/(q[:, 1]-p[:, 1]))
            for left, right in zip(xs[::2], xs[1::2]):
                start, end = max(0, int(np.ceil(left))), min(shape[1], int(np.ceil(right)))
                if end > start:
                    result[y, start:end] = True
        return result

    def fill_polygon(coordinates):
        if not coordinates:
            return
        polygon = raster_ring(coordinates[0])
        for hole in coordinates[1:]:
            polygon &= ~raster_ring(hole)
        mask[:] |= polygon

    for feature in features or []:
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if kind == "Polygon":
            fill_polygon(coordinates)
        elif kind == "MultiPolygon":
            for polygon in coordinates:
                fill_polygon(polygon)
    return mask.astype(bool)


def transform_sam_feature(feature, sx=1.0, sy=1.0, offset=(0, 0)):
    """Map API floating geometry to the annotation grid without rasterization."""
    geometry = feature.get('geometry') or {}
    def ring(points):
        return [[float(x)/sx+offset[0], float(y)/sy+offset[1]] for x,y in points]
    coords = geometry.get('coordinates') or []
    if geometry.get('type') == 'Polygon':
        coords = [ring(r) for r in coords]
    elif geometry.get('type') == 'MultiPolygon':
        coords = [[ring(r) for r in polygon] for polygon in coords]
    return {**feature, 'geometry': {**geometry, 'coordinates': coords}}


def sam_feature_rings(feature):
    geometry = feature.get('geometry') or {}
    coords = geometry.get('coordinates') or []
    if geometry.get('type') == 'Polygon':
        return coords
    if geometry.get('type') == 'MultiPolygon':
        return [ring for polygon in coords for ring in polygon]
    return []


def sam_roi_bounds(point, shape, requested_size):
    """Return a clipped square-ish ROI centered on an image point."""
    h, w = shape[:2]
    size = max(64, int(requested_size))
    roi_w, roi_h = min(size, w), min(size, h)
    cx, cy = int(point[0]), int(point[1])
    x0 = int(np.clip(cx - roi_w // 2, 0, max(0, w - roi_w)))
    y0 = int(np.clip(cy - roi_h // 2, 0, max(0, h - roi_h)))
    return x0, y0, x0 + roi_w, y0 + roi_h


def sam_context_bounds(point, shape, visible_bounds, minimum=256):
    """Keep surrounding structure even when the displayed field is tiny."""
    x0, y0, x1, y1 = visible_bounds
    size = max(minimum, int(np.ceil(max(x1-x0, y1-y0) * 1.4)))
    return sam_roi_bounds(point, shape, size)


def display_contour_points(contour, display_scale):
    # At high zoom retain all contour vertices; at low zoom use a subpixel
    # SCREEN-space tolerance, never a fixed image-space tolerance.
    if display_scale >= 1:
        return contour.reshape(-1, 2)
    epsilon = 0.35 / max(display_scale, 1e-6)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)


def rasterize_sam_candidates(features, shape, prompts, mode="点选小目标"):
    """Rasterize and rank SAM alternatives; avoid selecting a full-ROI mask by score alone."""
    options = []
    h, w = shape[:2]
    for index, feature in enumerate(features or []):
        mask = geojson_to_mask([feature], (h, w))
        area = int(mask.sum())
        if area == 0:
            continue
        score = (feature.get("properties") or {}).get("quality")
        score_value = -1.0 if score is None else float(score)
        obeys = True
        for (x, y), label in prompts:
            xi, yi = int(round(x)), int(round(y))
            inside = 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi])
            if (label == 1 and not inside) or (label == 0 and inside):
                obeys = False
                break
        options.append({"mask": mask, "score": score, "score_value": score_value,
                        "area": area, "obeys": obeys, "source_index": index, "feature": feature})
    if not options:
        raise SamApiError("SAM候选轮廓为空")
    valid = [item for item in options if item["obeys"]] or options
    non_global = [item for item in valid if item["area"] <= h * w * 0.80]
    if non_global:
        valid = non_global
    if mode == "最高质量":
        selected = max(valid, key=lambda item: (item["score_value"], -item["area"]))
    elif mode == "最大候选":
        selected = max(valid, key=lambda item: item["area"])
    else:
        selected = min(valid, key=lambda item: (item["area"], -item["score_value"]))
    ordered = [selected] + [item for item in options if item is not selected]
    return ordered


def make_sam_view(
    gray, manual_mask, candidate, prompts, alpha=0.45, roi_bounds=None, hover_point=None
):
    out = make_mask_overlay(gray, manual_mask, alpha)
    if candidate is not None:
        layer = out.copy()
        layer[candidate] = (255, 220, 0)
        out = cv2.addWeighted(out, 0.55, layer, 0.45, 0)
    for (x, y), label in prompts:
        color = (0, 255, 80) if label == 1 else (255, 70, 70)
        cv2.circle(out, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    if hover_point is not None:
        hx, hy = (int(hover_point[0]), int(hover_point[1]))
        cv2.circle(out, (hx, hy), 7, (0, 255, 80), 2, lineType=cv2.LINE_AA)
        cv2.circle(out, (hx, hy), 10, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    if roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 255), 1)
    return out


class ImagePane:
    def __init__(
        self,
        parent,
        title,
        view_getter=None,
        on_motion=None,
        on_leave=None,
        on_press=None,
        on_drag=None,
        on_release=None,
        on_wheel=None,
        on_pan_start=None,
        on_pan_drag=None,
        on_pan_end=None,
        on_secondary_press=None,
    ):
        self.frame = ttk.Frame(parent, padding=4)
        self.title = ttk.Label(self.frame, text=title)
        self.title.pack(anchor="w")
        self.canvas = tk.Canvas(self.frame, width=360, height=360, bg="#202020", highlightthickness=1, highlightbackground="#777")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_id = None
        self.photo = None
        self.base_pil = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.cross_items = []
        self.crop_item = None
        self.sam_overlay = None
        self.view_getter = view_getter
        self.canvas.bind("<Motion>", on_motion or (lambda e: None))
        self.canvas.bind("<Leave>", on_leave or (lambda e: None))
        self.canvas.bind("<ButtonPress-1>", on_press or (lambda e: None))
        self.canvas.bind("<B1-Motion>", on_drag or (lambda e: None))
        self.canvas.bind("<ButtonRelease-1>", on_release or (lambda e: None))
        self.canvas.bind("<MouseWheel>", on_wheel or (lambda e: None))
        self.canvas.bind("<ButtonPress-2>", on_pan_start or (lambda e: None))
        self.canvas.bind("<B2-Motion>", on_pan_drag or (lambda e: None))
        self.canvas.bind("<ButtonPress-3>", on_secondary_press or on_pan_start or (lambda e: None))
        self.canvas.bind("<B3-Motion>", on_pan_drag or (lambda e: None))
        self.canvas.bind("<ButtonRelease-2>", on_pan_end or (lambda e: None))
        self.canvas.bind("<ButtonRelease-3>", on_pan_end or (lambda e: None))
        self.canvas.bind("<Configure>", lambda event: self.refresh())

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def set_image(self, image_rgb):
        self.base_pil = Image.fromarray(image_rgb)
        self.refresh()

    def refresh(self):
        if self.base_pil is None:
            return
        cw = max(80, self.canvas.winfo_width())
        ch = max(80, self.canvas.winfo_height())
        iw, ih = self.base_pil.size
        fit_scale = min(cw / iw, ch / ih)
        if self.view_getter:
            zoom, center_x, center_y = self.view_getter()
        else:
            zoom, center_x, center_y = 1.0, iw / 2.0, ih / 2.0
        self.scale = fit_scale * max(0.05, float(zoom))
        self.offset_x = int(cw / 2 - center_x * self.scale)
        self.offset_y = int(ch / 2 - center_y * self.scale)
        # Inverse mapping samples only the viewport; allocation never grows with zoom.
        shown = self.base_pil.transform(
            (cw, ch), Image.Transform.AFFINE,
            (1 / self.scale, 0, -self.offset_x / self.scale,
             0, 1 / self.scale, -self.offset_y / self.scale),
            resample=Image.Resampling.BILINEAR, fillcolor=(32, 32, 32),
        )
        self.photo = ImageTk.PhotoImage(shown)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        else:
            self.canvas.itemconfigure(self.image_id, image=self.photo)
        self.canvas.tag_lower(self.image_id)
        self.canvas.delete("sam-overlay")
        self.canvas.delete("crop-window")
        self.crop_item = None
        self.clear_crosshair()
        self._draw_sam_overlay()

    def visible_bounds(self):
        iw, ih = self.base_pil.size
        cw, ch = max(80, self.canvas.winfo_width()), max(80, self.canvas.winfo_height())
        return (max(0, int(-self.offset_x / self.scale)),
                max(0, int(-self.offset_y / self.scale)),
                min(iw, int(np.ceil((cw - self.offset_x) / self.scale))),
                min(ih, int(np.ceil((ch - self.offset_y) / self.scale))))

    def set_sam_overlay(self, candidate=None, prompts=(), roi_bounds=None, hover_point=None, feature=None):
        self.sam_overlay = (candidate, tuple(prompts), roi_bounds, hover_point, feature)
        self.canvas.delete("sam-overlay")
        self._draw_sam_overlay()

    def _image_to_canvas(self, x, y):
        return self.offset_x + float(x) * self.scale, self.offset_y + float(y) * self.scale

    def _draw_sam_overlay(self):
        if self.base_pil is None or self.sam_overlay is None:
            return
        candidate, prompts, roi_bounds, hover_point, feature = self.sam_overlay
        if candidate is not None and feature is not None:
            for ring in sam_feature_rings(feature):
                if len(ring) < 3:
                    continue
                coords = [v for x,y in ring for v in self._image_to_canvas(x,y)]
                if ring[0] != ring[-1]:
                    coords.extend(coords[:2])
                self.canvas.create_line(*coords, fill='#ffd800', width=2, tags='sam-overlay')
        elif candidate is not None and np.any(candidate):
            x0 = y0 = 0
            region = candidate
            if roi_bounds is not None:
                x0, y0, x1, y1 = roi_bounds
                region = candidate[y0:y1, x0:x1]
            contours, _ = cv2.findContours(
                region.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) < 3 or cv2.contourArea(contour) < 2:
                    continue
                points = display_contour_points(contour, self.scale)
                coords = []
                for px, py in points:
                    cx, cy = self._image_to_canvas(px + x0, py + y0)
                    coords.extend((cx, cy))
                if len(coords) >= 6:
                    coords.extend(coords[:2])
                    self.canvas.create_line(
                        *coords, fill="#ffd800", width=2, tags="sam-overlay"
                    )
        if roi_bounds is not None:
            x0, y0, x1, y1 = roi_bounds
            cx0, cy0 = self._image_to_canvas(x0, y0)
            cx1, cy1 = self._image_to_canvas(x1, y1)
            self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1, outline="#00d8d8", width=1, tags="sam-overlay"
            )
        for (x, y), label in prompts:
            cx, cy = self._image_to_canvas(x, y)
            color = "#00f060" if label == 1 else "#ff5050"
            self.canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5,
                fill=color, outline="white", width=1, tags="sam-overlay",
            )
        if hover_point is not None:
            cx, cy = self._image_to_canvas(*hover_point)
            self.canvas.create_oval(
                cx - 7, cy - 7, cx + 7, cy + 7,
                outline="#00f060", width=2, tags="sam-overlay",
            )

    def canvas_to_image(self, x, y):
        if self.base_pil is None or self.scale <= 0:
            return None
        ix = int(np.floor((x - self.offset_x) / self.scale))
        iy = int(np.floor((y - self.offset_y) / self.scale))
        iw, ih = self.base_pil.size
        if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
            return None
        return ix, iy

    def draw_crosshair(self, ix, iy):
        if self.base_pil is None:
            return
        x = self.offset_x + ix * self.scale
        y = self.offset_y + iy * self.scale
        iw, ih = self.base_pil.size
        x0, x1 = self.offset_x, self.offset_x + iw * self.scale
        y0, y1 = self.offset_y, self.offset_y + ih * self.scale
        if len(self.cross_items) != 3:
            self.cross_items = [
                self.canvas.create_line(x0, y, x1, y, fill="#00ffff", width=2, tags="crosshair"),
                self.canvas.create_line(x, y0, x, y1, fill="#00ffff", width=2, tags="crosshair"),
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#ffff00", width=2, tags="crosshair"),
            ]
        else:
            self.canvas.coords(self.cross_items[0], x0, y, x1, y)
            self.canvas.coords(self.cross_items[1], x, y0, x, y1)
            self.canvas.coords(self.cross_items[2], x - 4, y - 4, x + 4, y + 4)
            for item in self.cross_items:
                self.canvas.itemconfigure(item, state="normal")

    def clear_crosshair(self):
        for item in self.cross_items:
            self.canvas.itemconfigure(item, state="hidden")

    def draw_crop_window(self, ix, iy, size):
        if self.base_pil is None:
            return
        h, w = np.asarray(self.base_pil).shape[:2]
        x0, y0, x1, y1 = sam_roi_bounds((ix, iy), (h, w), size)
        cx0, cy0 = self._image_to_canvas(x0, y0)
        cx1, cy1 = self._image_to_canvas(x1, y1)
        if self.crop_item is None:
            self.crop_item = self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1, outline="#ff9f00", width=2, tags="crop-window"
            )
        else:
            self.canvas.coords(self.crop_item, cx0, cy0, cx1, cy1)
            self.canvas.itemconfigure(self.crop_item, state="normal")

    def clear_crop_window(self):
        if self.crop_item is not None:
            self.canvas.itemconfigure(self.crop_item, state="hidden")


class MaskEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{Path(__file__).resolve().parent.name} 标注器")
        self.root.geometry("1600x920")
        self.root.minsize(1100, 680)
        self.before_input_path = tk.StringVar(value="")
        self.after_input_path = tk.StringVar(value="")
        self.before_magnification = tk.StringVar(value="自动")
        self.after_magnification = tk.StringVar(value="自动")
        self.registration_status = tk.StringVar(value="步骤1：请选择 Before 和 After")
        self.mask_source_status = tk.StringVar(value="标签：尚未加载")
        self.registration_mode = tk.StringVar(value="同倍率变化")
        self.annotation_mode = tk.StringVar(value=MODES[0])
        self.sam_fine_mode = tk.BooleanVar(value=False)
        self.annotation_side = tk.StringVar(value='before')
        self.annotation_class = tk.StringVar(value='1 新增孔隙')
        self.annotations = None
        self.active_annotation = 'before'
        self.registration_busy = False
        self.registration_ready = False
        self.last_registration_report = None
        self.index_path = tk.StringVar(value=str(DEFAULT_INDEX))
        self.status_filter = tk.StringVar(value="ok")
        # When a user opens a task folder or ready_to_annotate folder, keep
        # the selected folder so the sibling rows can still be navigated.
        self.folder_navigation_target = None
        self.brush_size = tk.IntVar(value=1)
        self.brush_kind = tk.StringVar(value="normal")
        self.reference_source = tk.StringVar(value="diff")
        self.color_tolerance = tk.IntVar(value=24)
        self.algorithm_radius = tk.IntVar(value=180)
        self.alpha = tk.DoubleVar(value=0.45)
        self.mode = tk.StringVar(value="add")
        self.mask_view_mode = tk.StringVar(value="pure")
        self.draw_target = tk.StringVar(value="mask+overlay")
        self.overlay_base = tk.StringVar(value="before")
        self.fast_mode = tk.BooleanVar(value=True)
        self.quality_status = tk.StringVar(value="合格")
        self.note_text = tk.StringVar(value="")
        self.sam_enabled = tk.BooleanVar(value=False)
        self.sam_model = tk.StringVar(value="sam2_s")
        self.sam_prompt_label = tk.IntVar(value=1)
        self.sam_apply_mode = tk.StringVar(value="add")
        self.sam_roi_size = tk.IntVar(value=128)
        self.sam_candidate_mode = tk.StringVar(value="最高质量")
        self.sam_status = tk.StringVar(value="SAM API：待检测")
        self.classical_source = tk.StringVar(value="diff")
        self.classical_polarity = tk.StringVar(value="dark")
        self.min_component_area = tk.IntVar(value=20)
        self.sam_client = SamClient()
        self.sam_busy = False
        self.sam_active_side = None
        self.sam_session_id = None
        self.sam_request_id = 0
        self.sam_active_roi = None
        self.sam_active_model = None
        self.sam_generation = 0
        self.sam_pending_side = None
        self.sam_prompts = {"before": [], "after": []}
        self.sam_candidates = {"before": None, "after": None}
        self.sam_scores = {"before": None, "after": None}
        self.sam_rois = {"before": None, "after": None}
        self.sam_candidate_options = {"before": [], "after": []}
        self.sam_candidate_indices = {"before": 0, "after": 0}
        self.sam_restart_required = {"before": False, "after": False}
        self.sam_hover_points = {"before": None, "after": None}
        self.sam_hover_job = None
        self.sam_hover_side = None
        self.advanced_visible = False
        self.rows = []
        self.idx = 0
        self.before = None
        self.after = None
        self.diff = None
        self.auto_mask = None
        self.mask = None
        self.original_mask = None
        self.current_row = None
        self.drawing = False
        self.last_brush_point = None
        self.unsaved = False
        self.zoom = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.pan_last = None
        self.cursor_xy = ""
        self.cursor_point = None
        self.last_sample = ""
        self.info_base_text = ""
        self.navigation_status = tk.StringVar(value="未加载任务")
        self.refresh_pending = False
        self.undo_stack = []
        self.redo_stack = []
        self._build()
        self.root.bind("<Prior>", lambda _event: self.prev_item())
        self.root.bind("<Next>", lambda _event: self.next_item())
        self.root.bind("<Return>", self.on_enter_action)

    def on_enter_action(self, _event=None):
        if (self.before is not None and self.cursor_point is not None
                and abs(self.zoom - 1.0) < 1e-6 and self.sam_candidates.get('before') is None
                and self.sam_candidates.get('after') is None):
            self.export_current_crop()
        else:
            self.accept_sam_candidate()
        return "break"

    def _build(self):
        workflow = ttk.LabelFrame(self.root, text="两步完成配准与标注", padding=10)
        workflow.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Label(workflow, text="① 原始TIF", font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(workflow, text="选择 Before.tif", command=self.pick_before_image).grid(row=0, column=1, padx=(10, 3))
        ttk.Entry(workflow, textvariable=self.before_input_path, state="readonly").grid(row=0, column=2, sticky="ew", padx=3)
        ttk.Button(workflow, text="选择 After.tif", command=self.pick_after_image).grid(row=0, column=3, padx=(10, 3))
        ttk.Entry(workflow, textvariable=self.after_input_path, state="readonly").grid(row=0, column=4, sticky="ew", padx=3)
        ttk.Button(workflow, text="读取倍率并配准", command=self.run_pair_registration).grid(row=0, column=5, padx=(12, 3))
        ttk.Label(workflow, textvariable=self.registration_status).grid(
            row=1, column=1, columnspan=5, sticky="w", padx=(10, 0), pady=(5, 7)
        )
        ttk.Separator(workflow).grid(row=2, column=0, columnspan=6, sticky="ew", pady=(0, 7))
        ttk.Label(workflow, text="② SAM2画Mask", font=("Microsoft YaHei UI", 10, "bold")).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(
            workflow, text="开始交互标注", variable=self.sam_enabled, command=self.toggle_sam_mode
        ).grid(row=3, column=1, padx=(10, 3), sticky="w")
        ttk.Button(workflow, text="接受轮廓  Enter", command=self.accept_sam_candidate).grid(row=3, column=2, padx=3, sticky="w")
        ttk.Button(workflow, text="保存Mask", command=self.save_mask).grid(row=3, column=3, padx=3, sticky="w")
        self.advanced_button = ttk.Button(workflow, text="高级工具…", command=self.toggle_advanced)
        self.advanced_button.grid(row=3, column=4, padx=(12, 3), sticky="e")
        ttk.Checkbutton(workflow, text='小孔隙精细模式（试验）', variable=self.sam_fine_mode,
                        command=self.change_sam_detail).grid(row=3, column=5, padx=6, sticky='w')
        ttk.Label(workflow, textvariable=self.sam_status).grid(row=4, column=1, columnspan=5, sticky="w", padx=(10, 0), pady=(5, 0))
        ttk.Label(workflow, textvariable=self.mask_source_status, foreground='#1769aa').grid(
            row=5, column=1, columnspan=5, sticky='w', padx=(10, 0), pady=(5, 0))
        workflow.columnconfigure(2, weight=1)
        workflow.columnconfigure(4, weight=1)
        annotate = ttk.Frame(self.root, padding=(10, 4))
        annotate.pack(fill=tk.X)
        ttk.Label(annotate, text='标注模式').pack(side=tk.LEFT)
        selector = ttk.Combobox(annotate, textvariable=self.annotation_mode,
                               values=MODES, state='readonly', width=14)
        selector.pack(side=tk.LEFT, padx=5)
        selector.bind('<<ComboboxSelected>>', self.change_annotation_mode)
        ttk.Button(annotate, text='上一张', command=self.prev_item).pack(side=tk.LEFT, padx=2)
        ttk.Button(annotate, text='下一张', command=self.next_item).pack(side=tk.LEFT, padx=2)
        ttk.Label(annotate, textvariable=self.navigation_status, foreground="#1769aa").pack(side=tk.LEFT, padx=(3, 6))
        ttk.Label(annotate, text='孔隙画笔侧').pack(side=tk.LEFT)
        self.side_selector = ttk.Combobox(annotate, textvariable=self.annotation_side,
                                         values=('before', 'after'), state='readonly', width=8)
        self.side_selector.pack(side=tk.LEFT, padx=5)
        self.side_selector.bind('<<ComboboxSelected>>', self.change_annotation_mode)
        ttk.Label(annotate, text='变化类别').pack(side=tk.LEFT)
        self.class_selector = ttk.Combobox(annotate, textvariable=self.annotation_class,
            values=[f'{k} {v}' for k, v in CLASSES.items() if k], state='disabled', width=20)
        self.class_selector.pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(annotate, text='添加', variable=self.mode, value='add').pack(side=tk.LEFT)
        ttk.Radiobutton(annotate, text='擦除', variable=self.mode, value='erase').pack(side=tk.LEFT)
        ttk.Label(annotate, text='笔径(px)').pack(side=tk.LEFT)
        ttk.Spinbox(annotate, from_=1, to=40, textvariable=self.brush_size, width=4).pack(side=tk.LEFT)
        ttk.Label(annotate, text='局部窗口(px)').pack(side=tk.LEFT, padx=(10, 2))
        crop_size_box = ttk.Combobox(annotate, textvariable=self.sam_roi_size,
                     values=[128, 256, 384, 512, 768, 1024],
                     state='readonly', width=6)
        crop_size_box.pack(side=tk.LEFT, padx=2)
        crop_size_box.bind('<<ComboboxSelected>>', lambda _event: self.redraw_crop_window())
        ttk.Label(annotate, text='未放大时按原始像素裁切').pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(annotate, text='导出当前裁切', command=self.export_current_crop).pack(side=tk.LEFT, padx=3)
        ttk.Button(annotate, text='撤销', command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(annotate, text='重做', command=self.redo).pack(side=tk.LEFT)
        ttk.Button(annotate, text='打开已有任务', command=self.open_annotation_project).pack(side=tk.LEFT, padx=5)
        ttk.Button(annotate, text='红蓝差异→多分类', command=self.initialize_multiclass_from_pore).pack(side=tk.LEFT, padx=5)
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="步骤1 输入与配准").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="选择Before", command=self.pick_before_image).grid(row=0, column=1, padx=(6, 2))
        ttk.Entry(top, textvariable=self.before_input_path, width=30).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Label(top, text="倍率").grid(row=0, column=3, padx=(6, 2))
        ttk.Combobox(
            top, textvariable=self.before_magnification, values=["自动", "500", "2000"], width=7,
        ).grid(row=0, column=4, padx=2)
        ttk.Button(top, text="选择After", command=self.pick_after_image).grid(row=0, column=5, padx=(10, 2))
        ttk.Entry(top, textvariable=self.after_input_path, width=30).grid(row=0, column=6, sticky="ew", padx=2)
        ttk.Label(top, text="倍率").grid(row=0, column=7, padx=(6, 2))
        ttk.Combobox(
            top, textvariable=self.after_magnification, values=["自动", "500", "2000"], width=7,
        ).grid(row=0, column=8, padx=2)
        ttk.Combobox(
            top, textvariable=self.registration_mode, values=["同倍率变化", "跨倍率定位"],
            state="readonly", width=12,
        ).grid(row=0, column=9, padx=(8, 2))
        ttk.Button(top, text="裁剪+配准并载入", command=self.run_pair_registration).grid(row=0, column=10, padx=2)
        ttk.Button(top, text="查看报告", command=self.open_registration_report).grid(row=0, column=11, padx=2)
        ttk.Button(top, text='增强匹配', command=lambda: self.run_pair_registration('lightglue')).grid(row=3,column=1,padx=2,pady=4)
        ttk.Button(top, text='人工辅助配准', command=lambda: self.run_pair_registration('manual')).grid(row=3,column=2,padx=2,pady=4)
        ttk.Label(top,text='困难样本可选；重新配准会建立新任务，不迁移旧Mask').grid(row=3,column=3,columnspan=7,sticky='w')
        ttk.Button(top,text='添加文件夹，配准',command=self.open_batch_registration).grid(row=4,column=1,pady=4)
        ttk.Button(top,text='检测配准质量',command=lambda:self.open_batch_registration(audit=True)).grid(row=4,column=2,pady=4)

        ttk.Label(top, text="或加载数据集CSV").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.index_path).grid(row=1, column=1, columnspan=6, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="选择CSV", command=self.pick_index).grid(row=1, column=7, padx=3, pady=(6, 0))
        ttk.Button(top, text="加载", command=self.load_index).grid(row=1, column=8, padx=3, pady=(6, 0))
        ttk.Label(top, text="状态筛选").grid(row=1, column=9, sticky="e", padx=(12, 3), pady=(6, 0))
        ttk.Entry(top, textvariable=self.status_filter, width=8).grid(row=1, column=10, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(top, textvariable=self.registration_status).grid(
            row=2, column=0, columnspan=12, sticky="w", pady=(5, 0)
        )
        top.columnconfigure(2, weight=1)
        top.columnconfigure(6, weight=1)

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="保存 manual_mask", command=self.save_mask).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(controls, text="撤销", command=self.undo).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="重做", command=self.redo).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="撤回到自动mask", command=self.reset_mask).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="打开manual文件夹", command=self.open_manual_dir).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(controls, text="适配窗口", command=self.fit_view).pack(side=tk.LEFT, padx=4)
        ttk.Separator(controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Radiobutton(controls, text="添加白色变化", value="add", variable=self.mode).pack(side=tk.LEFT)
        ttk.Radiobutton(controls, text="擦除变化", value="erase", variable=self.mode).pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="画笔").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Scale(controls, from_=1, to=40, variable=self.brush_size, orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.brush_size, width=4).pack(side=tk.LEFT)
        ttk.Label(controls, text="笔触").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(
            controls,
            textvariable=self.brush_kind,
            values=["normal", "color-aware"],
            state="readonly",
            width=11,
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="取色参考").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(
            controls,
            textvariable=self.reference_source,
            values=["before", "after", "diff"],
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="容差").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Scale(controls, from_=3, to=90, variable=self.color_tolerance, orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.color_tolerance, width=4).pack(side=tk.LEFT)
        ttk.Label(controls, text="算法半径").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Scale(controls, from_=40, to=420, variable=self.algorithm_radius, orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.algorithm_radius, width=4).pack(side=tk.LEFT)
        ttk.Label(controls, text="mask显示").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Combobox(
            controls,
            textvariable=self.mask_view_mode,
            values=["pure", "overlay"],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT)
        self.mask_view_mode.trace_add("write", lambda *_: self.refresh_images())
        ttk.Label(controls, text="绘制位置").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(
            controls,
            textvariable=self.draw_target,
            values=["mask", "overlay", "mask+overlay"],
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="mask透明度").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Scale(controls, from_=0.15, to=0.85, variable=self.alpha, orient=tk.HORIZONTAL, length=120, command=lambda _: self.refresh_images()).pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="快速模式", variable=self.fast_mode).pack(side=tk.LEFT, padx=(10, 0))

        self.info = tk.StringVar(value="请选择原始Before/After TIF，程序会自动读取相邻HDR。")
        self.info_label = ttk.Label(self.root, textvariable=self.info, padding=(10, 2))
        self.info_label.pack(fill=tk.X)

        semi = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        semi.pack(fill=tk.X)
        ttk.Label(semi, text="半自动贴边").pack(side=tk.LEFT)
        ttk.Button(semi, text="区域生长添加", command=lambda: self.region_grow("add")).pack(side=tk.LEFT, padx=(8, 3))
        ttk.Button(semi, text="区域生长擦除", command=lambda: self.region_grow("erase")).pack(side=tk.LEFT, padx=3)
        ttk.Button(semi, text="边缘贴合当前区域", command=self.edge_snap_roi).pack(side=tk.LEFT, padx=(12, 3))
        ttk.Label(semi, text="Overlay底图").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Combobox(
            semi,
            textvariable=self.overlay_base,
            values=["before", "after", "diff"],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT)
        self.overlay_base.trace_add("write", lambda *_: self.refresh_images())
        ttk.Label(semi, text="质量").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Combobox(
            semi,
            textvariable=self.quality_status,
            values=["合格", "配准差", "边界不确定", "跳过"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)
        ttk.Label(semi, text="备注").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(semi, textvariable=self.note_text, width=28).pack(side=tk.LEFT)

        classical = ttk.LabelFrame(self.root, text="OpenCV 经典图像处理（当前十字线ROI）", padding=(8, 4))
        classical.pack(fill=tk.X, padx=10, pady=(0, 6))
        ttk.Label(classical, text="图像").pack(side=tk.LEFT)
        ttk.Combobox(
            classical, textvariable=self.classical_source,
            values=["before", "after", "diff"], state="readonly", width=7,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(classical, text="目标").pack(side=tk.LEFT)
        ttk.Combobox(
            classical, textvariable=self.classical_polarity,
            values=["dark", "bright"], state="readonly", width=7,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(classical, text="最小连通域").pack(side=tk.LEFT)
        ttk.Spinbox(classical, from_=1, to=100000, textvariable=self.min_component_area, width=7).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Button(classical, text="分析/预览ROI", command=self.preview_classical_processing).pack(side=tk.LEFT, padx=3)
        ttk.Button(classical, text="Otsu结果添加", command=lambda: self.apply_classical_mask("add")).pack(side=tk.LEFT, padx=3)
        ttk.Button(classical, text="Otsu结果擦除", command=lambda: self.apply_classical_mask("erase")).pack(side=tk.LEFT, padx=3)
        ttk.Button(classical, text="当前Mask连通域统计", command=self.report_mask_components).pack(side=tk.LEFT, padx=(10, 3))

        sam = ttk.LabelFrame(self.root, text="SAM2 / SAM3 交互分割", padding=(8, 4))
        sam.pack(fill=tk.X, padx=10, pady=(0, 6))
        sam_top = ttk.Frame(sam)
        sam_top.pack(fill=tk.X)
        sam_bottom = ttk.Frame(sam)
        sam_bottom.pack(fill=tk.X, pady=(3, 0))
        ttk.Checkbutton(sam_top, text="启用：Before/After左键提示，右键排除", variable=self.sam_enabled).pack(side=tk.LEFT)
        ttk.Label(sam_top, text="模型").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(
            sam_top, textvariable=self.sam_model,
            values=["sam2_s", "sam2_t", "sam2_bp", "sam2_l", "sam3"],
            state="readonly", width=9,
        ).pack(side=tk.LEFT)
        ttk.Label(sam_top, text="候选策略").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(
            sam_top, textvariable=self.sam_candidate_mode,
            values=["点选小目标", "最高质量", "最大候选"], state="readonly", width=11,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(sam_top, text="左键前景", value=1, variable=self.sam_prompt_label).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Radiobutton(sam_top, text="左键背景", value=0, variable=self.sam_prompt_label).pack(side=tk.LEFT, padx=2)
        ttk.Label(sam_top, text="接受方式").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(
            sam_top, textvariable=self.sam_apply_mode,
            values=["add", "erase", "replace-local"], state="readonly", width=13,
        ).pack(side=tk.LEFT)
        ttk.Button(sam_bottom, text="接受候选（Enter）", command=self.accept_sam_candidate).pack(side=tk.LEFT)
        ttk.Button(sam_bottom, text="切换候选", command=self.cycle_sam_candidate).pack(side=tk.LEFT, padx=2)
        ttk.Button(sam_bottom, text="撤销提示点", command=self.undo_sam_prompt).pack(side=tk.LEFT, padx=2)
        ttk.Button(sam_bottom, text="清除当前侧（Esc）", command=self.clear_sam_prompts).pack(side=tk.LEFT, padx=2)
        ttk.Button(sam_bottom, text="检测服务", command=self.check_sam_service).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Label(sam_bottom, textvariable=self.sam_status).pack(side=tk.LEFT, padx=8)

        panes = ttk.Frame(self.root, padding=8)
        self.panes_frame = panes
        panes.pack(fill=tk.BOTH, expand=True)
        self.before_pane = ImagePane(
            panes,
            "Before 图",
            self.get_view,
            self.on_motion,
            self.on_leave,
            self.on_press,
            on_wheel=self.on_wheel,
            on_pan_start=self.on_pan_start,
            on_pan_drag=self.on_pan_drag,
            on_pan_end=self.on_pan_end,
            on_secondary_press=self.on_right_press,
        )
        self.after_pane = ImagePane(
            panes,
            "After 配准图",
            self.get_view,
            self.on_motion,
            self.on_leave,
            self.on_press,
            on_wheel=self.on_wheel,
            on_pan_start=self.on_pan_start,
            on_pan_drag=self.on_pan_drag,
            on_pan_end=self.on_pan_end,
            on_secondary_press=self.on_right_press,
        )
        self.mask_pane = ImagePane(
            panes,
            "Mask 编辑区：黑=非变化，白=变化",
            self.get_view,
            self.on_motion,
            self.on_leave,
            self.on_press,
            self.on_drag,
            self.on_release,
            self.on_wheel,
            self.on_pan_start,
            self.on_pan_drag,
            self.on_pan_end,
        )
        self.overlay_pane = ImagePane(
            panes,
            "Overlay 预览/编辑：Before + 当前Mask",
            self.get_view,
            self.on_motion,
            self.on_leave,
            self.on_press,
            self.on_drag,
            self.on_release,
            on_wheel=self.on_wheel,
            on_pan_start=self.on_pan_start,
            on_pan_drag=self.on_pan_drag,
            on_pan_end=self.on_pan_end,
        )
        self.before_pane.grid(row=0, column=0, sticky="nsew")
        self.after_pane.grid(row=0, column=1, sticky="nsew")
        self.mask_pane.grid(row=0, column=2, sticky="nsew")
        self.overlay_pane.grid(row=0, column=3, sticky="nsew")
        for c in range(4):
            panes.columnconfigure(c, weight=1)
        panes.rowconfigure(0, weight=1)

        help_text = (
            "操作：鼠标在任意图上移动，三栏十字基线同步；在右侧 mask 区左键拖动修改。"
            "滚轮同步缩放；中键拖动同步平移；SAM关闭时右键也可平移。"
            "color-aware 笔触会按 before/after/diff 的落笔范围灰度自动筛选相近像素。"
            "半自动贴边按钮会围绕当前十字线局部处理，不会直接改全图。"
            "Overlay底图可切 before/after/diff；快速模式拖动时先不实时刷新 overlay。"
            "Mask 编辑区可切 pure/overlay；最右侧 Overlay 预览会一直保留融合显示。"
            "绘制位置可选择 mask、overlay 或 mask+overlay，保存的始终是同一张黑白 manual_mask。"
            "SAM开启时Before/After左键按所选标签提示，右键始终是背景排除点；Enter接受，Esc清除。"
            "白色=确认变化，黑色=非变化。建议对照 Before/After，只修正真实溶蚀孔变化斑块。"
        )
        self.help_label = ttk.Label(self.root, text=help_text, padding=(10, 0, 10, 10), wraplength=1200)
        self.help_label.pack(fill=tk.X)
        self.advanced_frames = [top, controls, semi, classical, sam]
        for frame in self.advanced_frames:
            frame.pack_forget()
        self.help_label.pack_forget()
        self.root.after(250, self.check_sam_service)
        self.root.bind("<Return>", self.on_enter_action)
        self.root.bind("<Escape>", self.on_sam_cancel_key)
        self.root.bind("<Control-z>", self.on_ctrl_z)
        self.root.bind("<Control-s>", lambda _event: self.save_mask())

    def toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            for frame in self.advanced_frames:
                frame.pack(fill=tk.X, before=self.info_label)
            self.help_label.pack(fill=tk.X, before=self.panes_frame)
            self.advanced_button.configure(text="收起高级工具")
        else:
            for frame in self.advanced_frames:
                frame.pack_forget()
            self.help_label.pack_forget()
            self.advanced_button.configure(text="高级工具…")

    def annotation_folder(self, row):
        if row.get('annotation_dir'):
            return Path(row['annotation_dir'])
        if row.get('registration_metrics'):
            return Path(row['registration_metrics']).parent
        return Path(__file__).resolve().parent / 'annotation_projects' / str(row.get('pair_id', 'sample')).replace('/', '_').replace('\\', '_')

    def commit_annotation(self):
        if self.annotations is not None:
            self.annotations.layers[self.active_annotation].update(
                mask=self.mask, undo=self.undo_stack, redo=self.redo_stack, original=self.original_mask)

    def select_annotation(self, key):
        if self.annotations is None:
            return
        self.commit_annotation()
        self.active_annotation = key
        layer = self.annotations.layers[key]
        self.mask, self.undo_stack, self.redo_stack, self.original_mask = (
            layer['mask'], layer['undo'], layer['redo'], layer['original'])
        self.annotations.used.add(key)
        self.last_brush_point = None

    def change_annotation_mode(self, _event=None):
        self.stop_sam_mode('标注模式/编辑侧已切换；开启SAM后继续标注')
        self.drawing = False
        mode = self.annotation_mode.get()
        self.side_selector.configure(state='readonly' if mode == MODES[0] else 'disabled')
        self.class_selector.configure(state='readonly' if mode == MODES[2] else 'disabled')
        key = self.annotation_side.get() if mode == MODES[0] else ('binary' if mode == MODES[1] else 'multi')
        self.select_annotation(key)
        self.refresh_images()

    def annotation_value(self):
        return int(self.annotation_class.get().split()[0]) if self.active_annotation == 'multi' else 1

    def initialize_multiclass_from_pore(self):
        """Create an editable provisional change-class layer from Aedge pore layers."""
        if self.annotations is None:
            messagebox.showwarning('尚未载入任务', '请先打开 ready_to_annotate 中的任务。')
            return
        before = self.annotations.layers['before']['mask'].astype(bool)
        after = self.annotations.layers['after']['mask'].astype(bool)
        if not before.any() and not after.any():
            messagebox.showwarning('没有红蓝伪标签', '当前任务没有可转换的 Before/After 孔隙Mask。')
            return
        layer = self.annotations.layers['multi']
        if layer['mask'].any() and not messagebox.askyesno('覆盖多分类初始层', '当前多分类已有内容，是否重新生成暂定分类？'):
            return
        multi = np.zeros(before.shape, np.uint8)
        multi[after & ~before] = 1  # 暂定：新增孔隙，也可能是孔隙扩大
        multi[before & ~after] = 3  # 暂定：缩小/闭合
        self.commit_annotation()
        layer['mask'] = multi; layer['undo'] = []; layer['redo'] = []; layer['original'] = multi.copy()
        self.annotations.used.add('multi')
        self.annotation_mode.set(MODES[2]); self.side_selector.configure(state='disabled'); self.class_selector.configure(state='readonly')
        self.active_annotation = 'multi'; self.mask = layer['mask']; self.undo_stack = layer['undo']; self.redo_stack = layer['redo']; self.original_mask = layer['original']
        self.unsaved = True
        self.last_sample = '红蓝差异已载入多分类：红=暂定新增/扩大，蓝=暂定缩小/闭合；请人工确认'
        self.info.set(self.last_sample); self.refresh_images()

    def change_sam_detail(self):
        detail = '精细模式：局部范围最小64像素' if self.sam_fine_mode.get() else '常规模式：局部范围最小256像素'
        self.stop_sam_mode(detail + '；请重新开启SAM标注。已接受的Mask不变。')

    def open_annotation_project(self):
        folder = filedialog.askdirectory(title='选择已有样本任务文件夹')
        if not folder or not self.confirm_discard():
            return
        root = Path(folder).resolve()
        import json
        if (root/'annotation_index.csv').is_file():
            self.folder_navigation_target = None
            self.index_path.set(str(root/'annotation_index.csv'))
            self.status_filter.set('ok')
            self.load_index()
            return
        # A user often selects either ready_to_annotate itself or one of its
        # pair folders. Walk upward to the dataset manifest so the sibling
        # pairs become the navigation list instead of loading a one-row task.
        for ancestor in (root, *root.parents):
            if ancestor == root:
                continue
            try:
                relative = root.relative_to(ancestor)
            except ValueError:
                continue
            parts = relative.parts
            if not parts or parts[0] not in ('ready_to_annotate', 'needs_review'):
                continue
            csv_name = 'annotation_index.csv' if parts[0] == 'ready_to_annotate' else 'review_index.csv'
            index = ancestor / csv_name
            if index.is_file():
                # Selecting a container opens all tasks; selecting one pair
                # keeps that pair selected after the index is loaded.
                self.folder_navigation_target = None if len(parts) == 1 else root
                self.index_path.set(str(index))
                self.status_filter.set('ok' if csv_name == 'annotation_index.csv' else '')
                self.load_index()
                return
        manifest = root / 'annotation_project.json'
        saved = {}
        if manifest.exists():
            try:
                saved = json.loads(manifest.read_text(encoding='utf-8'))
                row = resolve_row(saved['row'], root)
            except (ValueError, KeyError, OSError) as exc:
                messagebox.showerror('无法打开', str(exc))
                return
            row['annotation_dir'] = str(root)
        else:
            row = {'pair_id': root.name, 'before': str(root/'before_core.png'),
                   'after': str(root/'after_registered_core.png'), 'mask': str(root/'change_mask_initial.png'),
                   'diff': str(root/'difference_core.png'),
                   'registration_metrics': str(root/'registration_metrics.json')}
        if not all(Path(row[k]).is_file() for k in ('before','after','mask')):
            messagebox.showerror('无法打开', '未找到任务对应的Before/After/初始Mask。')
            return
        self.unsaved = False
        if saved.get('mode') in MODES:
            self.annotation_mode.set(saved['mode'])
        if saved.get('side') in ('before', 'after'):
            self.annotation_side.set(saved['side'])
        if saved.get('class') in [f'{k} {v}' for k,v in CLASSES.items() if k != 0]:
            self.annotation_class.set(saved['class'])
        self.side_selector.configure(state='readonly' if self.annotation_mode.get() == MODES[0] else 'disabled')
        self.class_selector.configure(state='readonly' if self.annotation_mode.get() == MODES[2] else 'disabled')
        self.rows, self.idx = [row], 0
        self.load_current()
        self.note_text.set(saved.get('note', ''))
        self.quality_status.set(saved.get('quality', '待复核' if row.get('registration_status') == 'needs_review' else '合格'))

    def toggle_sam_mode(self):
        if self.sam_enabled.get():
            if self.before is None or self.after is None or not self.registration_ready:
                self.sam_enabled.set(False)
                messagebox.showinfo(
                    "请先通过配准",
                    "请先完成第一步，并让配准质量达到“优秀”或“良好”；未通过质量门槛的图像不能进入Mask标注。",
                )
                return
            self.sam_status.set("SAM2已开启：移动鼠标预览，左键包含，右键排除，Enter接受，Esc退出")
        else:
            self.stop_sam_mode("SAM2交互标注已停止")

    def stop_sam_mode(self, message="SAM2交互标注已停止"):
        self.sam_enabled.set(False)
        if self.sam_hover_job is not None:
            self.root.after_cancel(self.sam_hover_job)
            self.sam_hover_job = None
        old_session = self.sam_session_id
        self.sam_generation += 1
        self.sam_pending_side = None
        self.sam_active_side = None
        self.sam_session_id = None
        self.sam_request_id = 0
        self.sam_active_roi = None
        self.sam_active_model = None
        self.sam_prompts = {"before": [], "after": []}
        self.sam_hover_points = {"before": None, "after": None}
        self.sam_hover_side = None
        self.sam_candidates = {"before": None, "after": None}
        self.sam_scores = {"before": None, "after": None}
        self.sam_rois = {"before": None, "after": None}
        self.sam_candidate_options = {"before": [], "after": []}
        self.sam_candidate_indices = {"before": 0, "after": 0}
        self.sam_restart_required = {"before": False, "after": False}
        self.sam_status.set(message)
        self.refresh_sam_overlay()
        if old_session:
            threading.Thread(target=lambda: self.sam_client.close_session(old_session), daemon=True).start()

    def on_ctrl_z(self, _event):
        if self.sam_enabled.get() and (self.sam_prompts["before"] or self.sam_prompts["after"]):
            self.undo_sam_prompt()
        else:
            self.undo()

    def pick_before_image(self):
        path = filedialog.askopenfilename(
            title="选择反应前 Before 图像",
            filetypes=[("SEM TIFF", "*.tif *.tiff"), ("兼容图像", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if path:
            self.registration_ready = False
            self.before_input_path.set(path)
            metadata = read_sem_metadata(path)
            inferred = metadata["magnification"] or parse_magnification("自动", path)
            if inferred:
                self.before_magnification.set(f"{inferred:g}")
            pixel_text = metadata["pixel_size_x_um"]
            source = metadata.get("metadata_source") or "文件名"
            self.registration_status.set(
                f"Before：{inferred:g}×，{pixel_text:.6g} µm/px（{source}）"
                if inferred and pixel_text else f"Before：倍率={inferred or '未知'}（{source}）"
            )

    def pick_after_image(self):
        path = filedialog.askopenfilename(
            title="选择反应后 After 图像",
            filetypes=[("SEM TIFF", "*.tif *.tiff"), ("兼容图像", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if path:
            self.registration_ready = False
            self.after_input_path.set(path)
            metadata = read_sem_metadata(path)
            inferred = metadata["magnification"] or parse_magnification("自动", path)
            if inferred:
                self.after_magnification.set(f"{inferred:g}")
            pixel_text = metadata["pixel_size_x_um"]
            source = metadata.get("metadata_source") or "文件名"
            self.registration_status.set(
                f"After：{inferred:g}×，{pixel_text:.6g} µm/px（{source}）；请选择配准模式"
                if inferred and pixel_text else f"After：倍率={inferred or '未知'}（{source}）"
            )

    def run_pair_registration(self, matcher='classic'):
        batch=getattr(self,'batch_registration_window',None)
        if batch is not None and batch.winfo_exists() and batch.batch_is_busy():
            messagebox.showinfo('批量任务进行中','请先等待批量任务完成，或停止后续任务。')
            return
        if self.registration_busy:
            return
        before_path = Path(self.before_input_path.get().strip())
        after_path = Path(self.after_input_path.get().strip())
        if not before_path.exists() or not after_path.exists():
            messagebox.showwarning("输入不完整", "请先选择存在的 Before 和 After 图像。")
            return
        if not self.confirm_discard():
            return
        try:
            before_meta = read_sem_metadata(before_path)
            after_meta = read_sem_metadata(after_path)
            before_text = self.before_magnification.get()
            after_text = self.after_magnification.get()
            before_mag = (
                before_meta["magnification"] or parse_magnification("自动", before_path)
                if before_text.strip().lower() in {"", "auto", "自动"}
                else parse_magnification(before_text, before_path)
            )
            after_mag = (
                after_meta["magnification"] or parse_magnification("自动", after_path)
                if after_text.strip().lower() in {"", "auto", "自动"}
                else parse_magnification(after_text, after_path)
            )
        except RegistrationError as exc:
            messagebox.showerror("倍率错误", str(exc))
            return
        if before_mag is None or after_mag is None:
            messagebox.showerror(
                "倍率未知", "无法从.hdr或路径确认倍率。请在Before和After倍率框中手动填写后再配准。"
            )
            return
        before_px = before_meta["pixel_size_x_um"]
        after_px = after_meta["pixel_size_x_um"]
        magnification_ratio = max(before_mag, after_mag) / min(before_mag, after_mag)
        scale_basis = f"倍率 {before_mag:g}× / {after_mag:g}×"
        if before_px and after_px:
            scale_basis += f"；物理像素尺寸 {before_px:.6g} / {after_px:.6g} µm/px"
        analysis_mode = self.registration_mode.get()
        if analysis_mode == "同倍率变化" and magnification_ratio > 1.08:
            messagebox.showerror(
                "倍率层级不一致",
                f"当前是“同倍率变化”模式，但两图尺度不属于同一层级：\n{scale_basis}\n\n"
                "500×应与500×配准，2000×应与2000×配准。若只想寻找高倍图在低倍图中的位置，请改选“跨倍率定位”。",
            )
            return
        manual_points = None
        if matcher == 'manual':
            from manual_registration_ui import choose_landmarks
            try:
                manual_points = choose_landmarks(self.root, before_path, after_path)
            except Exception as exc:
                messagebox.showerror('人工辅助无法打开', str(exc))
                return
            if manual_points is None:
                return
        pair_id = "pair_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = REGISTRATION_RUNS_DIR / pair_id
        self.registration_busy = True
        self.registration_ready = False
        if self.sam_enabled.get():
            self.stop_sam_mode("输入已改变，等待新配准通过后再开始SAM2标注")
        self.registration_status.set("正在裁剪信息栏、匹配特征、RANSAC估计并进行ECC精修…")
        if matcher == 'lightglue':
            self.registration_status.set('增强匹配：CPU运行SuperPoint＋LightGlue；首次使用可能下载权重，请等待…')

        def work():
            try:
                result = register_pair(before_path, after_path, before_mag, after_mag, use_ecc=True,
                                       matcher=matcher, manual_points=manual_points)
                files = save_registration_result(result, output_dir)
                metrics = result.metrics

                def finish():
                    self.registration_busy = False
                    self.last_registration_report = files["metrics"]
                    median_text = (
                        f"{metrics.reprojection_median_px:.2f}px"
                        if np.isfinite(metrics.reprojection_median_px) else "无法计算"
                    )
                    p95_text = (
                        f"{metrics.reprojection_p95_px:.2f}px"
                        if np.isfinite(metrics.reprojection_p95_px) else "无法计算"
                    )
                    self.registration_status.set(
                        f"配准{metrics.quality}：内点 {metrics.inliers}，"
                        f"中位误差 {median_text}，P95 {p95_text}，重叠 {metrics.overlap_ratio:.1%}"
                    )
                    if metrics.quality not in {"优秀", "良好"}:
                        self.registration_ready = False
                        messagebox.showwarning(
                            "配准未通过质量门槛",
                            f"本次结果仅保存检查图和报告，不会载入Mask标注。\n\n"
                            f"内点：{metrics.inliers}\n覆盖率：{metrics.coverage_ratio:.1%}\n"
                            f"中位误差：{median_text}\nP95：{p95_text}\n"
                            f"原因：{metrics.warning or '质量指标不足'}\n\n"
                            f"检查目录：{output_dir}",
                        )
                        return
                    row = {
                        "pair_id": pair_id,
                        "before": files["before"],
                        "after": files["after"],
                        "diff": files["diff"],
                        "mask": files["mask"],
                        "manual_dir": str(output_dir / "manual_masks"),
                        "registration_metrics": files["metrics"],
                        "after_condition": "manual_pair",
                        "leaf": "",
                        "before_magnification": "" if before_mag is None else f"{before_mag:g}",
                        "after_magnification": "" if after_mag is None else f"{after_mag:g}",
                        "analysis_mode": analysis_mode,
                    }
                    self.unsaved = False
                    self.rows = [row]
                    self.idx = 0
                    self.load_current()
                    self.registration_ready = True
                    self.quality_status.set("合格" if metrics.quality in {"优秀", "良好"} else "配准差")
                    warning = f"\n\n注意：{metrics.warning}" if metrics.warning else ""
                    if getattr(result, 'manual_validation', None):
                        check = result.manual_validation
                        warning += (f"\n独立检查点中位误差：{check['check_median_px']:.2f}px"
                                    f"；P95：{check['check_p95_px']:.2f}px")
                    if analysis_mode == "跨倍率定位":
                        warning += "\n\n跨倍率结果只用于定位共同视野，不建议直接作为定量像素变化mask。"
                    messagebox.showinfo(
                        "配准完成",
                        f"质量：{metrics.quality}\n"
                        f"内点：{metrics.inliers}，覆盖率：{metrics.coverage_ratio:.1%}\n"
                        f"重投影中位误差：{median_text}\n"
                        f"重投影P95：{p95_text}\n"
                        f"结果目录：{output_dir}{warning}",
                    )

                self.root.after(0, finish)
            except Exception as exc:
                error_text = str(exc)

                def fail():
                    self.registration_busy = False
                    self.registration_ready = False
                    self.registration_status.set(f"配准失败：{error_text}")
                    messagebox.showerror(
                        "配准失败",
                        f"没有生成伪配准结果。请检查倍率、是否为同一区域以及图像纹理。\n\n{error_text}",
                    )

                self.root.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    def open_batch_registration(self, audit=False):
        if self.registration_busy:
            messagebox.showinfo('正在配准','请等待当前单对配准完成。')
            return
        from batch_registration_ui import open_batch_window
        existing=getattr(self,'batch_registration_window',None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            messagebox.showinfo('批量窗口已打开','请先完成或关闭当前批量窗口，再打开另一个入口。')
            return
        self.batch_registration_window=open_batch_window(self.root,audit=audit)

    def close_application(self):
        batch=getattr(self,'batch_registration_window',None)
        if batch is not None and batch.winfo_exists() and batch.batch_is_busy():
            messagebox.showinfo('批量任务进行中','请先在批量窗口停止后续任务，等待当前图对完成后再退出。')
            return
        if self.confirm_discard():
            self.root.destroy()

    def open_registration_report(self):
        if not self.last_registration_report or not Path(self.last_registration_report).exists():
            messagebox.showinfo("配准报告", "当前还没有通过本界面生成配准报告。")
            return
        os.startfile(str(self.last_registration_report))

    def pick_index(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.index_path.set(path)

    def load_index(self):
        if not self.confirm_discard():
            return
        path = Path(self.index_path.get())
        if not path.exists():
            self.info.set(f"找不到CSV: {path}")
            return
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = [resolve_row(row, path.parent) for row in csv.DictReader(f)]
        if not rows:
            messagebox.showinfo('无可标注图对', '此索引还没有通过筛选的图对，请查看数据集复核目录。')
            return
        status = self.status_filter.get().strip()
        if status and "status" in rows[0]:
            rows = [row for row in rows if row.get("status") == status]
        rejected_cross = 0
        rejected_unknown = 0
        scale_audited = []
        for row in rows:
            has_scale_fields = bool(
                row.get("before_raw") or row.get("after_raw")
                or row.get("before_magnification") or row.get("after_magnification")
                or row.get("magnification_group")
            )
            if not has_scale_fields:
                scale_audited.append(row)
                continue
            before_meta = read_sem_metadata(row.get("before_raw", "")) if row.get("before_raw") else {}
            after_meta = read_sem_metadata(row.get("after_raw", "")) if row.get("after_raw") else {}
            try:
                before_mag = float(row.get("before_magnification") or before_meta.get("magnification") or 0)
                after_mag = float(row.get("after_magnification") or after_meta.get("magnification") or 0)
            except (TypeError, ValueError):
                before_mag = after_mag = 0.0
            if before_mag > 0 and after_mag > 0:
                if max(before_mag, after_mag) / min(before_mag, after_mag) > 1.08:
                    rejected_cross += 1
                    continue
                row["before_magnification"] = f"{before_mag:g}"
                row["after_magnification"] = f"{after_mag:g}"
            elif row.get("magnification_group"):
                row["before_magnification"] = row["magnification_group"]
                row["after_magnification"] = row["magnification_group"]
            else:
                rejected_unknown += 1
                continue
            scale_audited.append(row)
        rows = scale_audited
        rows = [row for row in rows if Path(row.get("before", "")).exists() and Path(row.get("after", "")).exists() and Path(row.get("mask", "")).exists()]
        self.rows = rows
        self.idx = 0
        if not rows:
            messagebox.showwarning("无样本", "没有找到可用 before/after/mask。")
            return
        if rejected_cross or rejected_unknown:
            messagebox.showwarning(
                "已执行倍率安全筛选",
                f"为避免把不同尺度误当成像素变化，本次排除了：\n"
                f"跨倍率配对：{rejected_cross} 对\n倍率无法确认：{rejected_unknown} 对\n\n"
                f"当前保留 {len(rows)} 对可确认的同倍率样本。原CSV和原图未被修改。",
            )
        target = self.folder_navigation_target
        self.folder_navigation_target = None
        if target:
            target = Path(target).resolve()
            for position, row in enumerate(rows):
                annotation_dir = row.get("annotation_dir", "")
                if annotation_dir:
                    try:
                        if Path(annotation_dir).resolve() == target:
                            self.idx = position
                            break
                    except OSError:
                        pass
        self.load_current()
        self.root.title(f"{path.parent.name} 标注器 | 独立标签 | {len(self.rows)} 对")

    def load_current(self):
        self.stop_sam_mode("新图像已载入，请开启SAM标注")
        self.last_brush_point = None
        if not self.confirm_discard():
            return
        row = self.rows[self.idx]
        before = read_image(row["before"], grayscale=True)
        after = read_image(row["after"], grayscale=True)
        auto_mask = read_image(row["mask"], grayscale=True) > 0
        mask_path = self.manual_mask_path(row) if self.manual_mask_path(row).exists() else row["mask"]
        mask = read_image(mask_path, grayscale=True) > 0
        if row.get("diff") and Path(row["diff"]).exists():
            diff = read_image(row["diff"], grayscale=True)
        else:
            before_u8, after_u8 = normalize_u8(before), normalize_u8(after)
            before_tmp, after_tmp = align_shape(before_u8, after_u8)
            diff = cv2.absdiff(before_tmp, after_tmp)
        if row.get('dataset_version') and len({im.shape for im in (before,after,diff,auto_mask,mask)}) != 1:
            messagebox.showerror('坐标网格不一致','图像和Mask尺寸不一致，请核对任务来源。')
            return
        before, after, diff, auto_mask, mask = align_shape(before, after, diff, auto_mask, mask)
        try:
            annotations = AnnotationLayers(mask.shape, self.annotation_folder(row), legacy_binary=mask,
                                           filename_prefix=row.get('label_prefix'))
        except Exception as exc:
            messagebox.showerror('标注加载失败', str(exc))
            return
        self.before = before
        self.after = after
        self.diff = diff
        self.auto_mask = auto_mask.copy()
        self.annotations = None
        self.mask = mask.copy()
        self.original_mask = mask.copy()
        self.current_row = row
        self.registration_ready = True
        self.unsaved = False
        self.undo_stack = []
        self.redo_stack = []
        self.annotations = annotations
        key = self.annotation_side.get() if self.annotation_mode.get() == MODES[0] else (
            'binary' if self.annotation_mode.get() == MODES[1] else 'multi')
        self.active_annotation = key
        layer = self.annotations.layers[key]
        self.mask, self.undo_stack, self.redo_stack, self.original_mask = (
            layer['mask'], layer['undo'], layer['redo'], layer['original'])
        self.annotations.used.add(key)
        self.note_text.set("")
        self.quality_status.set('待复核' if row.get('registration_status') == 'needs_review' else '合格')
        if row.get('before_title'):
            self.before_pane.title.configure(text=row['before_title'])
            self.after_pane.title.configure(text=row['after_title'])
            self.before_input_path.set(row['before'])
            self.after_input_path.set(row['after'])
        self.zoom = 1.0
        self.center_x = before.shape[1] / 2.0
        self.center_y = before.shape[0] / 2.0
        self.cursor_xy = ""
        self.cursor_point = None
        self.sam_generation += 1
        self.sam_active_side = None
        self.sam_session_id = None
        self.sam_request_id = 0
        self.sam_active_roi = None
        self.sam_active_model = None
        self.sam_pending_side = None
        self.sam_prompts = {"before": [], "after": []}
        self.sam_candidates = {"before": None, "after": None}
        self.sam_scores = {"before": None, "after": None}
        self.sam_rois = {"before": None, "after": None}
        self.sam_candidate_options = {"before": [], "after": []}
        self.sam_candidate_indices = {"before": 0, "after": 0}
        self.sam_restart_required = {"before": False, "after": False}
        self.sam_status.set("SAM：点击 Before 或 After 添加提示")
        if row.get("registration_metrics"):
            self.last_registration_report = row["registration_metrics"]
            if row.get('dataset_version'):
                import json
                report=json.loads(Path(row['registration_metrics']).read_text(encoding='utf-8'))
                def error_text(key):
                    value=report.get(key)
                    return f'{value:.2f}px' if isinstance(value,(int,float)) and np.isfinite(value) else '不可计算'
                self.registration_status.set(
                    f"已载入配准图：{row.get('after_condition')}，{row.get('magnification_group')}倍；"
                    f"中位残差 {error_text('reprojection_median_px')}，P95 {error_text('reprojection_p95_px')}；"
                    + ('需复核' if row.get('registration_status')=='needs_review' else '自动筛选通过，标注前核对'))
        self.refresh_images()
        self.update_info()

    def update_info(self, include_metrics=True):
        if not self.current_row:
            return
        row = self.current_row
        if self.annotations is not None:
            layers = self.annotations.layers
            missing = [s for s in ('before', 'after') if not (self.annotations.folder / self.annotations.paths[s]).is_file()]
            label_state = '；缺少本项目标签：' + ', '.join(missing) + '（可手工或SAM标注）' if missing else ''
            self.mask_source_status.set(
                f"标签项目：{Path(self.index_path.get()).resolve().parent.name} | "
                f"蓝色 Before={int(np.count_nonzero(layers['before']['mask'])):,} px；"
                f"红色 After={int(np.count_nonzero(layers['after']['mask'])):,} px；紫色=重合"
                + label_state)
        area = int(np.count_nonzero(self.mask))
        pct = area / self.mask.size * 100.0
        saved = "有未保存修改" if self.unsaved else "无待保存修改"
        cursor = f"  坐标={self.cursor_xy}" if self.cursor_xy else ""
        sample = f"  {self.last_sample}" if self.last_sample else ""
        metric_text = ""
        if include_metrics and self.auto_mask is not None and self.active_annotation == 'binary':
            metrics = compare_masks(self.auto_mask, self.mask)
            metric_text = (
                f"  vs自动 Dice={metrics['manual_vs_auto_dice']:.3f} "
                f"IoU={metrics['manual_vs_auto_iou']:.3f} "
                f"增={metrics['added_area_px']}px 减={metrics['removed_area_px']}px"
            )
        self.info_base_text = (
            f"{self.idx + 1}/{len(self.rows)}  pair_id={row.get('pair_id')}  "
            f"condition={row.get('after_condition')}  leaf={row.get('leaf')}  "
            f"编辑={self.active_annotation} 面积={area} px ({pct:.4f}%)  zoom={self.zoom:.2f}x{metric_text}{sample}  {saved}"
        )
        if hasattr(self, "navigation_status"):
            self.navigation_status.set(f"第 {self.idx + 1} / 共 {len(self.rows)} 张")
        self.info.set(f"{self.info_base_text}{cursor}")

    def update_cursor_info(self):
        if not self.info_base_text:
            return
        cursor = f"  坐标={self.cursor_xy}" if self.cursor_xy else ""
        self.info.set(f"{self.info_base_text}{cursor}")

    def get_view(self):
        return self.zoom, self.center_x, self.center_y

    def fit_view(self):
        if self.before is None:
            return
        h, w = self.before.shape[:2]
        self.zoom = 1.0
        self.center_x = w / 2.0
        self.center_y = h / 2.0
        self.refresh_images()

    def refresh_panes(self):
        for pane in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
            pane.refresh()
        self.update_info()
        self.redraw_crosshair()

    def redraw_crosshair(self):
        """Restore the synchronized cursor after a canvas/image refresh."""
        if self.cursor_point is None:
            return
        ix, iy = self.cursor_point
        for pane in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
            pane.draw_crosshair(ix, iy)
        self.redraw_crop_window()

    def redraw_crop_window(self):
        if self.cursor_point is None or self.before is None or abs(self.zoom - 1.0) > 1e-6:
            for pane in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
                pane.clear_crop_window()
            return
        size = int(self.sam_roi_size.get())
        for pane in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
            pane.draw_crop_window(*self.cursor_point, size)

    def on_wheel(self, event):
        pane = self.get_event_pane(event)
        if pane is None or pane.base_pil is None:
            return
        point = pane.canvas_to_image(event.x, event.y)
        if point is None:
            point = (self.center_x, self.center_y)
        ix, iy = point
        old_zoom = self.zoom
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 64.0))
        if abs(self.zoom - old_zoom) < 1e-6:
            return
        if self.sam_enabled.get():
            self.stop_sam_mode("视野已改变；重新开启SAM后将按新视野编码")
        iw, ih = pane.base_pil.size
        cw = max(80, pane.canvas.winfo_width())
        ch = max(80, pane.canvas.winfo_height())
        fit_scale = min(cw / iw, ch / ih)
        new_scale = fit_scale * self.zoom
        self.center_x = float(np.clip(ix - (event.x - cw / 2.0) / new_scale, 0, iw))
        self.center_y = float(np.clip(iy - (event.y - ch / 2.0) / new_scale, 0, ih))
        self.refresh_panes()

    def on_pan_start(self, event):
        if self.sam_enabled.get():
            self.stop_sam_mode("视野平移；重新开启SAM后将按新视野编码")
        self.pan_last = (event.x, event.y, self.get_event_pane(event))

    def on_pan_drag(self, event):
        if self.pan_last is None:
            return
        last_x, last_y, pane = self.pan_last
        if pane is None or pane.scale <= 0 or pane.base_pil is None:
            return
        dx = event.x - last_x
        dy = event.y - last_y
        iw, ih = pane.base_pil.size
        self.center_x = float(np.clip(self.center_x - dx / pane.scale, 0, iw))
        self.center_y = float(np.clip(self.center_y - dy / pane.scale, 0, ih))
        self.pan_last = (event.x, event.y, pane)
        self.refresh_panes()

    def on_pan_end(self, _event):
        self.pan_last = None

    def refresh_images(self):
        if self.before is None:
            return
        self.commit_annotation()
        def blend(gray, colors):
            base = cv2.cvtColor(normalize_u8(gray), cv2.COLOR_GRAY2RGB)
            selected = np.any(colors != 0, axis=2)
            base[selected] = (base[selected]*(1-self.alpha.get()) + colors[selected]*self.alpha.get()).astype(np.uint8)
            return base
        def blend_rgb(base, colors):
            """Add a second registered layer without losing the original RGB image."""
            selected = np.any(colors != 0, axis=2)
            if np.any(selected):
                base = base.copy()
                base[selected] = (base[selected]*(1-self.alpha.get()) + colors[selected]*self.alpha.get()).astype(np.uint8)
            return base
        if self.annotation_mode.get() == MODES[0]:
            before = self.annotations.layers['before']['mask']
            after = self.annotations.layers['after']['mask']
            colors = pore_colors(before, after)
            before_rgb = blend(self.before, pore_colors(before, np.zeros_like(before)))
            after_rgb = blend(self.after, pore_colors(np.zeros_like(after), after))
            mask_view = colors
            self.before_pane.title.configure(text='Before原图 + Before孔隙Mask')
            self.after_pane.title.configure(text='After原图 + After孔隙Mask')
            self.mask_pane.title.configure(text=f'孔隙叠加：蓝Before / 红After / 紫重合；画笔={self.active_annotation}')
        else:
            colors = label_colors(self.mask) if self.active_annotation == 'multi' else binary_overlay_colors(self.mask)
            before_pore = self.annotations.layers['before']['mask']
            after_pore = self.annotations.layers['after']['mask']
            # Keep the mapped Aedge pore masks visible on their corresponding
            # registered original while also showing change labels.
            before_rgb = blend(self.before, colors)
            after_rgb = blend(self.after, colors)
            # Keep pore-only pixels visible without covering a confirmed
            # change-class color at the same registered coordinate.
            unchanged = ~np.any(colors != 0, axis=2)
            before_pore_colors = pore_colors(before_pore, np.zeros_like(before_pore))
            after_pore_colors = pore_colors(np.zeros_like(after_pore), after_pore)
            before_pore_colors[~unchanged] = 0
            after_pore_colors[~unchanged] = 0
            before_rgb = blend_rgb(before_rgb, before_pore_colors)
            after_rgb = blend_rgb(after_rgb, after_pore_colors)
            mask_view = colors if self.active_annotation == 'multi' else make_mask_view(self.mask)
            self.before_pane.title.configure(text='Before原图 + 已映射孔隙/变化Mask')
            self.after_pane.title.configure(text='After原图 + 已映射孔隙/变化Mask')
            self.mask_pane.title.configure(text='多分类变化：按类别颜色' if self.active_annotation == 'multi' else '二分类变化：黑0 / 白1')
        self.before_pane.set_image(before_rgb)
        self.after_pane.set_image(after_rgb)
        self.refresh_sam_overlay()
        self.mask_pane.set_image(mask_view)
        self.overlay_pane.title.configure(text=f'标注叠加 / 编辑：{self.active_annotation}')
        self.overlay_pane.set_image(blend(self.get_overlay_base_image(), colors))
        self.update_info()
        self.redraw_crosshair()

    def refresh_sam_overlay(self, side=None):
        sides = (side,) if side else ("before", "after")
        for current_side in sides:
            pane = self.before_pane if current_side == "before" else self.after_pane
            options = self.sam_candidate_options[current_side]
            index = self.sam_candidate_indices[current_side]
            feature = options[index].get('feature') if self.sam_candidates[current_side] is not None and index < len(options) else None
            pane.set_sam_overlay(
                self.sam_candidates[current_side],
                self.sam_prompts[current_side],
                self.sam_rois[current_side],
                self.sam_hover_points[current_side],
                feature=feature,
            )

    def refresh_mask_pane(self, include_overlay=True):
        if self.mask is None:
            return
        self.commit_annotation()
        if self.annotation_mode.get() == MODES[0]:
            mask_view = pore_colors(self.annotations.layers['before']['mask'], self.annotations.layers['after']['mask'])
        else:
            mask_view = label_colors(self.mask) if self.active_annotation == 'multi' else make_mask_view(self.mask)
        self.mask_pane.set_image(mask_view)
        if include_overlay:
            self.refresh_images()
        self.update_info(include_metrics=not self.drawing)
        if not include_overlay:
            self.redraw_crosshair()

    def schedule_mask_refresh(self):
        if self.refresh_pending:
            return
        self.refresh_pending = True
        self.root.after(45, self.flush_mask_refresh)

    def flush_mask_refresh(self):
        self.refresh_pending = False
        self.refresh_mask_pane(include_overlay=not self.fast_mode.get())

    def on_motion(self, event):
        pane = self.get_event_pane(event)
        if pane is None:
            return
        point = pane.canvas_to_image(event.x, event.y)
        if point is None:
            self.on_leave(event)
            return
        ix, iy = point
        self.cursor_xy = f"({ix}, {iy})"
        self.cursor_point = (ix, iy)
        # 四个视图使用同一配准坐标，十字线同步移动。
        for target in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
            target.draw_crosshair(ix, iy)
        self.redraw_crop_window()
        if self.sam_enabled.get() and pane in (self.before_pane, self.after_pane):
            side = "before" if pane == self.before_pane else "after"
            self.schedule_sam_hover(side, (ix, iy))
        self.update_cursor_info()

    def on_leave(self, event):
        pane = self.get_event_pane(event)
        if pane in (self.before_pane, self.after_pane):
            side = "before" if pane == self.before_pane else "after"
            self.sam_hover_points[side] = None
            if self.sam_hover_side == side:
                self.sam_hover_side = None
            if self.sam_hover_job is not None:
                self.root.after_cancel(self.sam_hover_job)
                self.sam_hover_job = None
            if not self.sam_prompts[side]:
                self.sam_candidates[side] = None
                self.sam_candidate_options[side] = []
                self.refresh_sam_overlay(side)
        for pane in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
            pane.clear_crosshair()
            pane.clear_crop_window()
        self.cursor_xy = ""
        self.cursor_point = None
        self.update_cursor_info()

    def schedule_sam_hover(self, side, point):
        self.sam_hover_points["after" if side == "before" else "before"] = None
        self.sam_hover_points[side] = point
        self.sam_hover_side = side
        # QuPath式合并队列：固定帧率提交最新位置，不等待鼠标完全停下。
        if self.sam_hover_job is None:
            self.sam_hover_job = self.root.after(50, self._submit_sam_hover)

    def _submit_sam_hover(self):
        self.sam_hover_job = None
        side = self.sam_hover_side
        point = self.sam_hover_points.get(side) if side else None
        if not self.sam_enabled.get() or side is None or point is None:
            return
        image = self.before if side == "before" else self.after
        if image is None:
            return
        roi = self.sam_rois[side]
        point_inside = roi and roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]
        if not point_inside:
            if self.sam_prompts[side]:
                return
            self.sam_rois[side] = self.sam_view_roi(side, point)
            self.sam_restart_required[side] = True
            self.sam_candidates[side] = None
            self.sam_candidate_options[side] = []
            self.refresh_sam_overlay(side)
        self.run_sam_prediction(side, hover_point=point)

    def sam_view_roi(self, side, point):
        pane = self.before_pane if side == "before" else self.after_pane
        bounds = pane.visible_bounds()
        image = self.before if side == "before" else self.after
        return sam_context_bounds(point, image.shape, bounds,
                                  minimum=64 if self.sam_fine_mode.get() else 256)

    def on_press(self, event):
        pane = self.get_event_pane(event)
        if pane in (self.before_pane, self.after_pane) and self.sam_enabled.get():
            point = pane.canvas_to_image(event.x, event.y)
            if point is not None:
                self.add_sam_prompt("before" if pane == self.before_pane else "after", point)
            return
        if self.annotation_mode.get() == MODES[0] and pane in (self.before_pane, self.after_pane):
            side = 'before' if pane == self.before_pane else 'after'
            if side != self.active_annotation:
                self.annotation_side.set(side)
                self.change_annotation_mode()
        if not self.can_draw_on(pane):
            return
        point = pane.canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.drawing = True
        self.last_brush_point = None
        self.push_history()
        self.paint(*point)

    def on_right_press(self, event):
        pane = self.get_event_pane(event)
        if pane in (self.before_pane, self.after_pane) and self.sam_enabled.get():
            self.pan_last = None
            point = pane.canvas_to_image(event.x, event.y)
            if point is not None:
                self.add_sam_prompt("before" if pane == self.before_pane else "after", point, label=0)
            return
        self.on_pan_start(event)

    def check_sam_service(self):
        if self.sam_busy:
            return
        self.sam_status.set("SAM API：检测中…")

        def work():
            try:
                result = self.sam_client.capabilities()
                models = result.get("interactive_models", [])
                if "sam2_s" in models:
                    message = "SAM2服务可用；完成配准后即可开始交互标注"
                else:
                    message = "SAM服务可用，但未发现默认模型 sam2_s"
            except Exception as exc:
                message = str(exc)
            self.root.after(0, lambda: self.sam_status.set(message))

        threading.Thread(target=work, daemon=True).start()

    def add_sam_prompt(self, side, point, label=None):
        if self.before is None:
            return
        image = self.before if side == "before" else self.after
        roi = self.sam_rois[side]
        if roi is None or not (roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]):
            roi = self.sam_view_roi(side, point)
            self.sam_rois[side] = roi
            self.sam_prompts[side] = []
            self.sam_candidate_options[side] = []
            self.sam_candidate_indices[side] = 0
            self.sam_restart_required[side] = True
        prompt_label = int(self.sam_prompt_label.get()) if label is None else int(label)
        if self.sam_hover_job is not None:
            self.root.after_cancel(self.sam_hover_job)
            self.sam_hover_job = None
        self.sam_hover_points[side] = None
        if self.sam_hover_side == side:
            self.sam_hover_side = None
        self.sam_prompts[side].append((point, prompt_label))
        self.sam_candidates[side] = None
        self.sam_pending_side = side
        self.refresh_sam_overlay(side)
        self.run_sam_prediction(side)

    def run_sam_prediction(self, side, hover_point=None):
        fixed_prompts = list(self.sam_prompts[side])
        prompts = list(fixed_prompts)
        if hover_point is not None:
            prompts.append((hover_point, 1))
        if not prompts:
            return
        self.sam_pending_side = side
        if self.sam_busy:
            self.sam_status.set(f"{side}：已记录新提示，等待当前预测完成…")
            return
        self.sam_pending_side = None
        image = self.before if side == "before" else self.after
        roi = self.sam_rois[side]
        if roi is None:
            roi = sam_roi_bounds(prompts[0][0], image.shape, self.sam_roi_size.get())
            self.sam_rois[side] = roi
        x0, y0, x1, y1 = roi
        local_prompts = [((x - x0, y - y0), label) for (x, y), label in prompts]
        render_scale = min(1.0, 2048.0 / max(x1-x0, y1-y0))
        render_w = max(1, round((x1-x0) * render_scale))
        render_h = max(1, round((y1-y0) * render_scale))
        sx, sy = render_w / (x1-x0), render_h / (y1-y0)
        rendered_prompts = [((min(render_w-1, x*sx), min(render_h-1, y*sy)), label)
                            for (x, y), label in local_prompts]
        model = self.sam_model.get()
        candidate_mode = self.sam_candidate_mode.get()
        generation = self.sam_generation
        can_reuse = (
            not self.sam_restart_required[side]
            and self.sam_active_side == side
            and self.sam_session_id
            and self.sam_active_roi == roi
            and self.sam_active_model == model
        )
        session_id_before = self.sam_session_id if can_reuse else None
        request_id_before = self.sam_request_id if can_reuse else 0
        if not can_reuse:
            self.sam_restart_required[side] = False
        self.sam_busy = True
        self.sam_status.set(
            f"{side}：{model} 正在预测 {x1-x0}×{y1-y0} 局部ROI（{len(prompts)}点）…"
        )

        def work():
            try:
                # 服务端一次只保留一个交互会话；切换 Before/After 时重新编码。
                if session_id_before is None:
                    roi_image = prepare_sam_roi(image[y0:y1, x0:x1])
                    if roi_image.shape[:2] != (render_h, render_w):
                        roi_image = cv2.resize(roi_image, (render_w, render_h), interpolation=cv2.INTER_AREA)
                    image_bytes = encode_png(roi_image)
                    started = self.sam_client.start_session(image_bytes, model_type=model)
                    session_id = started["session_id"]
                    request_id = 1
                else:
                    session_id = session_id_before
                    request_id = request_id_before + 1
                result = self.sam_client.predict(
                    session_id,
                    request_id,
                    [[float(x), float(y)] for (x, y), _ in rendered_prompts],
                    [prompt_label for _, prompt_label in rendered_prompts],
                    multimask=True,
                )
                features = result.get("masks", [])
                if not features:
                    raise SamApiError("SAM 没有返回候选轮廓")
                local_options = rasterize_sam_candidates(
                    features, (render_h, render_w), rendered_prompts, candidate_mode
                )
                full_options = []
                for item in local_options:
                    full_mask = np.zeros(image.shape[:2], dtype=bool)
                    local_feature = transform_sam_feature(item['feature'], sx, sy)
                    local_mask = geojson_to_mask([local_feature], (y1-y0, x1-x0))
                    full_mask[y0:y1, x0:x1] = local_mask
                    full_options.append({**item, "mask": full_mask, 'area': int(local_mask.sum()),
                                         'feature': transform_sam_feature(item['feature'], sx, sy, (x0,y0))})

                def finish():
                    if generation != self.sam_generation:
                        self.sam_busy = False
                        self._run_pending_sam()
                        return
                    self.sam_active_side = side
                    self.sam_session_id = session_id
                    self.sam_request_id = request_id
                    self.sam_active_roi = roi
                    self.sam_active_model = model
                    current_prompts = list(self.sam_prompts[side])
                    current_hover = self.sam_hover_points.get(side)
                    if current_hover is not None:
                        current_prompts.append((current_hover, 1))
                    current_key = (tuple(current_prompts), self.sam_rois[side], self.sam_model.get())
                    worker_key = (tuple(prompts), roi, model)
                    if current_key == worker_key:
                        self.sam_candidate_options[side] = full_options
                        self.sam_candidate_indices[side] = 0
                        self.sam_candidates[side] = full_options[0]["mask"]
                        self.sam_scores[side] = full_options[0]["score"]
                    self.sam_busy = False
                    if current_key == worker_key:
                        score = full_options[0]["score"]
                        score_text = "?" if score is None else f"{score:.3f}"
                        area = full_options[0]["area"]
                        action_text = "左键固定，右键排除" if hover_point is not None else "Enter接受"
                        self.sam_status.set(
                            f"{side} 实时预览，质量={score_text}，面积={area}px；{action_text}"
                        )
                    self.refresh_sam_overlay(side)
                    self._run_pending_sam()

                self.root.after(0, finish)
            except Exception as exc:
                error_text = str(exc)

                def fail():
                    self.sam_busy = False
                    if generation == self.sam_generation:
                        self.sam_restart_required[side] = True
                        self.sam_status.set(f"SAM预测失败：{error_text}")
                        messagebox.showerror(
                            "SAM预测失败",
                            f"SAM没有生成候选区域。\n\n{error_text}",
                        )
                    self._run_pending_sam()
                self.root.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    def _run_pending_sam(self):
        if self.sam_busy:
            return
        side = self.sam_pending_side
        self.sam_pending_side = None
        if side and (self.sam_prompts.get(side) or self.sam_hover_points.get(side) is not None):
            hover = self.sam_hover_points.get(side)
            generation = self.sam_generation
            def dispatch():
                if generation == self.sam_generation and self.sam_enabled.get():
                    self.run_sam_prediction(side, hover_point=hover)
            self.root.after(0, dispatch)

    def _selected_sam_side(self):
        if self.sam_active_side and self.sam_candidates.get(self.sam_active_side) is not None:
            return self.sam_active_side
        for side in ("after", "before"):
            if self.sam_candidates[side] is not None:
                return side
        return None

    def accept_sam_candidate(self):
        side = self._selected_sam_side()
        if side is None:
            messagebox.showinfo("SAM", "当前没有可接受的 SAM 候选。")
            return
        candidate = self.sam_candidates[side]
        if self.annotation_mode.get() == MODES[0]:
            self.annotation_side.set(side)
            self.select_annotation(side)
        self.push_history()
        mode = 'erase' if self.mode.get() == 'erase' else self.sam_apply_mode.get()
        value = self.annotation_value()
        if mode == "erase":
            self.mask[candidate] = False
        elif mode == "replace-local":
            ys, xs = np.where(candidate)
            if len(xs):
                margin = 10
                x0, x1 = max(0, xs.min() - margin), min(self.mask.shape[1], xs.max() + margin + 1)
                y0, y1 = max(0, ys.min() - margin), min(self.mask.shape[0], ys.max() + margin + 1)
                self.mask[y0:y1, x0:x1] = candidate[y0:y1, x0:x1].astype(self.mask.dtype) * value
        else:
            self.mask[candidate] = value
        self.unsaved = True
        self.sam_candidates[side] = None
        self.sam_candidate_options[side] = []
        self.sam_candidate_indices[side] = 0
        self.sam_prompts[side] = []
        self.sam_hover_points[side] = None
        self.sam_scores[side] = None
        # The installed API has no logits-reset endpoint. Start a fresh session
        # for the next object instead of carrying its predecessor's mask logits.
        self.sam_generation += 1
        self.sam_pending_side = None
        if self.sam_hover_job is not None:
            self.root.after_cancel(self.sam_hover_job)
            self.sam_hover_job = None
        self.sam_hover_side = None
        self.sam_restart_required[side] = True
        self.sam_status.set(
            f"已写入 {self.active_annotation} Mask；移动鼠标可继续预览下一个对象（Esc退出）"
        )
        self.refresh_images()

    def cycle_sam_candidate(self):
        side = self._selected_sam_side()
        if side is None or not self.sam_candidate_options[side]:
            messagebox.showinfo("SAM", "当前没有可切换的SAM候选。")
            return
        options = self.sam_candidate_options[side]
        index = (self.sam_candidate_indices[side] + 1) % len(options)
        self.sam_candidate_indices[side] = index
        item = options[index]
        self.sam_candidates[side] = item["mask"]
        self.sam_scores[side] = item["score"]
        score_text = "?" if item["score"] is None else f"{item['score']:.3f}"
        self.sam_status.set(
            f"{side} 候选{index + 1}/{len(options)}，质量={score_text}，面积={item['area']}px"
        )
        self.refresh_sam_overlay(side)

    def on_sam_accept_key(self, event):
        if event.widget.winfo_class() in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox"}:
            return
        if self.sam_enabled.get() and self._selected_sam_side() is not None:
            self.accept_sam_candidate()

    def on_sam_cancel_key(self, event):
        if event.widget.winfo_class() in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox"}:
            return
        if self.sam_enabled.get():
            self.stop_sam_mode("已取消当前候选并退出SAM2交互标注")

    def undo_sam_prompt(self):
        side = self.sam_pending_side or self.sam_active_side or ("after" if self.sam_prompts["after"] else "before")
        self.invalidate_sam_requests()
        if not self.sam_prompts[side]:
            return
        self.sam_prompts[side].pop()
        self.sam_candidates[side] = None
        self.sam_candidate_options[side] = []
        self.sam_candidate_indices[side] = 0
        self.sam_restart_required[side] = True
        self.refresh_sam_overlay(side)
        if self.sam_prompts[side]:
            self.sam_pending_side = side
            self.run_sam_prediction(side)
        else:
            self.sam_rois[side] = None
            self.sam_status.set(f"{side} 提示已清空")

    def clear_sam_prompts(self):
        side = self.sam_pending_side or self.sam_active_side or ("after" if self.sam_prompts["after"] else "before")
        self.invalidate_sam_requests()
        self.sam_prompts[side] = []
        self.sam_candidates[side] = None
        self.sam_scores[side] = None
        self.sam_candidate_options[side] = []
        self.sam_candidate_indices[side] = 0
        self.sam_rois[side] = None
        self.sam_restart_required[side] = True
        if self.sam_pending_side == side:
            self.sam_pending_side = None
        self.sam_status.set(f"{side} 提示和候选已清除")
        self.refresh_sam_overlay(side)

    def invalidate_sam_requests(self):
        self.sam_generation += 1
        self.sam_pending_side = None
        self.sam_hover_side = None
        self.sam_hover_points = {"before": None, "after": None}
        if self.sam_hover_job is not None:
            self.root.after_cancel(self.sam_hover_job)
            self.sam_hover_job = None
        self.sam_restart_required = {"before": True, "after": True}

    def on_drag(self, event):
        pane = self.get_event_pane(event)
        if not self.drawing or not self.can_draw_on(pane):
            return
        point = pane.canvas_to_image(event.x, event.y)
        if point is not None:
            self.paint(*point)

    def on_release(self, _event):
        if not self.drawing:
            return
        self.drawing = False
        self.last_brush_point = None
        self.refresh_mask_pane(include_overlay=True)

    def paint(self, ix, iy):
        if self.mask is None:
            return
        diameter = max(1, int(round(self.brush_size.get())))
        radius = diameter // 2
        previous = self.last_brush_point or (ix, iy)
        px, py = previous
        self.last_brush_point = (ix, iy)
        h, w = self.mask.shape
        x0, x1 = max(0, min(ix, px) - radius), min(w, max(ix, px) + radius + 1)
        y0, y1 = max(0, min(iy, py) - radius), min(h, max(iy, py) + radius + 1)
        # Connect event samples, then stamp an explicitly sized footprint.
        line = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.line(line, (px-x0, py-y0), (ix-x0, iy-y0), 1, 1)
        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        circle = cv2.dilate(line, footprint) > 0
        selected = circle
        if self.brush_kind.get() == "color-aware":
            ref = self.get_reference_image()
            patch = ref[y0:y1, x0:x1].astype(np.float32)
            values = patch[circle]
            if values.size:
                mean = float(values.mean())
                std = float(values.std())
                tol = float(self.color_tolerance.get())
                selected = np.logical_and(circle, np.abs(patch - mean) <= tol)
                self.last_sample = f"取色={self.reference_source.get()} mean={mean:.1f} std={std:.1f} 修改={int(selected.sum())}px"
        else:
            self.last_sample = f"普通笔触 修改={int(selected.sum())}px"
        sub = self.mask[y0:y1, x0:x1].copy()
        value = int(self.annotation_class.get().split()[0]) if getattr(self, 'active_annotation', 'binary') == 'multi' else 1
        sub[selected] = value if self.mode.get() == "add" else 0
        self.mask[y0:y1, x0:x1] = sub
        self.unsaved = True
        self.schedule_mask_refresh()

    def get_reference_image(self):
        source = self.reference_source.get()
        if source == "after":
            return normalize_u8(self.after)
        if source == "diff" and self.diff is not None:
            return normalize_u8(self.diff)
        return normalize_u8(self.before)

    def get_overlay_base_image(self):
        base = self.overlay_base.get()
        if base == "after":
            return self.after
        if base == "diff" and self.diff is not None:
            return self.diff
        return self.before

    def push_history(self):
        if self.mask is None:
            return
        if self.undo_stack and np.array_equal(self.undo_stack[-1], self.mask):
            return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.redo_stack = []

    def undo(self):
        if not self.undo_stack or self.mask is None:
            return
        self.redo_stack.append(self.mask.copy())
        self.mask = self.undo_stack.pop()
        self.unsaved = True
        self.last_sample = "撤销一步"
        self.refresh_mask_pane(include_overlay=True)

    def redo(self):
        if not self.redo_stack or self.mask is None:
            return
        self.undo_stack.append(self.mask.copy())
        self.mask = self.redo_stack.pop()
        self.unsaved = True
        self.last_sample = "重做一步"
        self.refresh_mask_pane(include_overlay=True)

    def can_draw_on(self, pane):
        target = self.draw_target.get()
        # In dual-pore mode the colored Before/After originals are editable
        # views of their selected layer, so blue/red masks can be corrected
        # directly on the corresponding registered image.
        if self.annotation_mode.get() == MODES[0]:
            if pane == self.before_pane and self.active_annotation == 'before':
                return True
            if pane == self.after_pane and self.active_annotation == 'after':
                return True
        if pane == self.mask_pane:
            return target in ["mask", "mask+overlay"]
        if pane == self.overlay_pane:
            return target in ["overlay", "mask+overlay"]
        return False

    def current_roi(self):
        if self.active_annotation == 'multi':
            messagebox.showinfo('多分类标注', '多分类请使用SAM或画笔指定类别；区域生长等二值工具仅用于孔隙/二分类模式。')
            return None
        if self.mask is None or self.cursor_point is None:
            messagebox.showwarning("缺少位置", "请先把鼠标移到要处理的孔洞/边缘附近，让十字线停在那里。")
            return None
        ix, iy = self.cursor_point
        radius = int(self.algorithm_radius.get())
        h, w = self.mask.shape
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        return ix, iy, x0, x1, y0, y1

    def _classical_processing_result(self):
        roi = self.current_roi()
        if roi is None:
            return None
        _ix, _iy, x0, x1, y0, y1 = roi
        source = {
            "before": self.before,
            "after": self.after,
            "diff": self.diff,
        }[self.classical_source.get()]
        patch = crop_image(source, x0, y0, x1, y1)
        result = process_roi(
            patch,
            dark_foreground=self.classical_polarity.get() == "dark",
            min_component_area=int(self.min_component_area.get()),
        )
        return (x0, x1, y0, y1), result

    def preview_classical_processing(self):
        processed = self._classical_processing_result()
        if processed is None:
            return
        _bounds, result = processed
        stages = [
            ("1 灰度化/裁剪", result["gray"]),
            ("2 中值去噪", result["denoised"]),
            ("3 锐化增强", result["enhanced"]),
            ("4 CLAHE", result["clahe"]),
            (f"5 Otsu (T={result['otsu_threshold']:.1f})", result["otsu"]),
            (f"6 Canny {result['canny_thresholds']}", result["canny"]),
            ("7 形态学开闭", result["morphology"]),
            (f"8 连通域 ({len(result['components'])})", result["components_mask"]),
        ]
        window = tk.Toplevel(self.root)
        window.title("OpenCV ROI处理预览")
        window.transient(self.root)
        photos = []
        for index, (title, image) in enumerate(stages):
            frame = ttk.Frame(window, padding=5)
            frame.grid(row=index // 4, column=index % 4, sticky="nsew")
            ttk.Label(frame, text=title).pack()
            pil = Image.fromarray(image).convert("L")
            pil.thumbnail((300, 260), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            photos.append(photo)
            ttk.Label(frame, image=photo).pack()
        window._processing_photos = photos
        components = result["components"]
        largest = int(components[0]["area"]) if components else 0
        ttk.Label(
            window,
            text=f"保留连通域 {len(components)} 个；最大面积 {largest} px；最小面积阈值 {self.min_component_area.get()} px",
            padding=8,
        ).grid(row=2, column=0, columnspan=4, sticky="w")

    def apply_classical_mask(self, action):
        processed = self._classical_processing_result()
        if processed is None:
            return
        (x0, x1, y0, y1), result = processed
        candidate = result["components_mask"] > 0
        if not candidate.any():
            messagebox.showinfo("OpenCV处理", "当前参数没有得到可用连通域。")
            return
        self.push_history()
        sub = self.mask[y0:y1, x0:x1].copy()
        if action == "erase":
            sub[candidate] = False
        else:
            sub[candidate] = True
        self.mask[y0:y1, x0:x1] = sub
        self.unsaved = True
        self.last_sample = (
            f"OpenCV-Otsu {('擦除' if action == 'erase' else '添加')} "
            f"阈值={result['otsu_threshold']:.1f} 连通域={len(result['components'])} "
            f"像素={int(candidate.sum())}"
        )
        self.refresh_mask_pane()

    def report_mask_components(self):
        if self.mask is None:
            return
        from image_processing import connected_components_analysis

        _filtered, _labels, components = connected_components_analysis(
            self.mask, min_area=int(self.min_component_area.get())
        )
        if not components:
            messagebox.showinfo("连通域统计", "当前mask没有达到最小面积的连通域。")
            return
        areas = np.asarray([item["area"] for item in components], dtype=np.float64)
        messagebox.showinfo(
            "连通域统计",
            f"数量：{len(components)}\n"
            f"总面积：{int(areas.sum())} px\n"
            f"最小/中位/最大：{int(areas.min())} / {np.median(areas):.1f} / {int(areas.max())} px",
        )

    def region_grow(self, action):
        roi = self.current_roi()
        if roi is None:
            return
        self.push_history()
        ix, iy, x0, x1, y0, y1 = roi
        ref = self.get_reference_image()
        patch = ref[y0:y1, x0:x1].astype(np.uint8)
        seed_x, seed_y = ix - x0, iy - y0
        if not (0 <= seed_x < patch.shape[1] and 0 <= seed_y < patch.shape[0]):
            return
        seed_value = int(patch[seed_y, seed_x])
        tol = int(self.color_tolerance.get())
        similar = np.abs(patch.astype(np.int16) - seed_value) <= tol
        if self.reference_source.get() == "diff" and action == "add":
            similar = np.logical_and(similar, patch >= max(12, seed_value - tol))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(similar.astype(np.uint8), connectivity=8)
        seed_label = int(labels[seed_y, seed_x])
        if seed_label <= 0 or seed_label >= num:
            messagebox.showinfo("区域生长", "当前落点没有形成可用的连通区域，请换一个点或调大容差。")
            return
        grown = labels == seed_label
        grown = self.refine_local_binary(grown)
        sub = self.mask[y0:y1, x0:x1].copy()
        sub[grown] = action == "add"
        self.mask[y0:y1, x0:x1] = sub
        self.unsaved = True
        self.last_sample = f"区域生长{('添加' if action == 'add' else '擦除')} seed={seed_value} 修改={int(grown.sum())}px"
        self.refresh_mask_pane()

    def edge_snap_roi(self):
        roi = self.current_roi()
        if roi is None:
            return
        self.push_history()
        _ix, _iy, x0, x1, y0, y1 = roi
        ref = self.get_reference_image()
        patch = ref[y0:y1, x0:x1].astype(np.uint8)
        local_mask = self.mask[y0:y1, x0:x1].copy()
        if not local_mask.any():
            messagebox.showinfo("边缘贴合", "当前局部区域还没有 mask。可以先粗画或用区域生长添加。")
            return
        band = cv2.dilate(local_mask.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1).astype(bool)
        values = patch[band]
        if values.size < 10:
            return
        if self.reference_source.get() == "diff":
            _, thresholded = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidate = thresholded > 0
        else:
            masked_mean = float(patch[local_mask].mean()) if local_mask.any() else float(values.mean())
            candidate = np.abs(patch.astype(np.float32) - masked_mean) <= max(8, float(self.color_tolerance.get()))
        candidate = np.logical_and(candidate, band)
        candidate = self.keep_near_existing(candidate, local_mask)
        candidate = self.refine_local_binary(candidate)
        self.mask[y0:y1, x0:x1] = np.logical_or(np.logical_and(local_mask, ~band), candidate)
        self.unsaved = True
        self.last_sample = f"边缘贴合 修改={int(candidate.sum())}px"
        self.refresh_mask_pane()

    def smooth_edges_roi(self):
        roi = self.current_roi()
        if roi is None:
            return
        self.push_history()
        _ix, _iy, x0, x1, y0, y1 = roi
        sub = self.mask[y0:y1, x0:x1]
        refined = self.refine_local_binary(sub)
        self.mask[y0:y1, x0:x1] = refined
        self.unsaved = True
        self.last_sample = f"平滑边缘 局部面积={int(refined.sum())}px"
        self.refresh_mask_pane()

    def remove_small_components_roi(self):
        roi = self.current_roi()
        if roi is None:
            return
        self.push_history()
        _ix, _iy, x0, x1, y0, y1 = roi
        sub = self.mask[y0:y1, x0:x1].astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        min_area = max(8, int((self.brush_size.get() ** 2) * 0.3))
        clean = np.zeros_like(sub)
        removed = 0
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                clean[labels == label] = 1
            else:
                removed += area
        self.mask[y0:y1, x0:x1] = clean.astype(bool)
        self.unsaved = True
        self.last_sample = f"去小斑点 删除={removed}px min_area={min_area}"
        self.refresh_mask_pane()

    def fill_holes_roi(self):
        roi = self.current_roi()
        if roi is None:
            return
        self.push_history()
        _ix, _iy, x0, x1, y0, y1 = roi
        sub = self.mask[y0:y1, x0:x1].astype(np.uint8)
        inv = (1 - sub).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        filled = sub.copy()
        changed = 0
        max_hole = max(20, int(self.algorithm_radius.get() * self.algorithm_radius.get() * 0.08))
        h, w = sub.shape
        for label in range(1, num):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
            if not touches_border and area <= max_hole:
                filled[labels == label] = 1
                changed += area
        self.mask[y0:y1, x0:x1] = filled.astype(bool)
        self.unsaved = True
        self.last_sample = f"填小孔 填充={changed}px max_hole={max_hole}"
        self.refresh_mask_pane()

    def refine_local_binary(self, binary):
        arr = binary.astype(np.uint8) * 255
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel5, iterations=1)
        arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel3, iterations=1)
        return arr > 0

    def keep_near_existing(self, candidate, existing):
        num, labels, _stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(candidate, dtype=bool)
        grown_existing = cv2.dilate(existing.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
        for label in range(1, num):
            comp = labels == label
            if np.logical_and(comp, grown_existing).any():
                keep[comp] = True
        return keep

    def get_event_pane(self, event):
        for pane in [self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane]:
            if event.widget == pane.canvas:
                return pane
        return None

    def manual_mask_path(self, row):
        pair_id = row.get("pair_id") or Path(row.get("mask", "mask")).stem
        manual_dir = self.annotation_folder(row) / 'manual_masks'
        return manual_dir / f"{pair_id}_manual_mask.png"

    def current_manual_dir(self):
        if self.annotations is not None:
            return self.annotations.folder
        return Path(__file__).resolve().parent / 'annotation_projects'

    def export_current_crop(self):
        """Export a paired native-pixel crop centred on the synchronized cursor."""
        if self.before is None or self.after is None or self.cursor_point is None or not self.current_row:
            messagebox.showwarning('无法裁切', '请先加载任务，并把鼠标移到四格图像中的目标位置。')
            return
        if self.before.shape[:2] != self.after.shape[:2]:
            messagebox.showwarning('尺寸不符', 'Before 和 After 尺寸不一致，无法使用同一坐标裁切。')
            return
        size = int(self.sam_roi_size.get())
        if size > 1024:
            messagebox.showwarning('窗口过大', '裁切窗口不能超过原图尺寸。')
            return
        x0, y0, x1, y1 = sam_roi_bounds(self.cursor_point, self.before.shape, size)
        pair = str(self.current_row.get('pair_id', f'item_{self.idx+1}'))
        safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in pair)[:80]
        folder = Path(self.index_path.get()).resolve().parent / '裁切结果' / safe / f'crops_{size}'
        folder.mkdir(parents=True, exist_ok=True)
        stem = f'{safe}_x{x0}_y{y0}_{size}'
        write_png(folder / f'{stem}_before.png', self.before[y0:y1, x0:x1])
        write_png(folder / f'{stem}_after.png', self.after[y0:y1, x0:x1])
        if self.annotations is not None:
            layers = self.annotations.layers
            write_png(folder / f'{stem}_before_mask.png', layers['before']['mask'][y0:y1, x0:x1].astype(np.uint8) * 255)
            write_png(folder / f'{stem}_after_mask.png', layers['after']['mask'][y0:y1, x0:x1].astype(np.uint8) * 255)
            write_png(folder / f'{stem}_multiclass.png', layers['multi']['mask'][y0:y1, x0:x1].astype(np.uint8))
        import json
        (folder / f'{stem}.json').write_text(json.dumps({
            'pair_id': self.current_row.get('pair_id'), 'x0': x0, 'y0': y0,
            'x1': x1, 'y1': y1, 'size': size,
            'before_shape': list(self.before.shape), 'after_shape': list(self.after.shape),
            'coordinate_system': 'registered native pixels'
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        self.info.set(f'已导出 {size}×{size} 配对裁切：{folder}')

    def save_mask(self):
        if self.mask is None or not self.current_row:
            return
        import json
        self.commit_annotation()
        folder = self.annotations.folder
        image_prefix = (self.current_row.get('label_prefix') + '_') if self.current_row.get('label_prefix') else ''
        try:
            keys = self.annotations.save()
            layers = self.annotations.layers
            if keys & {'before', 'after'}:
                Image.fromarray(pore_colors(layers['before']['mask'], layers['after']['mask'])).save(folder/self.annotations.pore_root/'overlay_blue_red.png')
            if 'multi' in keys:
                Image.fromarray(label_colors(layers['multi']['mask'])).save(folder/'change_multiclass'/f'{image_prefix}preview_rgb.png')
            if 'binary' in keys:
                Image.fromarray(layers['binary']['mask'].astype(np.uint8)*255).save(folder/'change_binary'/f'{image_prefix}preview.png')
            manifest = {'version': 1, 'row': portable_row(self.current_row, folder), 'mode': self.annotation_mode.get(),
                        'annotation_status': 'in_progress',
                        'side': self.annotation_side.get(), 'class': self.annotation_class.get(),
                        'saved_at': dt.datetime.now().isoformat(timespec='seconds'),
                        'quality': self.quality_status.get(), 'note': self.note_text.get(),
                        'label_counts': {key: {str(int(v)): int(n) for v,n in zip(*np.unique(layers[key]['mask'], return_counts=True))} for key in keys},
                        'meaning': '双侧非重叠仅为候选变化；多分类由人工确认。标签在共同配准网格中。'}
            (folder/'annotation_project.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:
            messagebox.showerror('保存失败', str(exc))
            return
        self.unsaved = False
        self.update_info()
        messagebox.showinfo("已保存", f"各模式标注已分别保存：\n{folder}")

    def write_annotation_log(self, mask_path, overlay_before, overlay_after, overlay_diff):
        if self.current_row is None or self.auto_mask is None:
            return
        manual_dir = self.current_manual_dir()
        manual_dir.mkdir(parents=True, exist_ok=True)
        annotation_log = manual_dir / "manual_annotation_log.csv"
        metrics = compare_masks(self.auto_mask, self.mask)
        fieldnames = [
            "timestamp",
            "pair_id",
            "condition",
            "leaf",
            "quality_status",
            "note",
            "auto_area_px",
            "manual_area_px",
            "added_area_px",
            "removed_area_px",
            "area_delta_px",
            "area_delta_percent_of_auto",
            "manual_vs_auto_iou",
            "manual_vs_auto_dice",
            "mask_path",
            "overlay_before",
            "overlay_after",
            "overlay_diff",
            "registration_metrics",
            "before_magnification",
            "after_magnification",
        ]
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "pair_id": self.current_row.get("pair_id"),
            "condition": self.current_row.get("after_condition"),
            "leaf": self.current_row.get("leaf"),
            "quality_status": self.quality_status.get(),
            "note": self.note_text.get(),
            "mask_path": str(mask_path),
            "overlay_before": str(overlay_before),
            "overlay_after": str(overlay_after),
            "overlay_diff": str(overlay_diff),
            "registration_metrics": self.current_row.get("registration_metrics", ""),
            "before_magnification": self.current_row.get("before_magnification", ""),
            "after_magnification": self.current_row.get("after_magnification", ""),
            **metrics,
        }
        write_header = not annotation_log.exists()
        with annotation_log.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def reset_mask(self):
        if self.original_mask is not None:
            self.push_history()
            self.mask = self.original_mask.copy()
            self.unsaved = True
            self.refresh_images()

    def confirm_discard(self):
        if not self.unsaved:
            return True
        return messagebox.askyesno("未保存修改", "当前 mask 有未保存修改，是否放弃并切换？")

    def prev_item(self):
        if not self.rows:
            return
        if not self.confirm_discard():
            return
        self.unsaved = False
        self.idx = max(0, self.idx - 1)
        self.load_current()

    def next_item(self):
        if not self.rows:
            return
        if not self.confirm_discard():
            return
        self.unsaved = False
        self.idx = min(len(self.rows) - 1, self.idx + 1)
        self.load_current()

    def open_manual_dir(self):
        manual_dir = self.current_manual_dir()
        manual_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(manual_dir))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path)
    args = parser.parse_args()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = MaskEditorApp(root)
    if args.index:
        app.index_path.set(str(args.index.resolve()))
        app.status_filter.set('ok')
        root.after(100, app.load_index)
    root.protocol('WM_DELETE_WINDOW', app.close_application)
    root.mainloop()


if __name__ == "__main__":
    main()
