"""Bridge the registration grid to immutable two-source annotation candidates."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from registration_pipeline import (
    register_arrays, read_image_unicode, read_sem_metadata, parse_magnification,
    save_registration_result, write_png_unicode, normalize_registration_image,
)

SIDES = ('before', 'after')
SOURCES = ('Aedge', 'SAM')


@dataclass
class PreparedPair:
    result: object
    masks: dict
    provenance: dict


def align_candidate(mask, side, result, raw_shape):
    """Original-image binary labels -> cropped, sampled common core; nearest only."""
    if side not in SIDES:
        raise ValueError('Unknown candidate side')
    mask = np.asarray(mask)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(raw_shape):
        raise ValueError(f'{side} candidate must match its original image shape {raw_shape}')
    if not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError('Candidate must be a binary mask with values 0/1 or 0/255')
    bounds = getattr(result, side + '_crop_bounds_xyxy')
    x0, y0, x1, y1 = bounds
    cropped = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    matrix = result.before_sampling if side == 'before' else result.transform_after_to_before
    h, w = result.before_crop.shape
    aligned = cv2.warpPerspective(cropped, np.asarray(matrix, np.float64), (w, h),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    x0, y0, x1, y1 = result.core_bounds_xyxy
    return (aligned[y0:y1, x0:x1] > 0) & result.overlap_core


def _fingerprint(path):
    path = Path(path)
    return {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_pair(before, after, candidates=None, before_magnification=None,
                 after_magnification=None, use_ecc=True, crop_footer=True):
    """Register originals once and apply precisely that geometry to their candidates."""
    paths = {'before': Path(before), 'after': Path(after)}
    images = {s: read_image_unicode(paths[s]) for s in SIDES}
    metadata = {s: read_sem_metadata(paths[s]) for s in SIDES}
    mags = {}
    for side, requested in zip(SIDES, (before_magnification, after_magnification)):
        auto = requested is None or str(requested).strip() in ('', '自动', 'auto')
        mags[side] = metadata[side]['magnification'] if auto else parse_magnification(requested)
        if not auto and (mags[side] is None or mags[side] <= 0):
            raise ValueError('倍率应为正数或“自动”')
        px = metadata[side].get('pixel_size_x_um')
        py = metadata[side].get('pixel_size_y_um')
        if px and py and abs(px / py - 1) > .01:
            raise ValueError('当前尺度统一要求近似等距像素；请先核对 PixelSizeX/Y')
    result = register_arrays(images['before'], images['after'], mags['before'], mags['after'],
        before_pixel_size_um=metadata['before']['pixel_size_x_um'],
        after_pixel_size_um=metadata['after']['pixel_size_x_um'], use_ecc=use_ecc,
        crop_footer=crop_footer)
    masks = {source: {} for source in SOURCES}
    records = {}
    for source, sides in (candidates or {}).items():
        if source not in SOURCES or set(sides) - set(SIDES):
            raise ValueError('候选来源仅支持 Aedge / SAM 槽位及 before / after 两侧')
        for side, path in sides.items():
            if not path:
                continue
            with Image.open(path) as image:
                raw = np.asarray(image).copy()
            masks[source][side] = align_candidate(raw, side, result, images[side].shape)
            records[f'{source}/{side}'] = _fingerprint(path)
    return PreparedPair(result, masks, {
        'originals': {s: _fingerprint(paths[s]) for s in SIDES},
        'original_shapes_hw': {s: list(images[s].shape) for s in SIDES},
        'candidate_files': records, 'candidate_input_grid': 'original image pixels',
        'candidate_interpolation': 'nearest', 'SAM_slot': 'user-supplied AP-SAM or fusion candidate',
        'model_inference_performed': False,
    })


def comparison_previews(result):
    """Display before versus moving image before alignment and on the final core."""
    ref = normalize_registration_image(result.before_crop)
    mov = normalize_registration_image(result.after_crop)
    h, w = ref.shape
    raw_canvas = np.zeros_like(ref)
    ch, cw = min(h, mov.shape[0]), min(w, mov.shape[1])
    raw_canvas[:ch, :cw] = mov[:ch, :cw]
    raw = np.stack((raw_canvas, ref, raw_canvas), axis=-1)
    b = normalize_registration_image(result.before_core)
    a = normalize_registration_image(result.after_core)
    aligned = np.stack((a, b, a), axis=-1)
    yy, xx = np.indices(b.shape)
    checker = b.copy()
    choose = ((xx // 32 + yy // 32) % 2) == 1
    checker[choose] = a[choose]
    return {'before_alignment': raw, 'after_alignment': aligned, 'checkerboard': checker}


def export_workspace(prepared, destination, reviewed=False, pair_id='pair_001'):
    """Create a fresh workspace only after human review; never overwrite source data."""
    if not reviewed:
        raise ValueError('请先检查配准预览并确认图对和有效重叠区域')
    if not pair_id or Path(pair_id).name != pair_id or any(c in pair_id for c in '/\\:') or pair_id in ('.', '..'):
        raise ValueError('Invalid pair identifier')
    result = prepared.result
    mags = (result.metrics.before_magnification, result.metrics.after_magnification)
    if all(mags) and max(mags) / min(mags) > 1.08:
        raise ValueError('不同倍率图对仅用于定位，请用同倍率图对建立前后变化标注任务')
    scales = (result.metrics.before_pixel_size_um, result.metrics.after_pixel_size_um)
    if all(scales) and max(scales) / min(scales) > 1.08:
        raise ValueError('物理像素尺度差异较大，请先核对成像尺度和分析用途')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    task = destination / 'ready_to_annotate' / pair_id
    files = save_registration_result(result, task)
    for source in SOURCES:
        for side, mask in prepared.masks[source].items():
            write_png_unicode(task/'source_masks'/source/f'{side}_pore_mask.png', mask.astype(np.uint8))
    for side in SIDES:
        initial = prepared.masks['SAM'].get(side, np.zeros(result.before_core.shape, bool))
        write_png_unicode(task/'pore_masks'/f'{side}_pore_mask.png', initial.astype(np.uint8))
    provenance = dict(prepared.provenance, pair_id=pair_id, review={
        'registration_checked': True, 'segmentation_verified': False,
        'note': 'Registration approval does not certify segmentation or scientific change.'})
    (task/'workflow_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    rel = task.relative_to(destination).as_posix()
    row = dict(pair_id=pair_id, before=f'{rel}/{Path(files["before"]).name}',
        after=f'{rel}/{Path(files["after"]).name}', mask=f'{rel}/{Path(files["mask"]).name}',
        diff=f'{rel}/{Path(files["diff"]).name}', annotation_dir=rel, status='ok',
        dataset_version='registered-fusion-v1', label_prefix='',
        registration_metrics=f'{rel}/registration_metrics.json',
        registration_status='reviewed', after_condition='After',
        magnification_group=str(result.metrics.target_magnification or ''))
    with (destination/'annotation_index.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    return destination
