"""Independent annotation layers and lossless label persistence."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

MODES = ('双侧孔隙', '变化二分类', '变化多分类')
CLASSES = {0: '未变化', 1: '新增孔隙', 2: '孔隙扩大', 3: '孔隙缩小/闭合',
           4: '新增裂缝', 255: '不确定/忽略'}
COLORS = {1: (255, 70, 60), 2: (255, 175, 0), 3: (40, 140, 255),
          4: (50, 220, 100), 255: (150, 150, 150)}
PATHS = {'before': 'pore_masks/before_pore_mask.png',
         'after': 'pore_masks/after_pore_mask.png',
         'binary': 'change_binary/mask.png', 'multi': 'change_multiclass/mask.png'}


class AnnotationLayers:
    def __init__(self, shape, folder, legacy_binary=None, filename_prefix=None):
        self.folder = Path(folder)
        # Each project owns its pore_masks. Never substitute labels from
        # another model's backup directory when this project's labels lack.
        self.pore_root = 'pore_masks'
        self.paths = dict(PATHS)
        self.paths['before'] = f'{self.pore_root}/before_pore_mask.png'
        self.paths['after'] = f'{self.pore_root}/after_pore_mask.png'
        if filename_prefix:
            if Path(filename_prefix).name != filename_prefix:
                raise ValueError('Mask名称前缀不能包含目录')
            self.paths = {
                'before': f'{self.pore_root}/{filename_prefix}_Before孔隙Mask.png',
                'after': f'{self.pore_root}/{filename_prefix}_After孔隙Mask.png',
                'binary': f'change_binary/{filename_prefix}_变化二分类Mask.png',
                'multi': f'change_multiclass/{filename_prefix}_变化多分类Mask.png',
            }
        self.layers = {}
        self.used = set()
        for key, relative in self.paths.items():
            path = self.folder / relative
            mask = np.zeros(shape, np.uint8 if key == 'multi' else bool)
            # Registered tasks store pore masks under stable short filenames
            # (before_pore_mask.png / after_pore_mask.png). Try the optional
            # row-specific names first, then fall back to those actual files.
            if filename_prefix and key in ('before', 'after'):
                fallback_rel = f'{self.pore_root}/{key}_pore_mask.png'
            else:
                fallback_rel = PATHS[key]
            fallback = self.folder / fallback_rel
            if filename_prefix and (len(str(path)) > 240 or not path.exists()):
                path = fallback
                self.paths[key] = fallback_rel
            if path.exists():
                with Image.open(path) as image:
                    loaded = np.asarray(image).copy()
                if loaded.shape != shape:
                    raise ValueError(f'标注尺寸与配准图不一致：{path}')
                valid = set(CLASSES) if key == 'multi' else {0, 1}
                if not set(np.unique(loaded)).issubset(valid):
                    raise ValueError(f'标注含不支持的类别编号：{path}')
                mask = loaded.astype(mask.dtype)
                self.used.add(key)
            elif key == 'binary' and legacy_binary is not None:
                mask = legacy_binary.astype(bool).copy()
            self.layers[key] = {'mask': mask, 'undo': [], 'redo': [], 'original': mask.copy()}

    def save(self):
        keys = set(self.used)
        if keys & {'before', 'after'}:
            keys.update(('before', 'after'))
        for key in keys:
            path = self.folder / self.paths[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.layers[key]['mask'].astype(np.uint8)).save(path)
        if 'multi' in keys:
            (self.folder / 'change_multiclass/classes.json').write_text(
                json.dumps({'classes': CLASSES, 'colors_rgb': COLORS,
                            'extent': '变化区域本身，不是整个孔隙对象'}, ensure_ascii=False, indent=2),
                encoding='utf-8')
        if 'binary' in keys:
            (self.folder / 'change_binary/classes.json').write_text(
                json.dumps({0: '未变化', 1: '变化'}, ensure_ascii=False), encoding='utf-8')
        return keys


def pore_colors(before, after):
    result = np.zeros((*before.shape, 3), np.uint8)
    result[before & ~after] = (40, 120, 255)
    result[after & ~before] = (255, 60, 60)
    result[before & after] = (190, 60, 230)
    return result


def label_colors(mask):
    result = np.zeros((*mask.shape, 3), np.uint8)
    for value, color in COLORS.items():
        result[mask == value] = color
    return result


def binary_overlay_colors(mask):
    """Display-only orange-red; binary label values remain unchanged."""
    result = np.zeros((*mask.shape, 3), np.uint8)
    result[mask.astype(bool)] = (255, 85, 35)
    return result
