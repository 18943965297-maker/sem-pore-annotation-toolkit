"""Named, independent fusion outputs, separate from input tasks and source masks."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import io
import json
import uuid
import numpy as np
from PIL import Image
from annotation_modes import pore_colors


def output_paths(project, row):
    root = Path(project).resolve()/'new integrate mask'
    pair = str(row['pair_id'])
    if Path(pair).name != pair or pair in ('.', '..'):
        raise ValueError('Invalid pair name')
    return {'root': root,
            'before': root/'Before'/(Path(row['before']).stem+'_mask.png'),
            'after': root/'After'/(Path(row['after']).stem+'_mask.png'),
            'preview': root/'Preview'/(pair+'_overlay.png'),
            'metadata': root/'Metadata'/(pair+'.json')}


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_bytes(data)
        temp.replace(path)
    finally:
        if temp.exists(): temp.unlink()


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def save_integrated(project, row, masks, state=None):
    paths = output_paths(project, row)
    masks = {side: np.asarray(masks[side], dtype=bool) for side in ('before','after')}
    if masks['before'].ndim != 2 or masks['before'].shape != masks['after'].shape:
        raise ValueError('Before/After mask shape mismatch')
    payloads = {side: png_bytes(masks[side].astype(np.uint8)) for side in masks}
    payloads['preview'] = png_bytes(pore_colors(masks['before'],masks['after']))
    record = {'pair_id':row['pair_id'], 'saved_at':dt.datetime.now().isoformat(),
              'shape_hw':list(masks['before'].shape), 'mask_values':[0,1], 'state':state or {},
              'input_task':str(Path(row['annotation_dir']).resolve().relative_to(Path(project).resolve()))}
    for side in masks:
        record[side+'_image'] = Path(row[side]).name
        record[side+'_mask'] = paths[side].relative_to(paths['root']).as_posix()
        record[side+'_sha256'] = hashlib.sha256(payloads[side]).hexdigest()
        record[side+'_pixels'] = int(masks[side].sum())
    for key, data in payloads.items(): atomic_write(paths[key], data)
    atomic_write(paths['metadata'], json.dumps(record,ensure_ascii=False,indent=2).encode('utf-8'))
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((paths['root']/'Metadata').glob('*.json'))]
    fields = ['pair_id','before_image','after_image','before_mask','after_mask','before_pixels','after_pixels','saved_at']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(records)
    atomic_write(paths['root']/'mask_index.csv',buffer.getvalue().encode('utf-8-sig'))
    atomic_write(paths['root']/'last_session.json',json.dumps(record,ensure_ascii=False,indent=2).encode('utf-8'))
    return paths


def load_integrated(project, row, shape):
    paths = output_paths(project,row)
    present = [paths[key].is_file() for key in ('before','after','metadata')]
    if not any(present): return None
    if not all(present): raise ValueError('独立输出不完整，请检查 new integrate mask 中的前后标签和对应记录。')
    record = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    if record['pair_id'] != row['pair_id']: raise ValueError('保存标签与当前图对名称不符')
    masks = {}
    for side in ('before','after'):
        if record[side+'_image'] != Path(row[side]).name: raise ValueError('保存标签与原图文件名不符')
        if hashlib.sha256(paths[side].read_bytes()).hexdigest() != record[side+'_sha256']:
            raise ValueError('独立输出已在外部改变或保存不完整，请核对：'+str(paths[side]))
        with Image.open(paths[side]) as im: mask = np.asarray(im).copy()
        if mask.shape != tuple(shape): raise ValueError('保存标签与原图尺寸不符')
        masks[side] = mask > 0
    return masks,record


def migrate_legacy(project):
    from dataset_paths import resolve_row
    project = Path(project).resolve()
    rows = [resolve_row(r,project) for r in csv.DictReader((project/'annotation_index.csv').open(encoding='utf-8-sig'))]
    migrated = []
    for row in rows:
        task = Path(row['annotation_dir'])
        state_file = task/'fusion_edit_state.json'
        if not state_file.exists() or output_paths(project,row)['metadata'].exists(): continue
        state = json.loads(state_file.read_text(encoding='utf-8'))
        state['migrated_from'] = str(task/'pore_masks')
        masks = {}
        for side in ('before','after'):
            with Image.open(task/'pore_masks'/f'{side}_pore_mask.png') as im: masks[side] = np.asarray(im).copy() > 0
        save_integrated(project,row,masks,state)
        migrated.append(row['pair_id'])
    return migrated
