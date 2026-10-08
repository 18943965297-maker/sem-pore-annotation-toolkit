# Preparing a local workspace

The editor consumes precomputed binary masks; it does not train or download models.
Generate the synthetic workspace first to inspect the file layout. Every image in
that demo is algorithmically generated geometry, not an experimental observation.

Each CSV row in `annotation_index.csv` needs `pair_id`, `before`, `after`, `mask`,
`annotation_dir`, `status`, `dataset_version`, and `label_prefix`. Paths are relative
to the CSV directory. Use a unique safe filename for `pair_id`, `status=ok`, a
nonempty `dataset_version` (enables equal-shape validation), and an empty
`label_prefix`. The `mask` column points to a binary change mask required by the
base editor; an all-zero image is sufficient when only pore masks are reviewed.

For each task under `ready_to_annotate/<pair_id>/`, provide:

```
before.png
after.png
change_mask.png
pore_masks/before_pore_mask.png
pore_masks/after_pore_mask.png
source_masks/Aedge/before_pore_mask.png
source_masks/Aedge/after_pore_mask.png
source_masks/SAM/before_pore_mask.png
source_masks/SAM/after_pore_mask.png
```

All arrays must have the same height/width and coordinate grid. Initial pore
masks must use 0/1 values, with 1 denoting pore. Source masks may use 0/1 or 0/255.
Initialize pore masks from a selected candidate or zeros. Missing source masks
are permitted but unavailable for regional fusion. Existing workspaces are never
automatically imported from the author's experiment directories.

The CSV may also contain `before_magnification` and `after_magnification`.
If supplied, mismatched magnifications are filtered by the inherited editor.
The check is not registration validation: users must verify the spatial alignment.
Document pixel spacing and resizing separately; this release does not enforce
physical-scale calibration or automatically establish correspondence between pores.

## Editing

1. Select the Before or After side in the left two panes.
2. Use the ROI tool (框选融合区域) and choose Aedge or SAM.
3. Local addition (局部补充) only adds foreground; replacement (局部替换) can
   both add and remove foreground inside the ROI.
4. Use the brush or connected-component deletion for manual corrections.
   Deletion uses 8-connectivity, including diagonally touching pixels.
5. Ctrl+Z / Ctrl+Y undo and redo; Ctrl+S saves.

Dual-side deletion uses the same coordinate independently on both masks; it does
not identify matching physical pores. Check each side before accepting changes.

Outputs go to `new integrate mask/Before`, `After`, `Preview`, and `Metadata`,
with `mask_index.csv` and `last_session.json`. These files stay local and are
excluded from publication. The tool reloads saved output to resume editing.

## Optional interactive SAM

`SAM_API_URL` defaults to `http://127.0.0.1:8000/sam`. The client expects the
session endpoints defined in `sam_client.py`, not an arbitrary SAM web service.
No server or weights are bundled. Interactive SAM is separate from importing
precomputed AP-SAM/fusion masks. Offline brush and source editing work without it.
Only enable this feature with a service you trust: it sends image content to that service.
