# Release and data-exclusion policy

v0.1.0 is a research preview of the annotation workflow and optional architectures.
It makes no claim of superior segmentation accuracy or reduced annotation time.

Do not publish experimental images, manual labels, visual previews, model weights,
checkpoints, positional-state tensors, private configuration, sample identifiers,
manifests, logs, cached features or experimental archives. Runtime outputs and
synthetic demo files are also ignored. Only reviewed source, docs and dependency
lists belong in this repository.

`.gitignore` is generated from the exact `PUBLIC_FILES.txt` allowlist. Before each
push, run `python tools/audit_public_files.py`; the check scans tracked contents,
checks staged filenames, rejects unexpected files and common sensitive artifacts.
Automated scans supplement human review and cannot guarantee the absence of all
sensitive information. Never force-add a private file to bypass the allowlist.

Use releases/tags for milestones. Keep paper versions intact; publish subsequent
changes under new version numbers. Record a specific commit and the configuration
used for every reported experiment. If a fix changes results, revise associated
claims and results as well. A future paper version can be archived with a DOI.
