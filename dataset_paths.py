"""Portable named dataset paths shared by the editor and quality checker."""
import os
from pathlib import Path

PATH_FIELDS = ('before','after','mask','diff','registration_metrics','annotation_dir',
               'before_raw','after_raw','source_metadata')


def resolve_row(row, base):
    result = dict(row)
    for key in PATH_FIELDS:
        if result.get(key):
            p = Path(result[key])
            result[key] = str(p.resolve() if p.is_absolute() else (Path(base)/p).resolve())
    return result


def portable_row(row, base):
    result = dict(row)
    for key in PATH_FIELDS:
        if result.get(key):
            try:
                result[key] = Path(os.path.relpath(result[key], base)).as_posix()
            except ValueError:
                pass
    return result
