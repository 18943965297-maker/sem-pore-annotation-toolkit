"""Generate a disposable geometric demo, not SEM observations or model predictions."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def make_demo(workspace):
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError(f"Refusing to overwrite an existing workspace: {workspace}")
    workspace.mkdir(parents=True)
    rows = []
    yy, xx = np.mgrid[:192, :256]
    for index, radius in enumerate((5, 10, 18), 1):
        pair = f"synthetic_{index:02d}"
        task = workspace / "ready_to_annotate" / pair
        task.mkdir(parents=True)
        for side, shift in (("before", 0), ("after", 4)):
            large = (xx - 90 - shift) ** 2 + (yy - 90) ** 2 < (radius * 2) ** 2
            small = (xx - 170) ** 2 + (yy - 80 - shift) ** 2 < radius ** 2
            distractor = (xx - 170) ** 2 + (yy - 145) ** 2 < 3 ** 2
            gray = np.full(xx.shape, 170, dtype=np.uint8)
            gray[large | small] = 45
            Image.fromarray(gray).save(task / f"{side}.png")
            # Deliberately imperfect geometric candidates, not real network output.
            for source, mask in (("SAM", large), ("Aedge", small | distractor)):
                dest = task / "source_masks" / source
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8)).save(dest / f"{side}_pore_mask.png")
            initial = task / "pore_masks"
            initial.mkdir(exist_ok=True)
            Image.fromarray(large.astype(np.uint8)).save(initial / f"{side}_pore_mask.png")
        Image.fromarray(np.zeros(xx.shape, np.uint8)).save(task / "change_mask.png")
        rel = task.relative_to(workspace).as_posix()
        rows.append(dict(pair_id=pair, before=f"{rel}/before.png", after=f"{rel}/after.png",
                         mask=f"{rel}/change_mask.png", annotation_dir=rel, status="ok",
                         dataset_version="synthetic-demo", label_prefix=""))
    with (workspace / "annotation_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("demo_workspace"))
    args = parser.parse_args()
    print(make_demo(args.workspace))
