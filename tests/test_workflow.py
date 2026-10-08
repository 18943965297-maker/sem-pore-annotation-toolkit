import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tkinter as tk

from demo_synthetic import make_demo
from fusion_editor import FusionEditor, connected_region, merge_region
from fusion_storage import load_integrated


class WorkflowTests(unittest.TestCase):
    def test_connected_component_and_roi(self):
        source = np.eye(6, dtype=bool)
        source[0, 5] = True
        self.assertEqual(int(connected_region(source, 0, 0).sum()), 6)
        self.assertFalse(connected_region(source, 1, 0).any())
        target = np.ones_like(source)
        got = merge_region(target, source, (1, 1, 4, 4), replace=True)
        self.assertTrue(got[0].all())
        np.testing.assert_array_equal(got[1:4, 1:4], source[1:4, 1:4])

    def test_edit_save_reload_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = make_demo(Path(tmp) / "workspace")
            sources = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in work.rglob("*.png")}
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"Graphical session unavailable: {exc}")
            root.withdraw()
            errors = []
            try:
                with patch("tkinter.messagebox.showinfo"), patch(
                    "tkinter.messagebox.showerror", side_effect=lambda *a: errors.append(a)
                ):
                    app = FusionEditor(root, work)
                    app.load_index()
                    self.assertEqual(len(app.rows), 3)
                    before = app.annotations.layers["before"]["mask"].copy()
                    after = app.annotations.layers["after"]["mask"].copy()
                    source = app.source_masks["Aedge"]["before"]
                    app.roi = (155, 60, 190, 100)
                    app.source_choice.set("Aedge")
                    app.apply_fusion(False)
                    expected = merge_region(before, source, app.roi)
                    np.testing.assert_array_equal(app.mask, expected)
                    self.assertGreater(int(app.mask.sum()), int(before.sum()))
                    app.undo()
                    np.testing.assert_array_equal(app.mask, before)
                    app.redo()
                    np.testing.assert_array_equal(app.mask, expected)
                    app.save_mask()
                    loaded, record = load_integrated(work, app.current_row, before.shape)
                    np.testing.assert_array_equal(loaded["before"], expected)
                    np.testing.assert_array_equal(loaded["after"], after)
                    app.load_current()
                    np.testing.assert_array_equal(app.mask, expected)
                    mask_path = work / "new integrate mask" / record["before_mask"]
                    mask_path.write_bytes(mask_path.read_bytes() + b"modified")
                    with self.assertRaises(ValueError):
                        load_integrated(work, app.current_row, before.shape)
                    self.assertFalse(errors, errors)
            finally:
                root.destroy()
            for path, digest in sources.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                make_demo(work)


if __name__ == "__main__":
    unittest.main()
