import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tkinter as tk

import cv2
import numpy as np
from PIL import Image

from demo_registration import make_capture_demo
from registration_workflow import align_candidate, export_workspace, prepare_pair
from fusion_editor import FusionEditor
from fusion_storage import load_integrated
from sem_workbench import RegistrationWindow


class RegistrationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root_dir,cls.prepared=make_capture_demo(Path(cls.temp.name)/'demo')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_known_capture_drift_and_candidate_alignment(self):
        report=json.loads((self.root_dir/'demo_report.json').read_text())
        self.assertLess(report['known_transform_landmark_max_error_px'],.7)
        self.assertGreater(report['ncc_after'],.90)
        self.assertGreater(report['ncc_after'],report['ncc_before']+.15)
        b=self.prepared.masks['SAM']['before'];a=self.prepared.masks['SAM']['after']
        self.assertGreater((a&b).sum()/(a|b).sum(),.97)
        self.assertEqual(a.shape,self.prepared.result.before_core.shape)
        self.assertTrue(set(np.unique(a)).issubset({False,True}))

    def test_mask_transform_uses_crop_sampling_core_and_nearest(self):
        raw=np.zeros((90,100),np.uint8);raw[20:45,30:70]=255
        sampling=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
        result=SimpleNamespace(before_crop_bounds_xyxy=(10,10,90,80),
            after_crop_bounds_xyxy=(10,10,90,80),before_sampling=sampling,
            transform_after_to_before=sampling.copy(),before_crop=np.zeros((35,40),np.uint8),
            core_bounds_xyxy=(2,3,38,33),overlap_core=np.ones((30,36),bool))
        got=align_candidate(raw,'before',result,raw.shape)
        expected=cv2.warpPerspective((raw[10:80,10:90]>0).astype(np.uint8),sampling,(40,35),flags=cv2.INTER_NEAREST)[3:33,2:38]>0
        np.testing.assert_array_equal(got,expected)
        np.testing.assert_array_equal(align_candidate(raw,'after',result,raw.shape),got)
        result.overlap_core[0]=False
        self.assertFalse(align_candidate(raw,'after',result,raw.shape)[0].any())

    def test_review_and_existing_output_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'workspace'
            with self.assertRaises(ValueError):export_workspace(self.prepared,output)
            self.assertFalse(output.exists())
            export_workspace(self.prepared,output,reviewed=True)
            sentinel=output/'keep.txt';sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):export_workspace(self.prepared,output,reviewed=True)
            self.assertEqual(sentinel.read_text(),'keep')

    def test_reject_wrong_mask_grid_and_non_binary_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong.png';Image.fromarray(np.zeros((32,32),np.uint8)).save(p)
            inputs=self.root_dir/'inputs'
            with self.assertRaises(ValueError):
                prepare_pair(inputs/'before.png',inputs/'after.png',{'Aedge':{'after':p}},500,500,crop_footer=False)
            raw=np.zeros((384,384),np.uint8);raw[0,0]=2
            with self.assertRaises(ValueError):align_candidate(raw,'after',self.prepared.result,raw.shape)

    def test_editor_accepts_registered_sources_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=export_workspace(self.prepared,Path(tmp)/'work',reviewed=True)
            snapshots={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in work.rglob('*.png')}
            try:root=tk.Tk()
            except tk.TclError as exc:self.skipTest(str(exc))
            root.withdraw()
            try:
                with patch('mask_editor_app.MaskEditorApp.check_sam_service'),patch('tkinter.messagebox.showinfo'):
                    app=FusionEditor(root,work);app.load_index()
                    self.assertEqual(len(app.rows),1)
                    np.testing.assert_array_equal(app.mask,self.prepared.masks['SAM']['before'])
                    app.roi=(0,0,app.mask.shape[1],app.mask.shape[0]);app.source_choice.set('Aedge')
                    app.apply_fusion(False);app.save_mask()
                    saved,record=load_integrated(work,app.current_row,app.mask.shape)
                    np.testing.assert_array_equal(saved['before'],self.prepared.masks['Aedge']['before'])
                    app.load_current();np.testing.assert_array_equal(app.mask,saved['before'])
            finally:root.destroy()
            for path,digest in snapshots.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_approval_invalidates_for_geometry_inputs_not_output_location(self):
        try:root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        root.withdraw()
        try:
            window=RegistrationWindow(root,on_ready=lambda _:None)
            window.show_prepared(self.prepared);window.reviewed.set(True);window.update_ready()
            window.paths['destination'].set('new_workspace')
            self.assertIs(window.prepared,self.prepared)
            self.assertTrue(window.reviewed.get())
            window.mags['after'].set('501')
            self.assertIsNone(window.prepared);self.assertFalse(window.reviewed.get())
            self.assertEqual(str(window.finish_button['state']),'disabled')
        finally:root.destroy()

    def test_callback_can_close_registration_window_after_export(self):
        try:root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        root.withdraw()
        child=tk.Toplevel(root)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                seen=[]
                def ready(path):seen.append(path);child.destroy()
                window=RegistrationWindow(child,on_ready=ready)
                window.show_prepared(self.prepared);window.reviewed.set(True)
                window.paths['destination'].set(str(Path(tmp)/'work'))
                window.finish()
                self.assertEqual(len(seen),1);self.assertTrue((seen[0]/'annotation_index.csv').is_file())
            finally:root.destroy()


if __name__=='__main__':unittest.main()
