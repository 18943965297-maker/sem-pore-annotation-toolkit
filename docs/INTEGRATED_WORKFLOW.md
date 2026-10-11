# Registration and two-source correction

The integrated entry point reuses the registration pipeline from SEM Registration
Studio and the candidate editor from SEM Pore Annotation Toolkit. Both remain
separate repositories; the toolkit can run independently with its own dependencies.

## Start

```powershell
python -m pip install -r requirements.txt
python sem_workbench.py
```

On Windows, `start_workbench.bat` prefers the local `.venv` if it exists.
The existing `fusion_editor.py --workspace ...` entry point still works.

1. Select Before and After originals from the same physical field. Check their
   magnification and SEM pixel-size metadata. Automatic footer removal is a
   heuristic for portrait images; turn it off for already-cropped image content.
2. Optionally select binary Aedge and AP-SAM/fusion candidates for each side.
   Each candidate must use its own original image's coordinates and dimensions.
   Registered/cropped labels from a different task must not be entered here.
3. Run registration. Review the raw overlay, aligned overlay and checkerboard.
   Green is Before intensity; magenta is After intensity. Residual fringes in
   stable structures can indicate remaining geometric error. Genuine structural
   change can also affect correlation.
4. Confirm the image pair, scale and overlap. Choose a **new**, nonexistent
   workspace directory, then create it and open the fusion editor.
5. Inspect read-only candidate sources, select an ROI, add or replace a local
   candidate, and correct with a brush or connected-region deletion. Save and
   reload the result before treating it as a completed annotation.

Changing originals, scales, candidates or registration settings invalidates the
previous preview and review confirmation. Changing the output location does not.
The editor also exposes a new-registration button and the current registration
report, making the geometry step visible during correction.

## What happens to image and label coordinates

Registration uses metadata, SIFT/ORB matches, RANSAC local affine estimation and
optional ECC refinement. Small capture translation, rotation and isotropic scale
differences can be estimated when sufficient stable texture overlaps.

Candidate labels follow the same image geometry: content crop, recorded sampling
matrix, After-to-Before transform and common-core crop. Label resampling uses
nearest-neighbor interpolation. Image intensities retain their original bit depth
in the analysis outputs. Previews are display images.

Source originals and input candidates are read only. The new workspace contains
aligned images, immutable source snapshots, initial per-side masks, the transform,
registration report and a provenance record. An absent SAM candidate initializes
only that side to an empty mask; it is not a completed negative annotation.
Edited labels are saved separately under `new integrate mask/`.

Large magnification or physical pixel-size differences are rejected when creating
a paired change-annotation task. They require a separate localization or scale
analysis workflow. A repeated nominal magnification alone does not prove equal
physical sampling.

## Synthetic capture-drift demonstration

```powershell
python demo_registration.py --output demo_registration_workspace
python fusion_editor.py --workspace demo_registration_workspace/workspace
```

The fixture introduces 7/-5 pixel translation, 1.2 degree rotation and a 1.015
scale factor into a synthetic textured image. It supplies geometric masks in
the Aedge/SAM slots, not network predictions. It records known-transform landmark
error and unregistered versus registered correlation. This verifies the software
connection and coordinate handling; it is not evidence of real SEM accuracy or
segmentation improvement. The fixture's registration approval is automatic only
because the generating transform is known; real inputs require human review.

## Scope

This version prepares and corrects **precomputed** candidates. It does not train
or invoke AP-SAM/Aedge models and does not include weights or experiment data.
Optional interactive SAM uses a separately installed service. Disagreement between
two candidates helps prioritize review; agreement does not guarantee correctness.
The integrated workflow is an engineering improvement, not a claim of a new
registration algorithm or proven annotation-time reduction.
