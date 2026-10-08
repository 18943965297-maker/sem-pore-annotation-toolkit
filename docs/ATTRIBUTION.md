# Attribution and licensing status

This research preview packages the author's existing paired-mask editor and
Aedge/fusion implementation. The fusion architecture is extracted from the
original model module; private checkpoint loaders and experiment scripts are
not included. The editor shares foundations with:
https://github.com/18943965297-maker/sem-registration-studio

External libraries retain their own licenses: Python/Tkinter, NumPy, Pillow,
OpenCV and optional PyTorch. No copies of those packages are vendored.
The optional SAM API protocol comes from the SAM Workbench API ecosystem:
https://github.com/siyuvision/sam-workbench-api
https://github.com/ksugar/samapi

AP-SAM, SAM, SAM 2, and published EdgeSAM are external works. Their implementations
and pretrained parameters are not distributed. Obtain external software from its
authors and follow the corresponding licensing conditions. References are listed
in MODELS.md.

No blanket open-source license grant for this repository is included in v0.1.0.
Public source availability is not an unrestricted permission to redistribute it.
Licensing and upstream provenance must be resolved before claiming this is an
OSI-licensed package or submitting to a journal requiring that status.

Release preparation used OpenAI Codex assistance for code organization,
documentation, synthetic examples and tests. This is not a statement that any
model accuracy or publication claim has been independently validated.
