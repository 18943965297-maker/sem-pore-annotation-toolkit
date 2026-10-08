# Model components and provenance

`models/aedge.py` contains the Aedge network used by the earlier fusion workflow:
an attention U-Net-style encoder/decoder with edge guidance, pore templates and
residual local refinement. It is not the later Aedgethree_v2 experiment.

`models/fusion.py` contains the original 7-channel gated correction network and
loss. Inputs are RGB image (3), AP-SAM probability (1), Aedge probability (1),
edge probability (1) and absolute AP/Aedge disagreement (1). The API receives
logits and computes probabilities internally. The correction head is zero initialized.

`models/fusion_v2.py` preserves all AP-SAM positive pixels and learns additions
within `Aedge AND NOT AP-SAM`. It includes the balanced candidate loss. This
constraint prevents removal of AP-SAM false positives. The original v2 experiment
used rule-generated pseudo labels; agreement with them is not human-ground-truth
accuracy. This repository publishes no experiment scores or private run metadata.

These are randomly initialized architecture components, not usable pretrained
segmenters. Inference with the historical trained pipeline needs externally
obtained compatible base implementations, weights, preprocessing, thresholds and
AP-SAM decoder positional state. Older upstream positional tensors were not
persisted in checkpoints; merely loading a weight file may not reproduce that
pipeline. Such state and all training artifacts are deliberately excluded.

The editor accepts saved masks and does not depend on these PyTorch modules.
This separation allows manual workflow testing without weights or a GPU.

Examples of related work (not claims of authorship or novelty):

- AP-SAM: https://doi.org/10.1016/j.fuel.2025.136366
- Edge-guided digital rock segmentation: https://doi.org/10.1016/j.cageo.2025.105884
- ESF-SAM: https://doi.org/10.1016/j.acags.2025.100292
- EdgeSAM for edge devices, a separate project: https://arxiv.org/abs/2312.06660

Public architecture names do not establish originality. A paper must distinguish
reused ideas, implementation contributions, and independently validated findings.
