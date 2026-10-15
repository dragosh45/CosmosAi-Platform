# Small Model Artifacts

Git ignores model outputs by default. Only the small, reviewed files explicitly
allowlisted in `.gitignore` are versioned here so another machine can reproduce
the documented experiments without copying a local run directory.

| Artifact | Purpose | Release status |
| --- | --- | --- |
| `tiny_galaxy_cnn_baseline.pt` | Early tiny-model checkpoint used while building the training/checkpoint path; SHA-256 `b8f4b8b864e99ab75b23327399d3991ae8ee0fb19dc31e2fe7dbc83ee99f7fd3`. | Historical development artifact. |
| `gz2-validation-30epochs-64px/` | SGD validation candidate plus evaluation report and model card. | Rejected; validation remained 25%, test was not used. |
| `gz2-learning-check-adam/` | Adam diagnostic candidate plus evaluation report and model card. | Rejected; 4/12 validation is insufficient evidence, test was not used. |

The deployed demo checkpoint is
[`../data/demo/gz2_pilot_80.pt`](../data/demo/gz2_pilot_80.pt), with its measured
limitations documented in [`../data/demo/README.md`](../data/demo/README.md).

## Repository Policy

- Keep raw datasets and ordinary training outputs outside Git.
- Do not unignore an entire future run directory.
- Add a checkpoint only when it is small, trusted, documented and needed to run
  or reproduce a reviewed result.
- Include a model card, evaluation evidence and SHA-256 checksum for promoted
  candidates.
- Use artifact storage or Git LFS for a future model that is too large for normal
  Git; do not push it to the repository by accident.

PyTorch checkpoints are binary serialized artifacts. Load only checkpoints from
trusted project history.
