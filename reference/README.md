# Reference retraining records

These path-free records accompany the Stage-I → Stage-IIa → Stage-IIb reference
retraining run described in `WEIGHTS.md` and `EVALUATION.md`.

- `training_selection.json`: ordered scene identities, frame counts and source
  metadata hashes for the four training pools. It omits local paths and video
  URLs; each pool also carries a hash of its ordered ID list.
- `checkpoint_lineage.json`: three stage SHA256 values, parent hashes, update
  counts and hashes of input manifests/external weights. The original manifest
  hashes refer to the run's full metadata files; the path-free selection export
  is a distinct representation. The Stage-I parent hash identifies the earlier
  Stage-I checkpoint from which this reference run continued; the released
  training driver also supports a fresh Stage-I run from the LoGeR base model.
- `evaluations/{dl3dv,re10k}_ctx{64,96,128}.json`: fixed scene order, per-target
  PSNR/SSIM/LPIPS and Gaussian counts. `summary.json` recomputes the six
  view-weighted aggregate results. Source metric file hashes identify the
  original run records without exposing machine paths.

Training images, source videos and model weights are not part of this directory.
The records can be regenerated with `scripts/export_reference_artifacts.py`
using the four prepared training manifests, three stage reports and six
evaluation result sets (a setting may be supplied as disjoint shard files).
