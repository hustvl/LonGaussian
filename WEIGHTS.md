# Weights

Download the external models with:

```bash
python scripts/download_assets.py weights --output /path/to/weights
```

The downloader records fixed Hugging Face revisions and verifies SHA256 before
accepting each artifact.

| Role | Source / file | SHA256 |
| --- | --- | --- |
| Frozen LoGeR backbone | `Junyi42/LoGeR`, `LoGeR_star/latest.pt` | `68a0ae71408eb5605b8409296e7d5bf43c87853595766195257af97c5e60373c` |
| Frozen AllTracker | `aharley/alltracker`, `alltracker.pth` | `ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d` |
| Sky segmentation | `JianyuanWang/skyseg`, `skyseg.onnx` | `ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39` |

## Training outputs

The training driver produces the following canonical chain. Hashes identify the
reference retraining run, not a requirement that independent retraining produce
bit-identical checkpoints.

| Stage | File | Reference SHA256 |
| --- | --- | --- |
| I | `stage1.pt` | `5eb98c87ddab96d0c2cd5967e5b8207541949eaf6ebe6e3b82e93ffd81a51950` |
| IIa | `stage2a.pt` | `efb2deba189bc5b00a1ecabb305a52c37d5f4b5356af7eaba53f483da5bd488a` |
| IIb | `stage2b.pt` | `d6d7e8b9614e35208f83ee229e6aff066c3295b82e3377ddc6b27362bf989a76` |

Stage IIa initializes from Stage I; Stage IIb initializes from Stage IIa. The
driver records parent hashes, data-manifest hashes, completed optimizer steps
and commands in each stage report. See [EVALUATION.md](EVALUATION.md) for the
reference Stage-IIb metrics.

For offline tracking, set `ALLTRACKER_CHECKPOINT=/path/to/weights/alltracker.pth`
and `ALLTRACKER_SHA256` to the value above. Formal Gaussian checkpoint loading
validates decoder, attribute-head and merger completeness and prints SHA256.
