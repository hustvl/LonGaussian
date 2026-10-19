# Third-party components

This inventory records source provenance and applicable upstream terms.
Verbatim license texts are in `LICENSES/`; source-file notices remain part of
the distribution.

| Local component | Upstream | Terms / bundled text |
| --- | --- | --- |
| LoGeR backbone and streaming implementation | [Junyi42/LoGeR](https://github.com/Junyi42/LoGeR) | No express source or checkpoint license was identified in the audited upstream snapshot. |
| Pi3/Pi3X architecture | [yyfz/Pi3](https://github.com/yyfz/Pi3) | BSD-3-Clause code; weights have separate terms. `LICENSES/Pi3-BSD-3-Clause.txt` |
| DINOv2 and DINO-style layers | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0. `LICENSES/DINOv2-Apache-2.0.txt` |
| AllTracker inference subset | [aharley/AllTracker](https://github.com/aharley/AllTracker), commit `e7553135e7b361590dbccd10e2b274b024f41cd6` | MIT. `LICENSES/AllTracker-MIT.txt` |
| `loger/models/layers/conv_head.py` | [microsoft/MoGe](https://github.com/microsoft/MoGe) | MIT, with Apache DINO portions upstream. `LICENSES/MoGe-MIT.txt` |
| `loger/models/layers/camera_head.py` | [nianticlabs/Marepo](https://github.com/nianticlabs/marepo), commit `9a45e2bb07e5bb8cb997620088d352b439b13e0e` | Custom research and redistribution terms. `LICENSES/Marepo-research-license.txt` |
| `loger/models/layers/pos_embed.py` | [naver/CroCo](https://github.com/naver/croco) | CC BY-NC-SA 4.0: attribution, non-commercial use and ShareAlike. `LICENSES/CroCo-CC-BY-NC-SA-4.0.txt` |
| `loger/utils/geometry.py` | [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR), commit `94e98b695be18acb43d5d3250f52226a8e36f839` | Apache-2.0. `LICENSES/LoFTR-Apache-2.0.txt` |
| `loger/utils/rotation.py` | [facebookresearch/PyTorch3D](https://github.com/facebookresearch/pytorch3d) | BSD-3-Clause. `LICENSES/PyTorch3D-BSD.txt` |
| `loger/models/ttt.py` | LaCT and [Moonlight/Muon](https://github.com/MoonshotAI/Moonlight) | MIT. `LICENSES/LaCT-MIT.txt`, `LICENSES/Moonlight-MIT.txt` |
| `loger/models/layers/prope.py` | PRoPE authors | MIT notice embedded in the source |

Dependencies in `requirements*.txt` retain their own licenses. Dataset and
model downloads are separate artifacts governed by their publishers' terms;
no datasets or checkpoints are bundled. This inventory does not grant additional
rights over upstream components.
