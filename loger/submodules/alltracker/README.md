# AllTracker inference subset

Source: [aharley/AllTracker](https://github.com/aharley/AllTracker), commit
`e7553135e7b361590dbccd10e2b274b024f41cd6`.

LonGaussian vendors the inference dependency subset: nets/alltracker.py,
nets/blocks.py and utils/misc.py. Training programs, dataset loaders, demos and
visualization assets are omitted. Retained Python files are unchanged from
the vendored snapshot. The upstream MIT license is preserved in LICENSE and
in the top-level LICENSES directory.

Use the LonGaussian weight downloader and loger/tracking/alltracker_runner.py
for model loading and tracking. See the upstream repository for full AllTracker
training and evaluation instructions.

## Citation

Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu,
Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus,
Katerina Fragkiadaki, Leonidas J. Guibas. AllTracker: Efficient Dense Point Tracking
at High Resolution. ICCV 2025.
