# Evaluation indices

These lists define 180 RE10K evaluation scenes and two 70-scene DL3DV shards. Entries preserve the evaluation ordering and contain only relative paths.

For dense C=64/96/128 protocols, pack the first C+floor(C/7) frames, hold out indices 7,15,..., and use the remaining indices as context. The scene list alone does not replace RE10K timestamp and image preparation provenance.
