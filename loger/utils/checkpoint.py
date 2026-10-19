"""Fail-closed loading of the trained Gaussian modules."""
import hashlib
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gaussian_weights(model, path, require_merger=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing model state dictionary")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    expected = model.state_dict()
    prefixes = ["gaussian_decoder.", "gaussian_head."]
    if require_merger:
        prefixes.append("neural_track_merger.")
    for prefix in prefixes:
        keys = [key for key in expected if key.startswith(prefix)]
        if not keys:
            raise ValueError(f"Model has no required module: {prefix}")
        missing = [key for key in keys if key not in state]
        if missing:
            raise ValueError(f"Incomplete {prefix}: {len(missing)} missing keys; first={missing[0]}")
    unexpected = set(state) - set(expected)
    invalid = [key for key in state if key in expected and (
        not isinstance(state[key], torch.Tensor) or state[key].shape != expected[key].shape
    )]
    if unexpected or invalid:
        raise ValueError(f"Incompatible checkpoint: unexpected={sorted(unexpected)[:5]}, shape/type={invalid[:5]}")
    model.load_state_dict(state, strict=False)
    digest = sha256_file(path)
    print(f"Loaded {len(state)} tensors; checkpoint SHA256={digest}", flush=True)
    return digest
