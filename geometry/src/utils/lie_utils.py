import torch

_CACHE = {}


def get_sign(device, dtype):
    key = ("sign", device, dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.tensor([-1, -1, -1, 1], device=device, dtype=dtype)
    return _CACHE[key]


def get_eye4(device, dtype):
    key = ("eye4", device, dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.eye(4, device=device, dtype=dtype)
    return _CACHE[key]
