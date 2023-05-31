import numpy as np

OHEncoding = tuple[int]

def encoder_dict(element_set: set) -> dict[str, OHEncoding]:
    l = len(element_set)
    d: dict[str, OHEncoding] = {}
    for i, e in enumerate(list(element_set)):
        v = [int(x) for x in np.zeros(l)]
        v[i] = 1
        d.setdefault(e, tuple(v))
    return d

def encode_phoneme(phoneme: str, encoder_dict: dict[str, OHEncoding]) -> OHEncoding:
    return encoder_dict[phoneme]
