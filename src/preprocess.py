import numpy as np

def normalize_landmarks(lm21_xyz: np.ndarray) -> np.ndarray:
    """
    lm21_xyz: (21,3)
    Output: flattened (63,) normalized by:
      - translation: subtract wrist
      - scaling: divide by hand size (wrist -> middle_mcp)
    """
    wrist = lm21_xyz[0]               # landmark 0
    middle_mcp = lm21_xyz[9]         # landmark 9 (middle finger MCP)
    scale = np.linalg.norm(middle_mcp - wrist) + 1e-6

    norm = (lm21_xyz - wrist) / scale
    return norm.reshape(-1).astype(np.float32)

def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    seq: (T, F)
    returns: (target_len, F)
    Uses linear interpolation over time.
    """
    t = seq.shape[0]
    if t == target_len:
        return seq.astype(np.float32)

    x_old = np.linspace(0, 1, t)
    x_new = np.linspace(0, 1, target_len)

    F = seq.shape[1]
    out = np.zeros((target_len, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(x_new, x_old, seq[:, f])
    return out