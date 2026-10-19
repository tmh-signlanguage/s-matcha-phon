"""Monotonic Alignment Search (MAS), pure PyTorch.

Matcha-TTS ships this as a Cython extension (``core.pyx`` + a build step). That
would mean compiling inside the singularity container, so this is a faithful
re-implementation of ``maximum_path_c`` using only torch ops: same banded
forward recurrence, same backtrace, same tie-breaking.

Only needed when ``use_precomputed_durations=False``. The forward recurrence is
inherently sequential over frames, so cost grows with T (vectorised over the
batch and the token axis, but one small kernel launch per frame). With mocap
sentences of ~1000 frames that is noticeably slower than reading the annotated
segment boundaries, which is why precomputed durations are the default here.
"""

import torch

MAX_NEG_VAL = -1e9


def maximum_path(value, mask):
    """Find the most likely monotonic alignment between tokens and frames.

    Args:
        value: (b, t_x, t_y) log-likelihood of each (token, frame) pair.
        mask:  (b, t_x, t_y) 1 where both the token and the frame are real.

    Returns:
        (b, t_x, t_y) hard alignment path, same dtype/device as ``value``.
    """
    device, dtype = value.device, value.dtype
    mask_bool = mask.to(torch.bool)

    # Work in fp32: the recurrence accumulates over the whole sequence and the
    # -1e9 sentinel is not representable in fp16.
    v = (value * mask).to(torch.float32).clone()
    b, t_x, t_y = v.shape

    t_x_max = mask_bool.sum(1)[:, 0].long()  # (b,) real tokens per item
    t_y_max = mask_bool.sum(2)[:, 0].long()  # (b,) real frames per item

    xr = torch.arange(t_x, device=device).unsqueeze(0)  # (1, t_x)
    neg = torch.full((b, t_x), MAX_NEG_VAL, device=device, dtype=torch.float32)

    # ── Forward recurrence ────────────────────────────────────────────────
    # v[x, y] = max(v[x, y-1], v[x-1, y-1]) + v[x, y], restricted to the band
    # of (token, frame) pairs a monotonic path can actually reach. Cells
    # outside the band keep their raw value, exactly as in the Cython version.
    prev_col = None
    for y in range(t_y):
        if y == 0:
            v_stay = neg
            v_step = neg.clone()
            v_step[:, 0] = 0.0  # only (token 0, frame 0) is a valid start
        else:
            v_stay = prev_col  # came from the same token one frame earlier
            v_step = torch.cat([neg[:, :1], prev_col[:, :-1]], dim=1)  # from x-1

        # Reaching token x at frame y==x means every frame advanced a token, so
        # there is no "stay" predecessor.
        v_stay = torch.where(xr == y, neg, v_stay)

        lo = torch.clamp(t_x_max + y - t_y_max, min=0).unsqueeze(1)
        hi = torch.minimum(t_x_max, torch.full_like(t_x_max, y + 1)).unsqueeze(1)
        in_band = (xr >= lo) & (xr < hi)

        col = torch.maximum(v_stay, v_step) + v[:, :, y]
        v[:, :, y] = torch.where(in_band, col, v[:, :, y])
        prev_col = v[:, :, y].clone()

    # ── Backtrace ─────────────────────────────────────────────────────────
    path = torch.zeros_like(v, dtype=torch.int32)
    index = (t_x_max - 1).clamp(min=0)
    bidx = torch.arange(b, device=device)

    for y in range(t_y - 1, -1, -1):
        active = y < t_y_max
        if not bool(active.any()):
            continue
        sel = bidx[active]
        idx = index[sel]
        path[sel, idx, y] = 1

        if y == 0:
            continue
        # Step back a token if we must (idx == y) or if doing so scores higher.
        stay = v[sel, idx, y - 1]
        step = v[sel, (idx - 1).clamp(min=0), y - 1]
        dec = (idx != 0) & ((idx == y) | (stay < step))
        index[sel] = idx - dec.long()

    return path.to(device=device, dtype=dtype)
