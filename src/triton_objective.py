"""Triton-accelerated batch objective for Q5.

The kernel evaluates a whole population of 40-D strategies on the GPU in one
launch: for each (candidate, missile) it computes the coarse shielding time
on a fixed 0.1 s time grid, using the same segment-distance model as the CPU
code.  The GPU metric is used only inside differential evolution; final
results are always re-evaluated with the exact CPU brentq refinement.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .config import (CLOUD_LIFETIME, CLOUD_RADIUS, G, MISSILES, MISSILE_SPEED,
                     SAMPLE_DT, SINK_SPEED, TARGET_CENTER, UAVS)


N_TIME = 700  # 0..70 s at dt=0.1

_UAV_ARR = np.asarray([UAVS[k] for k in ("FY1", "FY2", "FY3", "FY4", "FY5")],
                      dtype=np.float32)
_MIS_ARR = np.zeros((3, 7), dtype=np.float32)
for i, name in enumerate(("M1", "M2", "M3")):
    p0 = MISSILES[name].astype(np.float64)
    dist = float(np.linalg.norm(p0))
    u = -p0 / dist
    _MIS_ARR[i, 0:3] = p0.astype(np.float32)
    _MIS_ARR[i, 3:6] = u.astype(np.float32)
    _MIS_ARR[i, 6] = dist / MISSILE_SPEED

_UAV_T = torch.from_numpy(np.ascontiguousarray(_UAV_ARR)).cuda()
_MIS_T = torch.from_numpy(np.ascontiguousarray(_MIS_ARR)).cuda()
_TGT_T = torch.from_numpy(np.ascontiguousarray(TARGET_CENTER.astype(np.float32))).cuda()


@triton.jit
def _q5_durations_kernel(x_ptr, out_ptr, uav_ptr, mis_ptr, tgt_ptr,
                         N_TIME, R, DT, G_CONST, LIFETIME, SPEED, SINK,
                         BLOCK_T: tl.constexpr):
    b = tl.program_id(0)
    m = tl.program_id(1)
    base = b * 40

    mp = mis_ptr + m * 7
    p0x = tl.load(mp + 0)
    p0y = tl.load(mp + 1)
    p0z = tl.load(mp + 2)
    ux0 = tl.load(mp + 3)
    uy0 = tl.load(mp + 4)
    uz0 = tl.load(mp + 5)
    t_imp = tl.load(mp + 6)
    tx = tl.load(tgt_ptr + 0)
    ty = tl.load(tgt_ptr + 1)
    tz = tl.load(tgt_ptr + 2)

    total = 0.0
    for tb in tl.range(0, N_TIME, BLOCK_T):
        tt = tb.to(tl.float32) * DT + tl.arange(0, BLOCK_T).to(tl.float32) * DT
        valid_t = tt < N_TIME * DT
        active = tl.zeros([BLOCK_T], dtype=tl.int32)

        for k in tl.static_range(5):
            ub = base + k * 8
            th = tl.load(x_ptr + ub + 0)
            v = tl.load(x_ptr + ub + 1)
            uu0 = tl.load(x_ptr + ub + 2)
            uu1 = tl.load(x_ptr + ub + 3)
            uu2 = tl.load(x_ptr + ub + 4)
            t0 = tl.load(x_ptr + ub + 5)
            t1 = tl.load(x_ptr + ub + 6)
            t2 = tl.load(x_ptr + ub + 7)

            mn = tl.minimum(tl.minimum(uu0, uu1), uu2)
            mx = tl.maximum(tl.maximum(uu0, uu1), uu2)
            mid = uu0 + uu1 + uu2 - mn - mx
            tau_mn = tl.where(uu0 == mn, t0, tl.where(uu1 == mn, t1, t2))
            tau_mx = tl.where(uu0 == mx, t0, tl.where(uu1 == mx, t1, t2))
            tau_md = t0 + t1 + t2 - tau_mn - tau_mx

            ux = tl.load(uav_ptr + k * 3 + 0)
            uy = tl.load(uav_ptr + k * 3 + 1)
            uz = tl.load(uav_ptr + k * 3 + 2)
            ch = tl.cos(th)
            sh = tl.sin(th)

            for s in tl.static_range(3):
                if s == 0:
                    td = mn
                    tau = tau_mn
                elif s == 1:
                    td = mid + 1.0
                    tau = tau_md
                else:
                    td = mx + 2.0
                    tau = tau_mx
                te = td + tau

                ex = ux + v * (td + tau) * ch
                ey = uy + v * (td + tau) * sh
                ez = uz - 0.5 * G_CONST * tau * tau

                tvalid = (tt >= te) & (tt <= te + LIFETIME) & (tt <= t_imp) & valid_t
                cx = ex
                cy = ey
                cz = ez - SINK * (tt - te)

                px = p0x + SPEED * ux0 * tt
                py = p0y + SPEED * uy0 * tt
                pz = p0z + SPEED * uz0 * tt

                bax = tx - px
                bay = ty - py
                baz = tz - pz
                denom = bax * bax + bay * bay + baz * baz
                lam = ((cx - px) * bax + (cy - py) * bay + (cz - pz) * baz) / tl.maximum(denom, 1e-12)
                lam = tl.minimum(tl.maximum(lam, 0.0), 1.0)
                qx = px + lam * bax
                qy = py + lam * bay
                qz = pz + lam * baz
                ddx = cx - qx
                ddy = cy - qy
                ddz = cz - qz
                d = tl.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                hit = tl.where(tvalid & (d <= R), 1, 0)
                active = tl.maximum(active, hit)

        total += tl.sum(tl.where(active > 0, 1.0, 0.0) * DT)
    tl.store(out_ptr + b * 3 + m, total)


def q5_batch_durations_triton(x: np.ndarray) -> np.ndarray:
    """Return per-missile coarse shielding durations (B, 3) for a strategy batch."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    b = x.shape[0]
    xt = torch.from_numpy(x).cuda()
    out = torch.empty((b, 3), dtype=torch.float32, device="cuda")
    _q5_durations_kernel[(b, 3)](
        xt, out, _UAV_T, _MIS_T, _TGT_T,
        N_TIME, float(CLOUD_RADIUS), float(SAMPLE_DT), float(G),
        float(CLOUD_LIFETIME), float(MISSILE_SPEED), float(SINK_SPEED),
        BLOCK_T=128,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def q5_total_triton(x: np.ndarray) -> float:
    return float(q5_batch_durations_triton(np.asarray(x, dtype=np.float64)[None, :])[0].sum())
