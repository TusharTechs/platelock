import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from platelock import core  # noqa: E402


def synthetic_shot(t=25, h=270, w=480, seed=0):
    """Plate: textured panning scene with sensor noise.
    Gen:   the 'model output' - soft, tone shifted, shimmering, plus a glowing orb."""
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0, 1, h).view(h, 1)
    xs = torch.linspace(0, 1, w).view(1, w)
    tex = torch.rand(1, 3, h // 6, w // 6, generator=g)
    tex = torch.nn.functional.interpolate(tex, size=(h, w), mode="bicubic", align_corners=False)
    tex = tex[0].permute(1, 2, 0).clamp(0, 1)
    plates, gens, truth = [], [], []
    for i in range(t):
        shift = int(i * 2)
        base = torch.roll(tex, shifts=shift, dims=1) * 0.6 + 0.2
        base = base * torch.stack([0.8 + 0.2 * ys.expand(h, w), 0.9 * torch.ones(h, w),
                                   1.0 - 0.3 * xs.expand(h, w)], -1)
        plate = (base + torch.randn(h, w, 3, generator=g) * 0.01).clamp(0, 1)
        # model output: VAE-ish blur + global tone drift + low-amp shimmer
        soft = core.gaussian_blur(plate.permute(2, 0, 1)[None], 1.2)[0].permute(1, 2, 0)
        gen = soft * torch.tensor([1.05, 0.97, 1.02]) + 0.015
        gen = gen + torch.randn(h, w, 3, generator=g) * 0.006
        cx, cy = 0.3 + 0.4 * i / t, 0.5 + 0.1 * math.sin(i / 3)
        r = torch.sqrt(((xs - cx) * w / h) ** 2 + (ys - cy) ** 2)
        orb = torch.exp(-(r / 0.08) ** 2)
        gen = gen * (1 - orb[..., None]) + orb[..., None] * torch.tensor([1.0, 0.6, 0.15])
        plates.append(plate)
        gens.append(gen.clamp(0, 1))
        truth.append((orb > 0.15).float())
    return torch.stack(plates), torch.stack(gens), torch.stack(truth)


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def test_prep_plate_constraints():
    plate = torch.rand(50, 1080, 1920, 3)
    full, proxy, pw, ph, n = core.prep_plate(plate, max_long_side=768)
    assert n == 49 and (n - 1) % 8 == 0
    assert pw % 32 == 0 and ph % 32 == 0 and max(pw, ph) <= 800
    assert proxy.shape == (49, ph, pw, 3) and full.shape[0] == 49


def test_tone_fit_recovers_drift():
    plate, gen, _ = synthetic_shot()
    fit = core.fit_tone(plate, gen)
    assert torch.allclose(fit.gain.mean(0), torch.tensor([1.05, 0.97, 1.02]), atol=0.04)


def test_matte_finds_effect_and_ignores_shimmer():
    plate, gen, truth = synthetic_shot()
    matte, gen_tm, _ = core.difference_matte(plate, gen, sensitivity=0.5)
    matte = core.temporal_stabilize(matte)
    matte = core.refine_matte(matte, grow_px=0, feather_px=0)
    score = iou(matte, truth)
    false_pos = float(((matte > 0.5) & (truth < 0.5)).float().mean())
    print(f"IoU={score:.3f} false_pos={false_pos:.4f}")
    assert score > 0.75
    assert false_pos < 0.01


def test_composite_is_bit_exact_outside_matte():
    plate, gen, _ = synthetic_shot()
    small_gen = core.resize_frames(gen, 135, 240, "area")   # model runs at lower res
    matte, gen_tm, _ = core.difference_matte(plate, small_gen)
    matte = core.refine_matte(core.temporal_stabilize(matte))
    comp, mfull = core.composite(plate, gen_tm, matte)
    assert comp.shape == plate.shape
    outside = mfull == 0
    assert outside.float().mean() > 0.6
    assert torch.equal(comp[outside], plate[outside])
    rep = core.fidelity_report(plate, small_gen, comp, mfull)
    print(core.format_report(rep))
    assert rep["platelock_psnr_outside_effect_db"] == float("inf")
    assert rep["raw_model_psnr_outside_effect_db"] < 40


def test_elements_and_share():
    plate, gen, _ = synthetic_shot(t=9)
    matte, gen_tm, _ = core.difference_matte(plate, gen)
    rgb, a = core.extract_element(plate, gen_tm, core.refine_matte(matte), mode="emissive")
    assert rgb.shape == plate.shape and a.shape == plate.shape[:3]
    share = core.before_after(plate, gen, aspect="9:16")
    assert share.shape == (9, 1920, 1080, 3)
    ov = core.matte_overlay(plate, matte)
    assert ov.shape == plate.shape
