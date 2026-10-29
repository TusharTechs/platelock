"""PlateLock core: plate-preserving compositing of generative video edits.

Everything here is pure torch so it runs anywhere ComfyUI runs (CUDA, MPS, CPU)
and can be imported outside ComfyUI (the PlateLock app, tests).

Conventions (ComfyUI):
    IMAGE  -> float tensor [T, H, W, 3] in 0..1
    MASK   -> float tensor [T, H, W]    in 0..1
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Masks below this value are snapped to exactly zero so untouched pixels stay
# bit-identical to the original plate.
ZERO_SNAP = 1.0 / 512.0


# ---------------------------------------------------------------------------
# small tensor helpers
# ---------------------------------------------------------------------------

def _to_nchw(img: torch.Tensor) -> torch.Tensor:
    return img.permute(0, 3, 1, 2)


def _to_nhwc(img: torch.Tensor) -> torch.Tensor:
    return img.permute(0, 2, 3, 1)


def _gauss_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur on an NCHW tensor (reflect padded)."""
    if sigma <= 0:
        return x
    k = _gauss_kernel(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    # reflect padding needs pad < dim; fall back to replicate for tiny inputs
    mode = "reflect" if min(x.shape[-2:]) > r else "replicate"
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), ky, groups=c)
    return x


def blur_mask(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a [T,H,W] mask spatially."""
    if sigma <= 0:
        return m
    return gaussian_blur(m.unsqueeze(1), sigma).squeeze(1)


def dilate(m: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return m
    return F.max_pool2d(m.unsqueeze(1), 2 * px + 1, stride=1, padding=px).squeeze(1)


def erode(m: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return m
    return -dilate(-m, px)


def smoothstep(lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def resize_frames(img: torch.Tensor, h: int, w: int, mode: str = "bicubic") -> torch.Tensor:
    """Resize [T,H,W,C] frames, chunked to bound memory."""
    if img.shape[1] == h and img.shape[2] == w:
        return img
    out = []
    for chunk in img.split(16):
        x = _to_nchw(chunk)
        if mode == "area":
            x = F.interpolate(x, size=(h, w), mode="area")
        else:
            x = F.interpolate(x, size=(h, w), mode=mode, align_corners=False)
        out.append(_to_nhwc(x).clamp(0, 1))
    return torch.cat(out)


def resize_mask(m: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if m.shape[1] == h and m.shape[2] == w:
        return m
    out = []
    for chunk in m.split(16):
        out.append(F.interpolate(chunk.unsqueeze(1), size=(h, w), mode="bilinear",
                                 align_corners=False).squeeze(1))
    return torch.cat(out).clamp(0, 1)


def match_length(frames: torch.Tensor, n: int) -> torch.Tensor:
    """Nearest-neighbour temporal resample of a [T,...] tensor to n frames."""
    t = frames.shape[0]
    if t == n:
        return frames
    idx = torch.linspace(0, t - 1, n, device=frames.device).round().long()
    return frames[idx]


def broadcast_mask(m: torch.Tensor | None, t: int, h: int, w: int) -> torch.Tensor | None:
    """Accept a single-frame or per-frame mask of any size; return [t,h,w]."""
    if m is None:
        return None
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = resize_mask(m.float(), h, w)
    if m.shape[0] == 1:
        m = m.expand(t, h, w)
    return match_length(m, t)


def opponent(img_nchw: torch.Tensor) -> torch.Tensor:
    """RGB -> luma + two chroma-difference channels (cheap perceptual space)."""
    r, g, b = img_nchw[:, 0:1], img_nchw[:, 1:2], img_nchw[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.cat([y, b - y, r - y], dim=1)


# ---------------------------------------------------------------------------
# 1. plate prep: make footage LTX-friendly
# ---------------------------------------------------------------------------

def prep_plate(plate: torch.Tensor, max_long_side: int = 768, max_frames: int = 121,
               multiple: int = 32, frame_rule: int = 8):
    """Trim frames to 8n+1 and build a model-resolution proxy of the plate.

    Returns (plate_trimmed_full_res, plate_proxy, width, height, frames).
    """
    t, h, w, _ = plate.shape
    n = min(t, max_frames)
    n = ((n - 1) // frame_rule) * frame_rule + 1
    plate = plate[:n]
    scale = min(1.0, max_long_side / max(h, w))
    ph = max(multiple, int(round(h * scale / multiple)) * multiple)
    pw = max(multiple, int(round(w * scale / multiple)) * multiple)
    proxy = resize_frames(plate, ph, pw, mode="area")
    return plate, proxy, pw, ph, n


# ---------------------------------------------------------------------------
# 2. tone alignment: undo the global colour drift of the VAE / model
# ---------------------------------------------------------------------------

@dataclass
class ToneFit:
    gain: torch.Tensor   # [T,3]
    bias: torch.Tensor   # [T,3]


def fit_tone(plate_small: torch.Tensor, gen: torch.Tensor, iters: int = 4,
             temporal_smooth: int = 5) -> ToneFit:
    """Robust per-frame, per-channel affine fit gen ~= gain*plate + bias.

    Fit on pixels that agree (iteratively reweighted), so a large effect does
    not bias the estimate. Gains are smoothed over time to avoid flicker.
    """
    # Fit on a heavily blurred 1/8-res copy: equalises the detail lost in the
    # VAE (which would otherwise read as a contrast change), removes sensor
    # noise and tolerates small texture misalignment.
    p = gaussian_blur(_to_nchw(plate_small), 3.0)
    g = gaussian_blur(_to_nchw(gen), 3.0)
    hs, ws = max(8, p.shape[2] // 8), max(8, p.shape[3] // 8)
    p = F.interpolate(p, size=(hs, ws), mode="area").flatten(2)  # [T,3,N]
    g = F.interpolate(g, size=(hs, ws), mode="area").flatten(2)
    gain = torch.ones(p.shape[0], 3, 1, device=p.device, dtype=p.dtype)
    bias = torch.zeros_like(gain)
    for _ in range(iters):
        resid = (g - (gain * p + bias)).abs().mean(dim=1, keepdim=True)  # [T,1,N]
        scale = resid.median(dim=2, keepdim=True).values * 2.5 + 1e-3
        wgt = torch.exp(-((resid / scale) ** 2))
        sw = wgt.sum(2, keepdim=True) + 1e-6
        mp = (wgt * p).sum(2, keepdim=True) / sw
        mg = (wgt * g).sum(2, keepdim=True) / sw
        cov = (wgt * (p - mp) * (g - mg)).sum(2, keepdim=True)
        var_p = (wgt * (p - mp) ** 2).sum(2, keepdim=True) + 1e-6
        var_g = (wgt * (g - mg) ** 2).sum(2, keepdim=True) + 1e-6
        # Geometric-mean (reduced major axis) regression: unlike plain least
        # squares it is not biased towards 0 by sensor noise in the plate.
        gain = (torch.sign(cov) * torch.sqrt(var_g / var_p)).clamp(0.6, 1.6)
        bias = mg - gain * mp
    gain, bias = gain.squeeze(-1), bias.squeeze(-1)
    if temporal_smooth > 1 and gain.shape[0] > 2:
        k = min(temporal_smooth, gain.shape[0]) | 1
        pad = k // 2

        def sm(x):
            x = x.t().unsqueeze(0)  # [1,3,T]
            x = F.pad(x, (pad, pad), mode="replicate")
            return F.avg_pool1d(x, k, stride=1).squeeze(0).t()

        gain, bias = sm(gain), sm(bias)
    return ToneFit(gain=gain, bias=bias)


def apply_inverse_tone(gen: torch.Tensor, fit: ToneFit) -> torch.Tensor:
    """Map generated frames back into the plate's colour space."""
    g = fit.gain.view(-1, 1, 1, 3)
    b = fit.bias.view(-1, 1, 1, 3)
    return ((gen - b) / g).clamp(0, 1)


# ---------------------------------------------------------------------------
# 3. difference matte: find what the model actually added
# ---------------------------------------------------------------------------

def difference_matte(plate: torch.Tensor, gen: torch.Tensor, sensitivity: float = 0.5,
                     chroma_weight: float = 2.0, detail_sigma: float = 1.5,
                     hysteresis: bool = True, region_hint: torch.Tensor | None = None,
                     protect: torch.Tensor | None = None):
    """Compute a soft matte of the regions the generative edit changed.

    plate: original frames (any resolution, any length)
    gen:   generated frames (model resolution)
    Returns (matte [T,h,w] at gen res, gen_tone_matched [T,h,w,3], z-score heat [T,h,w]).
    """
    t, h, w, _ = gen.shape
    plate = match_length(plate, t)
    plate_small = resize_frames(plate, h, w, mode="area")

    fit = fit_tone(plate_small, gen)
    gen_tm = apply_inverse_tone(gen, fit)

    dists = []
    for ps, gs in zip(plate_small.split(16), gen_tm.split(16)):
        # blur both: the VAE never reproduces sensor noise / fine texture exactly
        a = opponent(gaussian_blur(_to_nchw(ps), detail_sigma))
        b = opponent(gaussian_blur(_to_nchw(gs), detail_sigma))
        d = (a - b) ** 2
        d = torch.sqrt(d[:, 0] + chroma_weight * (d[:, 1] + d[:, 2]) + 1e-8)
        dists.append(d)
    d = torch.cat(dists)  # [T,h,w]

    # Robust per-frame noise floor estimated from the quietest 50% of pixels,
    # so it stays valid even when the effect covers a big chunk of frame.
    flat = d.flatten(1)
    q = torch.quantile(flat[:, :: max(1, flat.shape[1] // 65536)].float(),
                       torch.tensor([0.25, 0.5], device=d.device), dim=1)
    floor = q[1].view(-1, 1, 1)
    spread = ((q[1] - q[0]).view(-1, 1, 1) * 2.0).clamp_min(4e-3)
    z = (d - floor) / spread

    s = float(min(max(sensitivity, 0.0), 1.0))
    lo = 9.0 - 6.5 * s     # weak threshold: 9 (strict) .. 2.5 (loose)
    hi = lo * 2.0          # strong threshold
    matte = smoothstep(lo, hi, z)

    if hysteresis:
        # 3D geodesic reconstruction: keep weak responses only if they connect
        # (in space or time) to a strong seed. Kills isolated VAE shimmer.
        strong = (z > hi).float()
        weak = (z > lo).float()
        r = strong.unsqueeze(0).unsqueeze(0)
        wk = weak.unsqueeze(0).unsqueeze(0)
        for _ in range(24):
            nr = F.max_pool3d(r, 3, stride=1, padding=1) * wk
            if torch.equal(nr, r):
                break
            r = nr
        keep = dilate(r[0, 0], 2)
        matte = matte * keep

    if region_hint is not None:
        hint = broadcast_mask(region_hint, t, h, w)
        matte = matte * blur_mask(dilate(hint, max(2, h // 40)), max(1.0, h / 100))
    if protect is not None:
        prot = broadcast_mask(protect, t, h, w)
        matte = matte * (1 - prot)

    return matte.clamp(0, 1), gen_tm, z.clamp_min(0)


# ---------------------------------------------------------------------------
# 4. temporal stabilisation + morphological refinement
# ---------------------------------------------------------------------------

def temporal_stabilize(m: torch.Tensor, window: int = 5, ema: float = 0.6) -> torch.Tensor:
    """Temporal median followed by a zero-phase (forward+backward) EMA."""
    t = m.shape[0]
    if t < 3:
        return m
    k = max(1, min(window, t) | 1)
    if k > 1:
        pad = k // 2
        x = torch.cat([m[:1].expand(pad, -1, -1), m, m[-1:].expand(pad, -1, -1)])
        x = x.unfold(0, k, 1)  # [T,H,W,k]
        m = x.median(dim=-1).values
    if 0 < ema < 1:
        fwd = m.clone()
        for i in range(1, t):
            fwd[i] = ema * fwd[i - 1] + (1 - ema) * m[i]
        bwd = m.clone()
        for i in range(t - 2, -1, -1):
            bwd[i] = ema * bwd[i + 1] + (1 - ema) * m[i]
        # min(fwd, bwd) only keeps what both directions agree on, so a fast
        # moving element leaves no ghost trail; max with the raw signal means
        # we never lose it either, while one-frame dropouts get filled in.
        m = torch.maximum(torch.minimum(fwd, bwd), m)
    return m.clamp(0, 1)


def refine_matte(m: torch.Tensor, open_px: int = 1, close_px: int = 3, grow_px: int = 4,
                 feather_px: float = 4.0) -> torch.Tensor:
    if open_px > 0:
        m = dilate(erode(m, open_px), open_px)
    if close_px > 0:
        m = erode(dilate(m, close_px), close_px)
    if grow_px > 0:
        m = dilate(m, grow_px)
    if feather_px > 0:
        m = blur_mask(m, feather_px / 2.0)
    return m.clamp(0, 1)


# ---------------------------------------------------------------------------
# 5. compositing back onto the full-resolution plate
# ---------------------------------------------------------------------------

def estimate_grain(plate: torch.Tensor, sample: int = 8) -> torch.Tensor:
    """Per-channel sensor-noise sigma of the plate (robust MAD of high-pass)."""
    frames = plate[:: max(1, plate.shape[0] // sample)]
    x = _to_nchw(frames)
    hp = x - gaussian_blur(x, 1.0)
    flat = hp.permute(1, 0, 2, 3).flatten(1)
    flat = flat[:, :: max(1, flat.shape[1] // 200000)]
    return (flat.abs().median(dim=1).values * 1.4826).clamp(0, 0.1)  # [3]


def composite(plate: torch.Tensor, gen: torch.Tensor, matte: torch.Tensor,
              grain_strength: float = 1.0, seed: int = 0):
    """Composite generated pixels onto the original plate through the matte.

    Pixels where the (upscaled) matte is zero are copied from the plate
    bit-for-bit. Returns (comp [T,H,W,3], matte_full [T,H,W]).
    """
    t, h, w, _ = plate.shape
    gen = match_length(gen, t)
    matte = match_length(matte, t)
    sigma = estimate_grain(plate) if grain_strength > 0 else None
    gen_cpu = torch.Generator(device="cpu").manual_seed(seed)

    comps, mattes = [], []
    for i in range(0, t, 8):
        p = plate[i:i + 8]
        g = resize_frames(gen[i:i + 8], h, w, mode="bicubic")
        m = resize_mask(matte[i:i + 8], h, w)
        m = torch.where(m < ZERO_SNAP, torch.zeros_like(m), m)
        if sigma is not None:
            noise = torch.randn(g.shape, generator=gen_cpu).to(g.device, g.dtype)
            # the generated region is upscaled and soft; re-grain it to match
            g = (g + noise * sigma.view(1, 1, 1, 3) * grain_strength).clamp(0, 1)
        m3 = m.unsqueeze(-1)
        comp = p * (1 - m3) + g * m3
        # guarantee exactness outside the matte (no float round-off)
        comp = torch.where(m3 == 0, p, comp)
        comps.append(comp)
        mattes.append(m)
    return torch.cat(comps), torch.cat(mattes)


# ---------------------------------------------------------------------------
# 6. element extraction for pro pipelines (Nuke / AE / Resolve)
# ---------------------------------------------------------------------------

def extract_element(plate: torch.Tensor, gen: torch.Tensor, matte: torch.Tensor,
                    mode: str = "alpha"):
    """Turn a generative edit into a reusable VFX element.

    mode="alpha":    premultiplied RGB + alpha (comp with "over")
    mode="emissive": what the model *added* in light, over black
                     (comp with "screen"/"add": fire, sparks, glow, lightning)
    Returns (rgb [T,H,W,3], alpha [T,H,W]).
    """
    t, h, w, _ = plate.shape
    gen = match_length(gen, t)
    matte = match_length(matte, t)
    rgbs, alphas = [], []
    for i in range(0, t, 8):
        p = plate[i:i + 8]
        g = resize_frames(gen[i:i + 8], h, w, mode="bicubic")
        m = resize_mask(matte[i:i + 8], h, w)
        m = torch.where(m < ZERO_SNAP, torch.zeros_like(m), m)
        if mode == "emissive":
            add = (g - p).clamp_min(0) * m.unsqueeze(-1)
            rgbs.append(add)
            alphas.append(add.max(dim=-1).values.clamp(0, 1))
        else:
            rgbs.append(g * m.unsqueeze(-1))
            alphas.append(m)
    return torch.cat(rgbs), torch.cat(alphas)


# ---------------------------------------------------------------------------
# 7. measurable proof: plate fidelity report
# ---------------------------------------------------------------------------

def _psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.unsqueeze(-1).expand_as(a)
    n = m.sum()
    if n < 1:
        return float("nan")
    mse = (((a - b) ** 2) * m).sum() / n
    if mse <= 1e-12:
        return float("inf")
    return float(10 * torch.log10(1.0 / mse))


def fidelity_report(plate: torch.Tensor, gen: torch.Tensor, comp: torch.Tensor,
                    matte_full: torch.Tensor) -> dict:
    """Quantify how much of the original footage survived the edit."""
    t, h, w, _ = plate.shape
    gen = match_length(gen, t)
    raw_sum = comp_sum = 0.0
    n_out = 0.0
    identical = 0.0
    total = float(t * h * w)
    for i in range(0, t, 8):
        p = plate[i:i + 8]
        g = resize_frames(gen[i:i + 8], h, w, mode="bicubic")
        c = comp[i:i + 8]
        outside = (matte_full[i:i + 8] == 0).float()
        o3 = outside.unsqueeze(-1)
        raw_sum += float((((g - p) ** 2) * o3).sum())
        comp_sum += float((((c - p) ** 2) * o3).sum())
        n_out += float(outside.sum()) * 3
        identical += float(((c == p).all(dim=-1)).float().sum())

    def psnr(sse):
        if n_out == 0:
            return float("nan")
        mse = sse / n_out
        return float("inf") if mse <= 1e-12 else 10 * math.log10(1.0 / mse)

    return {
        "frames": t,
        "resolution": f"{w}x{h}",
        "untouched_pct": round(100.0 * (n_out / 3) / total, 2),
        "bit_identical_pct": round(100.0 * identical / total, 2),
        "raw_model_psnr_outside_effect_db": round(psnr(raw_sum), 2),
        "platelock_psnr_outside_effect_db": psnr(comp_sum),
    }


def format_report(r: dict) -> str:
    pl = r["platelock_psnr_outside_effect_db"]
    pl_s = "lossless (bit-exact)" if pl == float("inf") else f"{pl:.2f} dB"
    return (
        f"PlateLock fidelity report - {r['frames']} frames @ {r['resolution']}\n"
        f"  untouched area:            {r['untouched_pct']:.2f}% of pixels\n"
        f"  bit-identical to original: {r['bit_identical_pct']:.2f}% of pixels\n"
        f"  raw model, outside effect: {r['raw_model_psnr_outside_effect_db']:.2f} dB PSNR\n"
        f"  PlateLock, outside effect: {pl_s}"
    )


# ---------------------------------------------------------------------------
# 8. visualisation + shareable before/after
# ---------------------------------------------------------------------------

def matte_overlay(img: torch.Tensor, matte: torch.Tensor, color=(1.0, 0.25, 0.55),
                  opacity: float = 0.55) -> torch.Tensor:
    t, h, w, _ = img.shape
    m = resize_mask(match_length(matte, t), h, w).unsqueeze(-1)
    col = torch.tensor(color, device=img.device, dtype=img.dtype).view(1, 1, 1, 3)
    grey = img.mean(dim=-1, keepdim=True).expand_as(img) * 0.6
    edge = (dilate(m.squeeze(-1), 1) - erode(m.squeeze(-1), 1)).clamp(0, 1).unsqueeze(-1)
    out = grey * (1 - m * opacity) + col * m * opacity + img * m * (1 - opacity)
    return (out * (1 - edge) + edge).clamp(0, 1)


def _fit(img: torch.Tensor, h: int, w: int, cover: bool) -> torch.Tensor:
    _, ih, iw, _ = img.shape
    s = max(h / ih, w / iw) if cover else min(h / ih, w / iw)
    nh, nw = max(1, round(ih * s)), max(1, round(iw * s))
    r = resize_frames(img, nh, nw, mode="bilinear")
    out = torch.zeros(img.shape[0], h, w, 3, device=img.device, dtype=img.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    sy, sx = max(0, -y0), max(0, -x0)
    dy, dx = max(0, y0), max(0, x0)
    ch, cw = min(nh - sy, h - dy), min(nw - sx, w - dx)
    out[:, dy:dy + ch, dx:dx + cw] = r[:, sy:sy + ch, sx:sx + cw]
    return out


def reframe(img: torch.Tensor, aspect: str) -> torch.Tensor:
    """Reframe to a social aspect ratio with a blurred fill background."""
    if aspect == "source":
        return img
    _, h, w, _ = img.shape
    num, den = {"9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "16:9": (16, 9)}[aspect]
    out_h = 1920 if num <= den else 1080
    out_w = int(round(out_h * num / den / 2)) * 2
    bg = _fit(img, out_h, out_w, cover=True)
    bg = _to_nhwc(gaussian_blur(_to_nchw(resize_frames(bg, out_h // 8, out_w // 8, "area")), 3))
    bg = resize_frames(bg, out_h, out_w, "bilinear") * 0.45
    fg = _fit(img, out_h, out_w, cover=False)
    fg_mask = _fit(torch.ones_like(img[:1]), out_h, out_w, cover=False)[..., :1]
    return bg * (1 - fg_mask) + fg * fg_mask


def before_after(before: torch.Tensor, after: torch.Tensor, style: str = "wipe",
                 aspect: str = "9:16", line_px: int = 3) -> torch.Tensor:
    """Animated wipe (or static split) between original and PlateLock result."""
    t = max(before.shape[0], after.shape[0])
    before, after = match_length(before, t), match_length(after, t)
    _, h, w, _ = after.shape
    before = resize_frames(before, h, w, "bilinear")
    xs = torch.arange(w, device=after.device).view(1, 1, w)
    frames = []
    for i in range(t):
        if style == "split":
            pos = w / 2
        else:
            # ease from left to right and back over the clip
            u = 0.5 - 0.5 * math.cos(2 * math.pi * i / max(1, t - 1))
            pos = w * (0.08 + 0.84 * u)
        left = (xs < pos).float().unsqueeze(-1).expand(1, h, w, 1)
        f = before[i:i + 1] * left + after[i:i + 1] * (1 - left)
        line = ((xs - pos).abs() < line_px).float().unsqueeze(-1).expand(1, h, w, 1)
        glow = torch.exp(-((xs - pos) / (line_px * 4)) ** 2).unsqueeze(-1).expand(1, h, w, 1)
        f = f * (1 - line) + line + glow * 0.25 * (1 - line)
        frames.append(f.clamp(0, 1))
    return reframe(torch.cat(frames), aspect)
