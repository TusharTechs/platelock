"""ComfyUI node wrappers for PlateLock."""

from __future__ import annotations

import json

import torch

from . import core

CATEGORY = "PlateLock"


def _device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:  # outside ComfyUI
        return torch.device("cpu")


def _on(dev, *xs):
    return [None if x is None else x.to(dev) for x in xs]


def _save_report(js: str, prefix: str = "platelock/report"):
    """Write the report next to the videos so API clients (comfy-api-proxy,
    Comfy Cloud) receive it as a file output. Returns ui entries."""
    try:
        import os
        import folder_paths
        full, name, counter, sub, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory())
        fname = f"{name}_{counter:05}_.json"
        with open(os.path.join(full, fname), "w") as f:
            f.write(js)
        return [{"filename": fname, "subfolder": sub, "type": "output"}]
    except Exception:
        return []


def _cpu(*xs):
    out = [x.float().cpu() for x in xs]
    return out[0] if len(out) == 1 else out


class PlateLockPrepPlate:
    """Trim footage to an LTX-friendly length (8n+1) and make a proxy at model res."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "max_long_side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
            "max_frames": ("INT", {"default": 121, "min": 9, "max": 481, "step": 8}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("plate_full", "plate_proxy", "width", "height", "frames")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, max_long_side, max_frames):
        full, proxy, w, h, n = core.prep_plate(plate, max_long_side, max_frames)
        return (full, proxy, w, h, n)


class PlateLockDifferenceMatte:
    """Find exactly what the generative model changed, ignoring VAE drift and shimmer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "generated": ("IMAGE",),
            "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "chroma_weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.1}),
            "detail_sigma": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 6.0, "step": 0.1}),
            "hysteresis": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "region_hint": ("MASK",),
            "protect": ("MASK",),
        }}

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("matte", "generated_tone_matched", "change_heatmap")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, generated, sensitivity, chroma_weight, detail_sigma, hysteresis,
            region_hint=None, protect=None):
        dev = _device()
        plate, generated, region_hint, protect = _on(dev, plate, generated, region_hint, protect)
        matte, gen_tm, z = core.difference_matte(
            plate, generated, sensitivity, chroma_weight, detail_sigma, hysteresis,
            region_hint, protect)
        heat = (z / 12.0).clamp(0, 1)
        heat = torch.stack([heat, heat ** 2, heat ** 4], dim=-1)
        return tuple(_cpu(matte, gen_tm, heat))


class PlateLockTemporalStabilize:
    """Temporal median + zero-phase smoothing so the matte never flickers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "matte": ("MASK",),
            "window": ("INT", {"default": 5, "min": 1, "max": 15, "step": 2}),
            "smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, matte, window, smoothing):
        return (core.temporal_stabilize(matte, window, smoothing),)


class PlateLockRefineMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "matte": ("MASK",),
            "open_px": ("INT", {"default": 1, "min": 0, "max": 16}),
            "close_px": ("INT", {"default": 3, "min": 0, "max": 32}),
            "grow_px": ("INT", {"default": 4, "min": 0, "max": 64}),
            "feather_px": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, matte, open_px, close_px, grow_px, feather_px):
        dev = _device()
        m = core.refine_matte(matte.to(dev), open_px, close_px, grow_px, feather_px)
        return (_cpu(m),)


class PlateLockComposite:
    """Put only the effect back onto the full-res original. Untouched pixels stay bit-exact."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "generated": ("IMAGE",),
            "matte": ("MASK",),
            "grain_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("comp", "matte_full")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, generated, matte, grain_match, seed):
        comp, mfull = core.composite(plate, generated, matte, grain_match, seed)
        return (comp, mfull)


class PlateLockExtractElement:
    """Export the effect as a reusable VFX element (alpha-over or emissive/screen)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "generated": ("IMAGE",),
            "matte": ("MASK",),
            "mode": (["alpha", "emissive"],),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("element_rgb", "element_alpha")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, generated, matte, mode):
        return core.extract_element(plate, generated, matte, mode)


class PlateLockFidelityReport:
    """Measure how much of the original footage survived. Shows the numbers in the UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "generated": ("IMAGE",),
            "comp": ("IMAGE",),
            "matte_full": ("MASK",),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, plate, generated, comp, matte_full):
        rep = core.fidelity_report(plate, generated, comp, matte_full)
        js = json.dumps({k: (str(v) if v == float("inf") else v) for k, v in rep.items()})
        return {"ui": {"text": [core.format_report(rep), js], "files": _save_report(js)},
                "result": (js,)}


class PlateLockMatteOverlay:
    """X-ray view: where did PlateLock let the model touch the shot?"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "matte": ("MASK",),
            "opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, matte, opacity):
        return (core.matte_overlay(image, matte, opacity=opacity),)


class PlateLockBeforeAfter:
    """Shareable animated before/after wipe, reframed for Reels/TikTok/Shorts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "before": ("IMAGE",),
            "after": ("IMAGE",),
            "style": (["wipe", "split"],),
            "aspect": (["9:16", "1:1", "4:5", "16:9", "source"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, before, after, style, aspect):
        return (core.before_after(before, after, style, aspect),)


class PlateLockGamma:
    """Sign-preserving gamma (x ** (1/gamma)). Drop-in for the gamma pair the
    LTX EditAnything template wraps around the model (2.0 in, 0.5 out)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, gamma):
        if gamma == 1.0:
            return (image,)
        return ((torch.sign(image) * image.abs().pow(1.0 / gamma)).clamp(0, 1),)


class PlateLockLock:
    """All-in-one: difference matte -> temporal stabilise -> refine -> composite."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate": ("IMAGE",),
            "generated": ("IMAGE",),
            "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "grow_px": ("INT", {"default": 4, "min": 0, "max": 64}),
            "feather_px": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5}),
            "grain_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
        }, "optional": {
            "region_hint": ("MASK",),
            "protect": ("MASK",),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("comp", "matte_full", "generated_tone_matched", "xray", "report_json")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plate, generated, sensitivity, grow_px, feather_px, grain_match,
            region_hint=None, protect=None):
        dev = _device()
        p, g, rh, pr = _on(dev, plate, generated, region_hint, protect)
        matte, gen_tm, _ = core.difference_matte(p, g, sensitivity, region_hint=rh, protect=pr)
        matte = core.temporal_stabilize(matte)
        matte = core.refine_matte(matte, grow_px=grow_px, feather_px=feather_px)
        matte, gen_tm = _cpu(matte, gen_tm)
        comp, mfull = core.composite(plate, gen_tm, matte, grain_match)
        rep = core.fidelity_report(plate, generated, comp, mfull)
        js = json.dumps({k: (str(v) if v == float("inf") else v) for k, v in rep.items()})
        xray = core.matte_overlay(plate, mfull)
        return {"ui": {"text": [core.format_report(rep)], "files": _save_report(js)},
                "result": (comp, mfull, gen_tm, xray, js)}


NODE_CLASS_MAPPINGS = {
    "PlateLockPrepPlate": PlateLockPrepPlate,
    "PlateLockDifferenceMatte": PlateLockDifferenceMatte,
    "PlateLockTemporalStabilize": PlateLockTemporalStabilize,
    "PlateLockRefineMatte": PlateLockRefineMatte,
    "PlateLockComposite": PlateLockComposite,
    "PlateLockExtractElement": PlateLockExtractElement,
    "PlateLockFidelityReport": PlateLockFidelityReport,
    "PlateLockMatteOverlay": PlateLockMatteOverlay,
    "PlateLockBeforeAfter": PlateLockBeforeAfter,
    "PlateLockGamma": PlateLockGamma,
    "PlateLockLock": PlateLockLock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlateLockPrepPlate": "PlateLock · Prep Plate (8n+1, proxy)",
    "PlateLockDifferenceMatte": "PlateLock · Difference Matte",
    "PlateLockTemporalStabilize": "PlateLock · Temporal Stabilize",
    "PlateLockRefineMatte": "PlateLock · Refine Matte",
    "PlateLockComposite": "PlateLock · Composite onto Plate",
    "PlateLockExtractElement": "PlateLock · Extract VFX Element",
    "PlateLockFidelityReport": "PlateLock · Fidelity Report",
    "PlateLockMatteOverlay": "PlateLock · X-Ray Overlay",
    "PlateLockBeforeAfter": "PlateLock · Before/After Share",
    "PlateLockGamma": "PlateLock · Gamma",
    "PlateLockLock": "PlateLock 🔒 Lock Plate (all-in-one)",
}
