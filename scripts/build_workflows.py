"""Generate PlateLock API-format workflows.

platelock_comp_api.json  plate + any generated clip -> locked comp, x-ray,
                         VFX element, share clip, fidelity report
platelock_ltx_api.json   plate + effect prompt -> LTX-2.3 EditAnything
                         (open weights) -> the same PlateLock pipeline

Nodes the app touches are found by _meta.title, so these graphs can be opened,
tweaked and re-exported from ComfyUI freely.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

LTX_CKPT = "ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA = "gemma_3_12B_it_fp8_scaled.safetensors"
EDIT_LORA = "ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
NEGATIVE = "pc game, console game, video game, ugly, 3d render, photo, still, static, slow"


class Graph(dict):
    def __init__(self, start=0):
        super().__init__()
        self.i = start

    def add(self, class_type, title=None, **inputs):
        self.i += 1
        node = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self[str(self.i)] = node
        return str(self.i)


def add_platelock(g: Graph, plate, gen, fps, audio, prefix="platelock"):
    """PlateLock nodes. plate/gen/fps/audio are [node_id, slot] refs."""
    matte = g.add("PlateLockDifferenceMatte", "Difference Matte", plate=plate, generated=gen,
                  sensitivity=0.5, chroma_weight=2.0, detail_sigma=1.5, hysteresis=True)
    stab = g.add("PlateLockTemporalStabilize", "Temporal Stabilize",
                 matte=[matte, 0], window=5, smoothing=0.6)
    ref = g.add("PlateLockRefineMatte", "Refine Matte", matte=[stab, 0], open_px=1,
                close_px=3, grow_px=4, feather_px=4.0)
    comp = g.add("PlateLockComposite", "Composite onto Plate", plate=plate,
                 generated=[matte, 1], matte=[ref, 0], grain_match=1.0, seed=0)
    g.add("PlateLockFidelityReport", "Fidelity Report", plate=plate, generated=gen,
          comp=[comp, 0], matte_full=[comp, 1])
    xray = g.add("PlateLockMatteOverlay", "X-Ray", image=plate, matte=[comp, 1], opacity=0.55)
    elem = g.add("PlateLockExtractElement", "Extract Element", plate=plate,
                 generated=[matte, 1], matte=[ref, 0], mode="emissive")
    share = g.add("PlateLockBeforeAfter", "Before/After Share", before=plate,
                  after=[comp, 0], style="wipe", aspect="9:16")
    for name, ref_, aud in [("comp", [comp, 0], audio), ("raw", gen, audio),
                            ("xray", [xray, 0], None), ("element", [elem, 0], None),
                            ("share", [share, 0], audio)]:
        kw = {"images": ref_, "fps": fps}
        if aud:
            kw["audio"] = aud
        v = g.add("CreateVideo", **kw)
        g.add("SaveVideo", f"Save {name}", video=[v, 0], filename_prefix=f"{prefix}/{name}",
              format="auto", **{"format.codec": "auto"})


def comp_only():
    g = Graph()
    lv = g.add("LoadVideo", "Plate (original footage)", file="plate.mp4")
    pc = g.add("GetVideoComponents", video=[lv, 0])
    gv = g.add("LoadVideo", "Generated (model output)", file="generated.mp4")
    gc = g.add("GetVideoComponents", video=[gv, 0])
    add_platelock(g, [pc, 0], [gc, 0], [pc, 2], [pc, 1])
    return g


def ltx_fx(max_long_side=1024, max_frames=97):
    g = Graph()
    lv = g.add("LoadVideo", "Plate (original footage)", file="plate.mp4")
    pc = g.add("GetVideoComponents", video=[lv, 0])
    prep = g.add("PlateLockPrepPlate", "Prep Plate", plate=[pc, 0],
                 max_long_side=max_long_side, max_frames=max_frames)
    full, proxy, w, h, n = [prep, 0], [prep, 1], [prep, 2], [prep, 3], [prep, 4]
    fps = [pc, 2]

    ckpt = g.add("CheckpointLoaderSimple", "LTX-2.3 (open weights)", ckpt_name=LTX_CKPT)
    te = g.add("LTXAVTextEncoderLoader", text_encoder=GEMMA, ckpt_name=LTX_CKPT, device="default")
    avae = g.add("LTXVAudioVAELoader", ckpt_name=LTX_CKPT)
    model = g.add("LTXICLoRALoaderModelOnly", "EditAnything IC-LoRA", model=[ckpt, 0],
                  lora_name=EDIT_LORA, strength_model=1.0)
    pos = g.add("CLIPTextEncode", "Effect Prompt", text="Add a glowing orb of energy above the open palm.",
                clip=[te, 0])
    neg = g.add("CLIPTextEncode", "Negative Prompt", text=NEGATIVE, clip=[te, 0])
    cond = g.add("LTXVConditioning", positive=[pos, 0], negative=[neg, 0], frame_rate=fps)
    lat = g.add("EmptyLTXVLatentVideo", width=w, height=h, length=n, batch_size=1)
    guide_img = g.add("PlateLockGamma", "Guide Gamma", image=proxy, gamma=2.0)
    guide = g.add("LTXAddVideoICLoRAGuide", "IC-LoRA Guide (your shot)",
                  positive=[cond, 0], negative=[cond, 1], vae=[ckpt, 2], latent=[lat, 0],
                  image=[guide_img, 0], frame_idx=0, strength=1.0, latent_downscale_factor=1.0,
                  crop="disabled", use_tiled_encode=False, tile_size=256, tile_overlap=64)
    alat = g.add("LTXVEmptyLatentAudio", frames_number=n, frame_rate=fps, batch_size=1,
                 audio_vae=[avae, 0])
    av = g.add("LTXVConcatAVLatent", video_latent=[guide, 2], audio_latent=[alat, 0])
    noise = g.add("RandomNoise", "Seed", noise_seed=42)
    sig = g.add("ManualSigmas", sigmas=SIGMAS)
    smp = g.add("KSamplerSelect", sampler_name="euler_ancestral")
    cfg = g.add("CFGGuider", model=[model, 0], positive=[guide, 0], negative=[guide, 1], cfg=1.0)
    out = g.add("SamplerCustomAdvanced", "LTX Sampler", noise=[noise, 0], guider=[cfg, 0],
                sampler=[smp, 0], sigmas=[sig, 0], latent_image=[av, 0])
    sep = g.add("LTXVSeparateAVLatent", av_latent=[out, 0])
    crop = g.add("LTXVCropGuides", positive=[guide, 0], negative=[guide, 1], latent=[sep, 0])
    dec = g.add("VAEDecode", samples=[crop, 2], vae=[ckpt, 2])
    gen = g.add("PlateLockGamma", "Output Gamma", image=[dec, 0], gamma=0.5)
    add_platelock(g, full, [gen, 0], fps, [pc, 1])
    return g


if __name__ == "__main__":
    out = ROOT / "workflows"
    out.mkdir(exist_ok=True)
    for name, g in [("platelock_comp_api.json", comp_only()), ("platelock_ltx_api.json", ltx_fx())]:
        (out / name).write_text(json.dumps(g, indent=2))
        print("wrote", out / name, len(g), "nodes")
