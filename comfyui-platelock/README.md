# comfyui-platelock

**Add the VFX. Keep the shot.** ComfyUI nodes that composite the output of any video-to-video model
(LTX-2.3 EditAnything, IC-LoRAs, Wan, …) back onto your original footage, so that every pixel the effect
doesn't need stays **bit-identical** to the source.

Pure PyTorch, no extra dependencies, runs on CUDA / MPS / CPU.

| Node | Purpose |
|---|---|
| PlateLock · Prep Plate | 8n+1 frame trim + /32 model-res proxy |
| PlateLock · Difference Matte | tone-aligned, noise-adaptive change detection with 3-D hysteresis |
| PlateLock · Temporal Stabilize | flicker-free, ghost-free matte |
| PlateLock · Refine Matte | open / close / grow / feather |
| PlateLock · Composite onto Plate | upscale + re-grain + bit-exact composite onto the full-res plate |
| PlateLock · Extract VFX Element | alpha or emissive element for Nuke / AE / Resolve |
| PlateLock · Fidelity Report | untouched %, bit-identical %, PSNR outside the effect (raw vs PlateLock) |
| PlateLock · X-Ray Overlay | visualise where the model touched the shot |
| PlateLock · Before/After Share | animated wipe, 9:16 / 1:1 / 4:5 reframing |
| PlateLock · Gamma | sign-preserving gamma (EditAnything template helper) |
| PlateLock 🔒 Lock Plate | all-in-one |

Install: clone into `ComfyUI/custom_nodes/`. Example workflows: see the
[PlateLock repo](https://github.com/TusharTechs/platelock) `workflows/`.

Tests: `python -m pytest tests` (synthetic shots with known ground truth).
