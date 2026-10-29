# PlateLock: LTX-2.3 v2v research findings

Researched 2026-10-29. Every URL and size below was checked live (HF API or HTTP HEAD) unless marked **UNVERIFIED**.

---

## TL;DR: decisions

| Decision | Choice |
|---|---|
| Workflow | "LTX 2.3 - EditAnything", downloaded as `research/downloads/ (not committed) editanything_1fb07c3b4b99.json`. It's a single-pass IC-LoRA v2v workflow with 8 distilled steps and CFG 1. |
| Base model | `ltx-2.3-22b-distilled-fp8.safetensors` (29.53 GB), replacing the template's dev-fp8 + distill LoRA 384. This is equivalent, smaller, and what the LoRA author recommends trying first. |
| Text encoder | `gemma_3_12B_it_fp8_scaled.safetensors` (13.21 GB), replacing the template's bf16 `gemma_3_12B_it.safetensors` (24.38 GB) |
| Edit LoRA | `ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors` (1.31 GB), from `Alissonerdx/LTX-LoRAs` |
| **Total minimum download** | **44.05 GB** (3 files). The full fallback profile adds about 4.3 GB. |
| Custom node packs | ComfyUI-LTXVideo, VideoHelperSuite, KJNodes, ComfyMath, plus radiance (only for `Float32ColorCorrect`, which can be replaced) |
| Working resolution | **1024x576** for 16:9 footage (exact 16:9, both sides multiples of 32). Frames 8n+1: 73 / 97 / 121. |
| Fallback | Lightricks **IC-LoRA In/Outpainting** (mask-based, preserves outside the mask), then **Union Control** (canny/depth). |
| API | `comfy-api-proxy` (PyPI 0.1.6) on :8189 in front of ComfyUI :8188, driven by `comfy-sdk` 0.5.0 with `COMFY_BASE_URL=http://127.0.0.1:8189`. Video upload and custom nodes both work through it (see §4). |

---

## 1. The EditAnything workflow

### Provenance
- Page: https://comfy.org/workflows/1fb07c3b4b99-1fb07c3b4b99/ (credited to Purz; the canvas note says the LoRA is by **Alisson Pereira / Alissonerdx**). The page doesn't list model files.
- **JSON download (works):** `https://comfy.org/workflows/download/1fb07c3b4b99.json`, saved as `research/downloads/ (not committed) editanything_1fb07c3b4b99.json` (UI format with one subgraph, 135 KB).
- It is **not** in `Comfy-Org/workflow_templates`. Structurally it is a clone of the official template `templates/template_ltx2_3_lora_video_outpainting.json` (same node IDs and the same subgraph UUID `2e37ecc9-…`), with the outpaint LoRA swapped for the EditAnything LoRA. That official template carries the model URLs used below (`research/downloads/ (not committed) wt_template_ltx2_3_lora_video_outpainting.json`).
- The LoRA author's own workflow is `Alissonerdx/LTX-LoRAs/workflows/ltx23_edit_anything_v1.1.json` (saved as `research/downloads/ (not committed) alissonerdx_ltx23_edit_anything_v1.1.json`). It uses UNETLoader + DualCLIPLoader, KJ `LTXVAddGuideMulti`, an upscaler 2nd pass, and pulls in more packs (Easy-Use, GGUF, pysssss).
- Related: `Alissonerdx/EditAnything` (HF) has `edit_anything_v1.1_r256.safetensors` (2.62 GB, LTX-2.3, "standalone LoRA, any loader") and `edit_anything_v2_ltx2.5.safetensors` (**LTX-2.5 only**, trigger `edit_anything:`, trained 704x384x73, CFG 3-5 / 30 steps). Neither is used by this template.

### Node list (`class_type`)
Top level: `VHS_LoadVideo`, `VHS_VideoCombine` x2 (5069 comparison, **5152 edited output**), `MarkdownNote`, plus subgraph `2e37ecc9-770a-4ac2-972c-e953901f49d3` ("LTX 2.3 - EditAnything", node 5151). API export flattens the subgraph.

| class_type | Pack | Repo |
|---|---|---|
| CheckpointLoaderSimple, LoraLoaderModelOnly, LTXAVTextEncoderLoader, LTXVAudioVAELoader, CLIPTextEncode, LTXVConditioning, EmptyLTXVLatentVideo, LTXVEmptyLatentAudio, LTXVConcatAVLatent, LTXVSeparateAVLatent, LTXVCropGuides, LTXVPreprocess, LTXVAudioVAEDecode, CFGGuider, RandomNoise, ManualSigmas, KSamplerSelect, SamplerCustomAdvanced, VAEDecode, VAEDecodeTiled (unused), GetImageSize, ResizeImageMaskNode, LoadImage, PrimitiveInt, PrimitiveBoolean, ComfyMathExpression | ComfyUI core (needs a recent build: ResizeImageMaskNode "scale to multiple" is v0.18.1+, ComfyMathExpression is `comfy_extras/nodes_math.py`) | https://github.com/Comfy-Org/ComfyUI |
| LTXAddVideoICLoRAGuide, LTXICLoRALoaderModelOnly, LTXVImgToVideoConditionOnly | ComfyUI-LTXVideo | https://github.com/Lightricks/ComfyUI-LTXVideo |
| VHS_LoadVideo, VHS_VideoCombine, VHS_VideoInfo | VideoHelperSuite | https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite |
| ImageResizeKJv2, ImagePadKJ, ImageConcanate | KJNodes | https://github.com/kijai/ComfyUI-KJNodes |
| CM_FloatToInt | ComfyMath | https://github.com/evanspearman/ComfyMath |
| Float32ColorCorrect | radiance (FXTD) | https://github.com/FXTD-Studios/radiance |

`Float32ColorCorrect` is used as a gamma pair: 5144 applies gamma=2 to the guide (out = in^(1/2)) and 5143 applies gamma=0.5 to the decode (out = in^2). Nothing else is used from radiance, and its requirements are heavy (diffusers, OpenImageIO, OpenColorIO). **Recommendation:** replace both nodes with a 5-line pow node in `comfyui-platelock`, or just wire through. The pair is meant to cancel out, but its effect on edit quality is **UNVERIFIED**. The setup script installs radiance by default so the template loads as-is (`INSTALL_RADIANCE=0` skips it).

### Models (template as shipped vs. recommended for 24 GB)

| Role | Template filename | Recommended (4090) | HF URL | Size | Folder |
|---|---|---|---|---|---|
| Checkpoint (transformer + video VAE + audio VAE + text connector) | `ltx-2.3-22b-dev-fp8.safetensors` | **`ltx-2.3-22b-distilled-fp8.safetensors`** | https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors | 29.53 GB | `models/checkpoints/` |
| Distill LoRA | `ltx-2.3-22b-distilled-lora-384.safetensors` @1.0 | **drop** (not needed with the distilled ckpt) | https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors | 7.61 GB | `models/loras/` |
| Text encoder (Gemma 3 12B) | `gemma_3_12B_it.safetensors` (bf16) | **`gemma_3_12B_it_fp8_scaled.safetensors`** | https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors | 13.21 GB | `models/text_encoders/` |
| EditAnything IC-LoRA | `ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors` @1.0 | same | https://huggingface.co/Alissonerdx/LTX-LoRAs/resolve/main/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors | 1.31 GB | `models/loras/` |
| VAE / audio VAE | loaded from the checkpoint by `CheckpointLoaderSimple` / `LTXVAudioVAELoader` | same | (inside the ckpt) | n/a | n/a |
| Upscaler | none in this workflow | optional: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors | 1.00 GB | `models/latent_upscale_models/` |

Template-faithful alternatives: dev ckpt `https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors` (29.15 GB) and bf16 Gemma `https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it.safetensors` (24.38 GB).

LoRA safetensors metadata: `reference_downscale_factor = 1`, so the guide is full resolution and the latent_downscale_factor widget stays at 1. The LoRA README says it's experimental: trained on 8k video pairs, "start with a distilled model at CFG 1; if too weak raise CFG; distill-LoRA strength ~1.2 can help". Prompt templates:
- Add: `Add a/an [subject] with [visual attributes], [precise location].`
- Remove: `Remove the [object] [location].`
- Replace: `Replace the [original] [location] with a/an [new] with [attributes].`
- Style: `Convert the video into a [style] style.`

The model can't do multi-verb prompts or "change background". For PlateLock, try: `Add bright blue-white lightning arcs crackling around the person's raised right hand, in the center of the frame.` and `Add a glowing orange circular magic portal opening on the wall directly behind the person.`

### Wiring (inside the subgraph)
```
VHS_LoadVideo.IMAGE ──► ImageResizeKJv2(w=target_w, h=target_h, lanczos, keep_proportion="crop", center, divisible_by=2)
                      ──► Float32ColorCorrect(gamma 2) ──► LTXAddVideoICLoRAGuide.image   (the IC guide = source video)
VHS_LoadVideo.frame_count ─► LTXVEmptyLatentAudio.frames_number ; VHS_VideoInfo.loaded_fps ─► LTXVConditioning.frame_rate (+CM_FloatToInt ─► audio fps)
CheckpointLoaderSimple ─► LoraLoaderModelOnly(distill 384) ─► LTXICLoRALoaderModelOnly(EditAnything) ─► CFGGuider(cfg=1)
LTXAVTextEncoderLoader(gemma, ckpt) ─► CLIPTextEncode(+ prompt / – "pc game, console game, video game, ugly, 3d render, photo, still, static, slow") ─► LTXVConditioning(24 → linked to fps)
EmptyLTXVLatentVideo(w=target_w, h=target_h, length=batch_size of a separately padded/0.5x copy) ─► LTXVImgToVideoConditionOnly(strength 1, bypass=True)
   ─► LTXAddVideoICLoRAGuide(frame_idx 0, strength 1, downscale 1, crop disabled) ─► LTXVConcatAVLatent(+empty audio latent)
SamplerCustomAdvanced(RandomNoise, CFGGuider, KSamplerSelect euler_ancestral, ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
   ─► LTXVSeparateAVLatent ─► LTXVCropGuides ─► VAEDecode ─► Float32ColorCorrect(gamma 0.5) ─► [out 2] ─► VHS_VideoCombine 5152 (+ original audio)
[out 0] = ImageConcanate(edited | 0.5x source) ─► VHS_VideoCombine 5069 ("outpaint_comparison")
```
- **Input video:** top-level `VHS_LoadVideo` (5060): `video`, `frame_load_cap`=77, **`skip_first_frames`=1** (set this to 0 or PlateLock will be off by one frame), `force_rate`=0.
- **Prompt:** `CLIPTextEncode` 5138 (positive) inside the subgraph. `PrimitiveInt` 5097/5098 hold target W/H (default **1080x1920 vertical**; change this).
- **i2v branch:** `LoadImage` 5145 → `LTXVImgToVideoConditionOnly` with `bypass_i2v=True`. It's bypassed, but in API format LoadImage still needs a real file in `input/`. Upload any frame, or rewire it to the first video frame.
- The generated audio is discarded. Output videos mux the original audio.

### Constraints and settings
- Spatial: latent is /32. Use W, H that are multiples of 32 (template default 1080 isn't, so it silently floors). Temporal: /8, so **frames = 8n+1** (77 → 73 output frames). Set `frame_load_cap` to 8n+1.
- ImageResizeKJv2 uses **center crop** to the target aspect, so the edited plate is a cropped, resized version of the source. PlateLock must apply the same crop/resize to the original before compositing, or the source should already match the target aspect.
- Sampling: 8 steps (9 manual sigmas), CFG 1.0, `euler_ancestral`, random seed. Single pass, no upscale stage.
- The LTX VAE round-trip shifts unchanged pixels a little (detail and colour). That's exactly what PlateLock's changed-region composite fixes.
- Suggested sizes for 4090: 1024x576 x 97 frames (~4 s at 24 fps) to start, then 121 frames if memory allows. The IC guide doubles the token count. **VRAM and runtime not measured (UNVERIFIED).**

### Edits to make after loading the template (for PlateLock)
1. Ckpt widgets (`CheckpointLoaderSimple`, `LTXVAudioVAELoader`, `LTXAVTextEncoderLoader.ckpt_name`) → `ltx-2.3-22b-distilled-fp8.safetensors`.
2. Bypass or delete `LoraLoaderModelOnly` (distill 384) and wire Checkpoint MODEL straight into `LTXICLoRALoaderModelOnly`.
3. `LTXAVTextEncoderLoader.text_encoder` → `gemma_3_12B_it_fp8_scaled.safetensors`.
4. Target W/H → 1024/576. `skip_first_frames` → 0. `frame_load_cap` → 97.
5. Optionally replace the `Float32ColorCorrect` pair, and point `LoadImage` at any image.
6. Export with **File → Export (API)**. Subgraph nodes get flattened IDs like `"5151:5138"`. **UNVERIFIED:** check the exported JSON for the exact IDs before `set_input`.

---

## 2. Fitting a 24 GB RTX 4090
- **Use `Lightricks/LTX-2.3-fp8` → `ltx-2.3-22b-distilled-fp8.safetensors`** (29.53 GB on disk; the transformer is about 22-23 GB fp8). It won't fit fully in VRAM with activations. ComfyUI auto-offloads part of it ("loaded partially") into system RAM, so it runs, just slower. **Rent a pod with 64 GB+ system RAM.** The official Comfy IC-LoRA template (`video_ltx2_3_ic_lora.json`) uses this exact checkpoint for consumer GPUs. Ada supports FP8. `--fast fp8_matrix_mult` is optional (faster, possibly slight quality loss).
- **EditAnything + distilled:** yes. The template already runs the distilled sigma schedule at CFG 1 via dev + distill LoRA @1.0, which is functionally the distilled model. The LoRA README explicitly says to start with the distilled model at CFG 1. If the edit is too weak, go back to dev-fp8 + distill LoRA (strength 0.6-1.2) and raise CFG. CFG>1 doubles compute.
- Gemma: `gemma_3_12B_it_fp8_scaled` (13.21 GB) is the safe pick on Ada. `gemma_3_12B_it_fp4_mixed` (9.45 GB, used by official templates) depends on NVFP4 support, which is native only on Blackwell. Whether it falls back on Ada is **UNVERIFIED**, so avoid it.
- If offload is too slow, use GGUF (needs https://github.com/city96/ComfyUI-GGUF and `UnetLoaderGGUF` + separate VAE): `https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled/ltx-2.3-22b-distilled-Q5_K_M.gguf` (16.07 GB; Q4_K_M = 14.33 GB, Q6_K about 17.8 GB, `distilled-1.1/` also available). Needs the VAEs from `Kijai/LTX2.3_comfy` (`vae/LTX23_video_vae_bf16.safetensors` 1.45 GB, `vae/LTX23_audio_vae_bf16.safetensors` 0.36 GB) and `text_encoders/ltx-2.3_text_projection_bf16.safetensors` (2.31 GB) via DualCLIPLoader. This needs rewiring, so treat it as a plan B. **Not tested.**
- Kijai transformer-only fp8 (`Kijai/LTX2.3_comfy/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors`, 23.47 GB) is about the same size. No advantage.

---

## 3. Fallback v2v approaches (LTX-2.3)

**A. IC-LoRA In/Outpainting (Lightricks, best fit for PlateLock if a mask is available)**
- LoRA: `ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` (1.31 GB) from https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting. **Gated (auto-approve):** accept the terms, then download with `HF_TOKEN`. Goes in `models/loras/`.
- Workflow: `ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json` (saved in downloads). Inputs are the source video plus a B/W mask video (white = generate, same frame count and aspect). The prompt describes only the masked region.
- Nodes (LTXVideo): `LTXICLoRALoaderModelOnly`, `LTXVDilateVideoMask`, `LTXVInpaintPreprocess`, `LTXAddVideoICLoRAGuideAdvanced`, `LTXVImgToVideoConditionOnly`, `LTXVLaplacianPyramidBlend`, `LTXVTiledVAEDecode`. Core: `LoadVideo`, `GetVideoComponents`, `ImageToMask`, `ResizeImageMaskNode`, `ComfyMathExpression`, `VAEEncodeTiled`, `VAEEncodeAudio`, `SaveVideo`, plus the same sampler stack. Two stages: 8 sigmas, then `0.725, 0.4219, 0.0` (euler_cfg_pp), CFG 1. Example is 960x544x121 using dev + distill LoRA 1.1 @0.5; swapping in distilled-fp8 should work (**UNVERIFIED**). `GemmaAPITextEncode` nodes in it are mode 4 (bypassed), so no API dependency.
- The mask (for example a box around the hand) has to come from PlateLock, SAM, or the user. Note that this workflow already Laplacian-blends back onto the original.
- Community alternative: `Alissonerdx/LTX-LoRAs/ltx23_inpaint_masked_t2v_rank128_v1_02500steps.safetensors` (1.31 GB, apache-2.0, not gated). The mask is **embedded in the guide video as magenta (255,0,255)** and passed via KJ `LTXVAddGuideMulti`, with an 8-px `BlockifyMask` (KJ). Workflow: `research/downloads/ (not committed) alissonerdx_ltx23_inpaint_v1.json` (also uses SAM2/SAM3, Essentials, lf-nodes, Easy-Use).

**B. IC-LoRA Union Control (canny / depth / pose)**
- LoRA: `https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (0.65 GB, not gated) in `models/loras/`. Reference at **0.5x**: wire `LTXICLoRALoaderModelOnly.latent_downscale_factor` (=2) into `LTXAddVideoICLoRAGuide`.
- Official Comfy template `templates/video_ltx2_3_ic_lora.json` uses core nodes only: `LTXVAddGuide`, `GetICLoRAParameters`, `LoadMoGeModel`/`MoGeInference`/`MoGeRender` for depth, with `moge_2_vitl_normal_fp16.safetensors` (https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors) in `models/geometry_estimation/`. Core `Canny` works for canny with zero extra models.
- Lightricks version: `LTX-2.3_ICLoRA_Union_Control_Distilled.json` needs comfyui_controlnet_aux (`CannyEdgePreprocessor`, `DWPreprocessor`), ComfyUI-Video-Depth-Anything, and Essentials (`SimpleMath+`).
- This regenerates the whole frame following structure plus a full-scene prompt ("a man holding crackling lightning…"). Identity and texture drift more than with EditAnything, but PlateLock's composite contains it.

**C. Raw latent masking (SetLatentNoiseMask etc.)**
- Core `SetLatentNoiseMask` on a `VAEEncode`d LTX latent is used in Alissonerdx's workflows (with `SolidMask`). Core LTX already uses per-token `noise_mask` for guide frames, so spatial masks on 5D latents (mask at latent res W/32 x H/32 x (F-1)/8+1) *may* work without a LoRA, but this is **UNVERIFIED** and likely to give seams. `LTXVAddGuide` (core) / `LTXVAddGuideMulti` (KJ) only add guide frames; they don't mask. Prefer A.

---

## 4. comfy-api-proxy + comfy-sdk

Repo: https://github.com/Comfy-Org/comfy-api-proxy (MIT, PyPI `comfy-api-proxy` 0.1.6, Python ≥3.10, deps aiohttp/blake3/pydantic/PyYAML). It exposes Comfy API v2 (`/api/v2/jobs`, `/api/v2/assets`, SSE events, health) in front of ComfyUI's native `/prompt`, `/history`, `/view`, `/upload/image`, `/ws`.

On the pod:
```bash
# ComfyUI (raise upload limit for videos; default 100 MB)
cd /workspace/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --max-upload-size 2048 &
# Proxy in its own venv
python3 -m venv /workspace/venv-proxy && /workspace/venv-proxy/bin/pip install comfy-api-proxy
/workspace/venv-proxy/bin/comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189 \
  --comfyui-base-dir /workspace/ComfyUI --max-upload-mb 2048 --state-dir /workspace/proxy-state &
curl http://127.0.0.1:8189/api/v2/health
```
From a laptop: `ssh -N -L 8189:127.0.0.1:8189 root@<pod-ip> -p <ssh-port>` (recommended). Alternatively bind publicly: `--host 0.0.0.0 --token $TOKEN`, expose 8189 in Runpod, `COMFY_BASE_URL=https://<POD_ID>-8189.proxy.runpod.net`, `COMFY_API_KEY=$TOKEN` (the SDK sends `Authorization: Bearer`, which matches `--token`). Runpod's HTTP proxy may cap long or large requests (**UNVERIFIED**), which is why SSH is safer for uploads.

Client:
```bash
pip install comfy-sdk            # 0.5.0
export COMFY_BASE_URL=http://127.0.0.1:8189   # no API key needed for self-hosted
```
```python
from comfy_sdk import Comfy
c = Comfy()
wf = c.workflows.from_file("editanything_api.json")     # must be API format (Export (API)); UI format raises WorkflowFormatUi
wf.set_input("5060", "video", c.assets.from_file("plate.mp4"))   # lazy upload: blake3 dedup, multipart, then core/ASSET ref
wf.set_input("5151:5138", "text", "Add bright blue lightning arcs around the person's right hand, center of frame.")  # ID: check export
job = c.run(wf)                  # submit + poll to completion; raises JobFailed with node details
for o in job.get_outputs("5152"):   # VHS_VideoCombine edited output
    o.to_file(o.name)            # o.type == "video" for VHS "gifs" outputs
```
- **Video assets:** yes. `assets.from_file` guesses MIME (video/mp4). The proxy accepts image/video/audio into `input/` (routes through ComfyUI `/upload/image`) and substitutes the filename for the `core/ASSET` ref, which VHS_LoadVideo/LoadVideo read. Raise **both** limits (`--max-upload-mb` on the proxy, `--max-upload-size` on ComfyUI), since both default to 100 MB.
- **Outputs:** the proxy maps ComfyUI history keys `images`→image, `gifs`→video (VHS_VideoCombine), `audio`, `text`, `files`, `latents`. Core `SaveVideo` reports under `images` and so is typed "image", but the MIME is still video/mp4 by filename. VHS outputs with `save_output=false` go to `temp/` and are still downloadable while the files exist.
- **Custom nodes:** yes. The proxy forwards the API graph verbatim to `/prompt`, so any pack installed in ComfyUI works, including PlateLock's own nodes. Only `extra_data.api_key_comfy_org` is accepted as extra data (not needed here).
- Optional: `--comfyui-base-dir` lets the SDK upload `.safetensors` straight into `models/*` (safetensors-header check, no traversal).

---

## 5. Setup script
`research/runpod_setup.sh` (not run). Idempotent: it clones only if missing, uses `aria2c -c` with size checks, and doesn't start servers that are already healthy. It does the following:
- apt installs git/aria2/ffmpeg.
- Creates a venv with `--system-site-packages`, with torch pinned by a constraints file so node requirements can't upgrade it.
- Installs ComfyUI, then LTXVideo, VHS, KJNodes, ComfyMath, Manager, and radiance (radiance is optional).
- Downloads the 3 core models (44.05 GB). `PROFILE=full` adds union control, Alissonerdx inpaint, the x2 upscaler, and the gated In/Outpainting LoRA if `HF_TOKEN` is set. `MODEL_VARIANT=dev` gets the template-faithful dev ckpt + distill LoRA.
- Saves the EditAnything, Inpaint, and Union Control workflow JSONs into `user/default/workflows`.
- Installs comfy-api-proxy in a separate venv and starts ComfyUI (:8188) and the proxy (:8189).
- Prints a missing-node check against `/object_info`.

Pod sizing: RTX 4090, **≥64 GB RAM**, **≥80 GB volume** at /workspace (≥100 GB for `PROFILE=full` + dev). Pick a PyTorch template with torch ≥2.6 / CUDA 12.4+ (latest ComfyUI may refuse older torch: **UNVERIFIED** minimum).

---

## Uncertain / unverified (check these first on the pod)
1. Peak VRAM, RAM, and runtime of 22B fp8 + IC guide at 1024x576x97 on a 4090.
2. Quality of the EditAnything LoRA (v1, 9000-step) on the distilled ckpt vs dev + distill LoRA. The author calls it experimental, and edits sometimes don't apply at all.
3. Flattened API node IDs for subgraph nodes (`"5151:5138"` style).
4. Whether `gemma_3_12B_it_fp8_scaled` loads in `LTXAVTextEncoderLoader`. It's very likely (same Comfy-Org repo, and DualCLIPLoader uses it in Alissonerdx's workflows), but not tested.
5. Whether radiance's dependencies install cleanly alongside ComfyUI (they're why replacing `Float32ColorCorrect` is recommended).
6. Lightricks inpaint workflow with the distilled-fp8 ckpt (written for dev bf16 + distill LoRA 0.5).
