#!/usr/bin/env bash
# PlateLock: Runpod (L40S 48GB / RTX 4090) bootstrap for ComfyUI + LTX-2.3 EditAnything + comfy-api-proxy.
#
# Target: a fresh Runpod pod (PyTorch template, Ubuntu, CUDA, persistent volume at /workspace).
# Safe to re-run: finished steps are skipped, partial downloads resume, running servers are left alone.
#
# Usage:
#   bash runpod_setup.sh                    # install everything needed for EditAnything, then start both servers
#   PROFILE=full bash runpod_setup.sh       # also download the fallback models (union control, inpaint, upscaler)
#   MODEL_VARIANT=dev bash runpod_setup.sh  # template-faithful dev-fp8 + distill LoRA 384 (about 7 GB more)
#   SKIP_START=1 bash runpod_setup.sh       # install only
#   ONLY_START=1 bash runpod_setup.sh       # only (re)start servers that are not running
#
# Optional env:
#   HF_TOKEN=hf_xxx        needed only for the gated Lightricks In/Outpainting IC-LoRA (PROFILE=full)
#   PROXY_TOKEN=xxx        bind the proxy on 0.0.0.0 with bearer auth (otherwise it stays on 127.0.0.1)
#   COMFY_LISTEN=0.0.0.0   ComfyUI bind address (default 0.0.0.0 so the Runpod HTTP proxy can reach :8188)
#   COMFY_EXTRA_ARGS="--fast fp8_matrix_mult"   extra ComfyUI flags
#   INSTALL_RADIANCE=0     skip the FXTD radiance pack (only provides Float32ColorCorrect; heavy deps)
#   TORCH_CUDA=cu128       PyTorch wheel index tag, used only if torchaudio has to be installed

set -Eeuo pipefail

WS="${WS:-/workspace}"
COMFY_DIR="${COMFY_DIR:-$WS/ComfyUI}"
VENV="${VENV:-$WS/venv}"
PROXY_VENV="${PROXY_VENV:-$WS/venv-proxy}"
LOG_DIR="${LOG_DIR:-$WS/logs}"
PROFILE="${PROFILE:-min}"              # min | full
MODEL_VARIANT="${MODEL_VARIANT:-distilled}"  # distilled | dev
COMFY_PORT="${COMFY_PORT:-8188}"
PROXY_PORT="${PROXY_PORT:-8189}"
COMFY_LISTEN="${COMFY_LISTEN:-0.0.0.0}"
COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS:-}"
MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-2048}"
INSTALL_RADIANCE="${INSTALL_RADIANCE:-1}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

log()  { printf '\n\033[1;36m[platelock]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[platelock WARN]\033[0m %s\n' "$*" >&2; }

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- system packages
install_system() {
  local need=()
  for b in git aria2c ffmpeg curl; do command -v "$b" >/dev/null 2>&1 || need+=("$b"); done
  if ((${#need[@]})); then
    log "apt: installing ${need[*]}"
    local pkgs=()
    for b in "${need[@]}"; do case "$b" in aria2c) pkgs+=(aria2);; *) pkgs+=("$b");; esac; done
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends "${pkgs[@]}" python3-venv
  fi
}

# ---------------------------------------------------------------- python env
# The venv sees the template's system torch (--system-site-packages), so torch is not reinstalled.
# The constraints file pins torch/torchvision/torchaudio so custom node requirements cannot upgrade them.
setup_venv() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    log "Creating venv $VENV (system site packages, so the template's torch is reused)"
    python3 -m venv --system-site-packages "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -q -U pip wheel setuptools

  python - <<'PY' > "$WS/torch-constraints.txt"
import importlib
for m in ("torch", "torchvision", "torchaudio"):
    try:
        v = importlib.import_module(m).__version__.split("+")[0]
        print(f"{m}=={v}")
    except Exception:
        pass
PY
  log "Pinned torch stack: $(tr '\n' ' ' < "$WS/torch-constraints.txt")"
  if ! grep -q '^torch==' "$WS/torch-constraints.txt"; then
    warn "torch not found in the base image; installing torch from the $TORCH_CUDA index"
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    python - <<'PY' > "$WS/torch-constraints.txt"
import torch, torchvision, torchaudio
for m in (torch, torchvision, torchaudio):
    print(f"{m.__name__}=={m.__version__.split('+')[0]}")
PY
  elif ! grep -q '^torchaudio==' "$WS/torch-constraints.txt"; then
    local tv; tv="$(sed -n 's/^torch==//p' "$WS/torch-constraints.txt")"
    log "Installing torchaudio==$tv to match torch (index $TORCH_CUDA)"
    pip install "torchaudio==$tv" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
      || warn "torchaudio install failed; LTX audio nodes may not import"
    echo "torchaudio==$tv" >> "$WS/torch-constraints.txt"
  fi
  export PIP_CONSTRAINT="$WS/torch-constraints.txt"
  python -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda,"gpu",torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
  pip install -q -U "huggingface_hub[hf_xet]"
}

# ---------------------------------------------------------------- ComfyUI + custom nodes
clone_or_skip() {  # url dest
  local url="$1" dest="$2"
  if [[ -d "$dest/.git" ]]; then
    log "present: $(basename "$dest") (set UPDATE_NODES=1 to git pull)"
    [[ "${UPDATE_NODES:-0}" == 1 ]] && git -C "$dest" pull --ff-only || true
  else
    log "cloning $url"
    git clone --depth 1 "$url" "$dest"
  fi
}

install_comfy() {
  clone_or_skip https://github.com/Comfy-Org/ComfyUI "$COMFY_DIR"
  pip install -q -r "$COMFY_DIR/requirements.txt"

  local CN="$COMFY_DIR/custom_nodes"
  # Needed by the EditAnything workflow
  # PlateLock's own workflow needs only LTXVideo (+ the PlateLock pack below).
  local nodes=(https://github.com/Lightricks/ComfyUI-LTXVideo)
  if [[ "${TEMPLATE_NODES:-0}" == 1 ]]; then  # to open the original EditAnything template in the UI
    nodes+=(
      https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
      https://github.com/kijai/ComfyUI-KJNodes
      https://github.com/evanspearman/ComfyMath
      https://github.com/Comfy-Org/ComfyUI-Manager
    )
    [[ "$INSTALL_RADIANCE" == 1 ]] && nodes+=(https://github.com/FXTD-Studios/radiance)
  fi
  for url in "${nodes[@]}"; do
    local name; name="$(basename "$url")"
    clone_or_skip "$url" "$CN/$name"
    if [[ -f "$CN/$name/requirements.txt" ]]; then
      pip install -q -r "$CN/$name/requirements.txt" || warn "requirements failed for $name (continuing)"
    fi
  done

  # PlateLock node pack: from the repo this script lives in, or PLATELOCK_REPO
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [[ -d "$here/comfyui-platelock" ]]; then
    ln -sfn "$here/comfyui-platelock" "$CN/comfyui-platelock"
    log "linked PlateLock nodes from $here"
  elif [[ -n "${PLATELOCK_REPO:-}" ]]; then
    clone_or_skip "$PLATELOCK_REPO" "$WS/platelock"
    ln -sfn "$WS/platelock/comfyui-platelock" "$CN/comfyui-platelock"
  else
    warn "PlateLock nodes not found; run this script from the platelock repo"
  fi
}

# ---------------------------------------------------------------- models
# fetch <url> <dest_dir> <filename> [expected_bytes]
fetch() {
  local url="$1" dir="$2" name="$3" want="${4:-}"
  local dest="$dir/$name"
  mkdir -p "$dir"
  if [[ -f "$dest" && ! -f "$dest.aria2" ]]; then
    if [[ -z "$want" || "$(stat -c %s "$dest")" == "$want" ]]; then
      log "have $name"; return 0
    fi
    warn "$name size mismatch ($(stat -c %s "$dest") != $want), resuming"
  fi
  log "downloading $name -> $dir"
  local hdr=()
  [[ -n "${HF_TOKEN:-}" ]] && hdr=(--header="Authorization: Bearer $HF_TOKEN")
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 16 -s 16 -k 50M --file-allocation=none --console-log-level=warn \
      --summary-interval=30 "${hdr[@]}" -d "$dir" -o "$name" "$url"
  else
    local wh=(); [[ -n "${HF_TOKEN:-}" ]] && wh=(--header="Authorization: Bearer $HF_TOKEN")
    wget -c "${wh[@]}" -O "$dest" "$url"
  fi
  if [[ -n "$want" && "$(stat -c %s "$dest")" != "$want" ]]; then
    warn "$name finished with unexpected size $(stat -c %s "$dest") (expected $want)"
  fi
}

download_models() {
  local M="$COMFY_DIR/models"
  local HF=https://huggingface.co

  # --- EditAnything, minimal 24 GB setup (about 44 GB) ---
  if [[ "$MODEL_VARIANT" == dev ]]; then
    fetch "$HF/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
      "$M/checkpoints" ltx-2.3-22b-dev-fp8.safetensors 29145431166
    fetch "$HF/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors" \
      "$M/loras" ltx-2.3-22b-distilled-lora-384.safetensors 7605507256
  else
    fetch "$HF/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors" \
      "$M/checkpoints" ltx-2.3-22b-distilled-fp8.safetensors 29531884062
  fi
  fetch "$HF/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    "$M/text_encoders" gemma_3_12B_it_fp8_scaled.safetensors 13205434827
  fetch "$HF/Alissonerdx/LTX-LoRAs/resolve/main/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors" \
    "$M/loras" ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors 1308756416

  # --- Fallbacks (PROFILE=full), about 4.6 GB more ---
  if [[ "$PROFILE" == full ]]; then
    fetch "$HF/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
      "$M/loras" ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 654465352
    fetch "$HF/Alissonerdx/LTX-LoRAs/resolve/main/ltx23_inpaint_masked_t2v_rank128_v1_02500steps.safetensors" \
      "$M/loras" ltx23_inpaint_masked_t2v_rank128_v1_02500steps.safetensors 1308756416
    fetch "$HF/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
      "$M/latent_upscale_models" ltx-2.3-spatial-upscaler-x2-1.1.safetensors 995743560
    if [[ -n "${HF_TOKEN:-}" ]]; then
      # Gated (auto-approve): accept the terms at https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting first
      fetch "$HF/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/resolve/main/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors" \
        "$M/loras" ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors \
        || warn "In/Outpainting LoRA download failed (accepted the gate terms?)"
    else
      warn "HF_TOKEN unset: skipping the gated Lightricks In/Outpainting IC-LoRA"
    fi
  fi
}

# ---------------------------------------------------------------- workflows (JSON only, loaded by the UI)
fetch_workflows() {
  local W="$COMFY_DIR/user/default/workflows"
  mkdir -p "$W"
  [[ -s "$W/LTX23_EditAnything_1fb07c3b4b99.json" ]] || \
    curl -fsSL -o "$W/LTX23_EditAnything_1fb07c3b4b99.json" https://comfy.org/workflows/download/1fb07c3b4b99.json || true
  local LX=https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/2.3
  for f in LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json LTX-2.3_ICLoRA_Union_Control_Distilled.json; do
    [[ -s "$W/$f" ]] || curl -fsSL -o "$W/$f" "$LX/$f" || true
  done
}

# ---------------------------------------------------------------- comfy-api-proxy (separate venv to avoid pydantic/aiohttp pin clashes)
install_proxy() {
  if [[ ! -x "$PROXY_VENV/bin/comfy-api-proxy" ]]; then
    log "Installing comfy-api-proxy into $PROXY_VENV"
    python3 -m venv "$PROXY_VENV"
    "$PROXY_VENV/bin/pip" install -q -U pip
    "$PROXY_VENV/bin/pip" install -q comfy-api-proxy
  else
    log "comfy-api-proxy present ($("$PROXY_VENV/bin/pip" show comfy-api-proxy 2>/dev/null | sed -n 's/^Version: //p'))"
  fi
}

# ---------------------------------------------------------------- start
wait_http() {  # url seconds
  local url="$1" t="${2:-300}"
  for ((i = 0; i < t; i += 2)); do curl -fs -o /dev/null "$url" && return 0; sleep 2; done
  return 1
}

start_comfy() {
  if curl -fs -o /dev/null "http://127.0.0.1:$COMFY_PORT/system_stats"; then
    log "ComfyUI already running on :$COMFY_PORT"; return 0
  fi
  log "Starting ComfyUI on $COMFY_LISTEN:$COMFY_PORT (log: $LOG_DIR/comfyui.log)"
  [[ "$COMFY_LISTEN" == 0.0.0.0 ]] && warn "ComfyUI has no auth; anyone with the Runpod proxy URL can use it."
  # shellcheck disable=SC2086
  (cd "$COMFY_DIR" && nohup "$VENV/bin/python" main.py --listen "$COMFY_LISTEN" --port "$COMFY_PORT" \
      --max-upload-size "$MAX_UPLOAD_MB" $COMFY_EXTRA_ARGS >> "$LOG_DIR/comfyui.log" 2>&1 &
   echo $! > "$WS/comfyui.pid")
  wait_http "http://127.0.0.1:$COMFY_PORT/system_stats" 600 \
    || { warn "ComfyUI did not come up; tail $LOG_DIR/comfyui.log"; tail -n 50 "$LOG_DIR/comfyui.log"; exit 1; }
  log "ComfyUI up. Missing-node check:"
  curl -fs "http://127.0.0.1:$COMFY_PORT/object_info" | python3 -c '
import json,sys
have=set(json.load(sys.stdin))
need="""LTXAddVideoICLoRAGuide LTXICLoRALoaderModelOnly LTXAVTextEncoderLoader LTXVAudioVAELoader
LTXVConcatAVLatent LTXVSeparateAVLatent LTXVCropGuides LTXVEmptyLatentAudio EmptyLTXVLatentVideo ManualSigmas
SamplerCustomAdvanced CFGGuider LoadVideo GetVideoComponents CreateVideo SaveVideo PlateLockPrepPlate
PlateLockDifferenceMatte PlateLockComposite PlateLockFidelityReport PlateLockBeforeAfter PlateLockGamma""".split()
miss=[n for n in need if n not in have]
print("  all PlateLock workflow nodes present" if not miss else "  MISSING: "+" ".join(miss))'
}

start_proxy() {
  if curl -fs -o /dev/null "http://127.0.0.1:$PROXY_PORT/api/v2/health"; then
    log "comfy-api-proxy already running on :$PROXY_PORT"; return 0
  fi
  local args=(--comfyui "http://127.0.0.1:$COMFY_PORT" --port "$PROXY_PORT"
              --comfyui-base-dir "$COMFY_DIR" --max-upload-mb "$MAX_UPLOAD_MB"
              --state-dir "$WS/proxy-state")
  if [[ -n "${PROXY_TOKEN:-}" ]]; then
    args+=(--host 0.0.0.0 --token "$PROXY_TOKEN")
  fi
  mkdir -p "$WS/proxy-state"
  log "Starting comfy-api-proxy on :$PROXY_PORT (log: $LOG_DIR/proxy.log)"
  nohup "$PROXY_VENV/bin/comfy-api-proxy" "${args[@]}" >> "$LOG_DIR/proxy.log" 2>&1 &
  echo $! > "$WS/proxy.pid"
  wait_http "http://127.0.0.1:$PROXY_PORT/api/v2/health" 60 \
    || { warn "proxy did not come up; tail $LOG_DIR/proxy.log"; tail -n 30 "$LOG_DIR/proxy.log"; exit 1; }
  log "Proxy health: $(curl -fs "http://127.0.0.1:$PROXY_PORT/api/v2/health")"
}

preflight() {
  local ram_gb disk_gb
  ram_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
  disk_gb=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
  log "RAM ${ram_gb} GB, free disk on $WS ${disk_gb} GB"
  ((ram_gb < 48)) && warn "Under 48 GB system RAM. The 22B fp8 model is partly offloaded from the 24 GB GPU into RAM, so expect OOM or swapping. Pick a pod with 64 GB+ RAM."
  local need=60; [[ "$PROFILE" == full ]] && need=70; [[ "$MODEL_VARIANT" == dev ]] && need=$((need + 10))
  ((disk_gb < need)) && warn "Less than ${need} GB free on $WS; the downloads may not fit (ComfyUI and the venv need about 10 GB more)."
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
}

main() {
  if [[ "${ONLY_START:-0}" == 1 ]]; then
    start_comfy; start_proxy; exit 0
  fi
  preflight
  install_system
  mkdir -p "$COMFY_DIR/models"
  log "Starting model downloads in the background (log: $LOG_DIR/downloads.log)"
  ( download_models >> "$LOG_DIR/downloads.log" 2>&1; echo "exit=$?" >> "$LOG_DIR/downloads.log" ) &
  local dl_pid=$!
  # clone into the (possibly pre-created) ComfyUI dir: models/ already exists, so clone to a temp dir
  if [[ ! -d "$COMFY_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/Comfy-Org/ComfyUI "$WS/.comfy-src"
    cp -rn "$WS/.comfy-src/." "$COMFY_DIR/" && rm -rf "$WS/.comfy-src"
  fi
  setup_venv
  install_comfy
  install_proxy
  fetch_workflows
  log "Waiting for model downloads to finish…"
  wait "$dl_pid" || true
  tail -n 5 "$LOG_DIR/downloads.log"
  if [[ "${SKIP_START:-0}" != 1 ]]; then
    start_comfy
    start_proxy
  fi
  cat <<EOF

[platelock] Done.
  ComfyUI UI : http://<pod>:$COMFY_PORT  (Runpod: https://<POD_ID>-$COMFY_PORT.proxy.runpod.net if the port is exposed)
  API proxy  : http://127.0.0.1:$PROXY_PORT/api/v2  (from your laptop: ssh -N -L $PROXY_PORT:127.0.0.1:$PROXY_PORT root@<pod-ip> -p <ssh-port>)
  SDK        : pip install comfy-sdk ; export COMFY_BASE_URL=http://127.0.0.1:$PROXY_PORT
  Logs       : $LOG_DIR/comfyui.log  $LOG_DIR/proxy.log
  Stop       : kill \$(cat $WS/comfyui.pid) \$(cat $WS/proxy.pid)
EOF
}

main "$@"
