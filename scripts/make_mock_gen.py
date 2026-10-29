"""Make a fake 'LTX output' from a real clip so the PlateLock pipeline can be
tested without a GPU: model-res, softened, tone-drifted, shimmering, plus an
added fireball. Usage: make_mock_gen.py in.mp4 out.mp4 [frames]"""
import math
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/comfyui-platelock")
from platelock import core  # noqa: E402


def read_frames(path, n):
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                     "-show_entries", "stream=width,height", "-of", "csv=p=0", path])
    w, h = map(int, probe.decode().strip().split(","))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(n),
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return torch.from_numpy(arr.copy()).float() / 255


def write_frames(frames, path, fps=24):
    t, h, w, _ = frames.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                          "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write((frames.clamp(0, 1) * 255).round().byte().numpy().tobytes())
    p.stdin.close()
    p.wait()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 49
    plate = read_frames(src, n)
    _, proxy, pw, ph, n = core.prep_plate(plate, 768, n)
    g = torch.Generator().manual_seed(1)
    ys = torch.linspace(0, 1, ph).view(ph, 1)
    xs = torch.linspace(0, 1, pw).view(1, pw)
    out = []
    for i in range(n):
        f = proxy[i]
        soft = core.gaussian_blur(f.permute(2, 0, 1)[None], 1.0)[0].permute(1, 2, 0)
        gen = soft * torch.tensor([1.04, 0.98, 0.95]) + 0.01 + torch.randn(ph, pw, 3, generator=g) * 0.008
        u = i / max(1, n - 1)
        cx, cy = 0.62 - 0.1 * u, 0.15 + 0.5 * u
        r = torch.sqrt(((xs - cx) * pw / ph) ** 2 + (ys - cy) ** 2)
        core_ = torch.exp(-(r / 0.03) ** 2)
        flame = torch.exp(-(r / (0.07 + 0.02 * math.sin(i))) ** 2) * (0.8 + 0.2 * torch.rand(ph, pw, generator=g))
        trail = torch.exp(-(((xs - cx - 0.04 * (ys - cy) / 0.5) * pw / ph) / 0.025) ** 2) * ((ys < cy) & (ys > cy - 0.35)).float() * 0.7
        glow = (flame + trail).clamp(0, 1)[..., None]
        fire = torch.tensor([1.0, 0.45, 0.08]) * glow + core_[..., None] * torch.tensor([0.0, 0.5, 0.8])
        gen = gen * (1 - glow * 0.85) + fire
        out.append(gen.clamp(0, 1))
    write_frames(torch.stack(out), dst)
    print(f"wrote {dst}: {n} frames {pw}x{ph}")


if __name__ == "__main__":
    main()
