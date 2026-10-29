"""Minimal ffmpeg-backed video IO (frames as float [T,H,W,3] tensors)."""
import subprocess

import numpy as np
import torch


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height", "-of", "csv=p=0",
                                   str(path)])
    w, h = map(int, out.decode().strip().split(",")[:2])
    return w, h


def read_video(path, max_frames=None):
    w, h = probe(path)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    raw = subprocess.check_output(cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return torch.from_numpy(arr.copy()).float() / 255
