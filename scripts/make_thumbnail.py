"""YouTube thumbnails (1280x720) from real PlateLock outputs.
    .venv/bin/python scripts/make_thumbnail.py  ->  brand/thumbnail_a.png, brand/thumbnail_b.png
"""
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
W, H = 1280, 720
BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
GOLD, PINK, WHITE = (255, 201, 64), (255, 79, 139), (255, 255, 255)


def frame(path, n=60):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n}),scale=1920:1080",
                                   "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)


def text(d, xy, s, size, fill, font=BLACK, anchor="la", stroke=6):
    d.text(xy, s, font=ImageFont.truetype(font, size), fill=fill, anchor=anchor,
           stroke_width=stroke, stroke_fill=(0, 0, 0))


def pill(d, xy, s, size, fg, bg, anchor="lt"):
    f = ImageFont.truetype(BOLD, size)
    x, y = xy
    tw = d.textlength(s, font=f)
    if anchor == "rt":
        x -= tw + 36
    d.rounded_rectangle([x, y, x + tw + 36, y + size + 24], 14, fill=bg)
    d.text((x + 18, y + 10), s, font=f, fill=fg)


def mark(size):
    return Image.open(ROOT / "brand" / "logo-mark-512.png").convert("RGBA").resize((size, size), Image.LANCZOS)


def variant_a():
    """Split: what raw AI changed (difference map) vs PlateLock (only the UFO). Same crop on both halves."""
    slug = "mothership"
    orig, raw, comp = frame(EX / slug / "original.mp4"), frame(EX / slug / "raw.mp4"), frame(EX / slug / "comp.mp4")
    diff = np.clip(np.abs(raw.astype(int) - orig.astype(int)) * 8, 0, 255).astype(np.uint8)
    box = (480, 0, 1440, 1080)  # centred 0.889-aspect crop that contains the whole ship
    half = (W // 2, H)
    left = Image.fromarray(diff).crop(box).resize(half, Image.LANCZOS)
    right = Image.fromarray(comp).crop(box).resize(half, Image.LANCZOS)
    img = Image.new("RGB", (W, H))
    img.paste(left, (0, 0)); img.paste(right, (W // 2, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.line([(W // 2, 0), (W // 2, H)], fill=WHITE, width=6)
    pill(d, (24, 24), "RAW AI", 38, WHITE, (255, 79, 139, 235))
    pill(d, (W - 24, 24), "PLATELOCK", 38, (20, 14, 0), (255, 201, 64, 245), anchor="rt")
    d.rectangle([0, H - 200, W, H], fill=(0, 0, 0, 165))
    text(d, (W // 2, H - 186), "AI ADDED ONLY THE UFO", 78, WHITE, anchor="ma")
    text(d, (W // 2, H - 88), "93.6% of the shot = original pixels", 44, GOLD, font=BOLD, anchor="ma", stroke=4)
    m = mark(130)
    img.paste(m, (W // 2 - 65, H // 2 - 120), m)
    return img


def variant_b():
    """Hero: the pocket dragon, big claim, logo."""
    slug = "pocket-dragon"
    comp = Image.fromarray(frame(EX / slug / "comp.mp4", 70)).resize((W, H), Image.LANCZOS)
    # zoom towards the dragon (centre-ish) for a punchy crop
    cx, cy, z = 0.5, 0.42, 1.35
    cw, ch = W / z, H / z
    comp = comp.crop((cx * W - cw / 2, cy * H - ch / 2, cx * W + cw / 2, cy * H + ch / 2)).resize((W, H), Image.LANCZOS)
    img = comp.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 210], fill=(0, 0, 0, 140))
    text(d, (40, 24), "AI added the dragon.", 74, WHITE)
    text(d, (40, 112), "Nothing else changed.", 74, GOLD)
    pill(d, (40, H - 96), "95.8% of the shot = original pixels", 36, (20, 14, 0), (255, 201, 64, 240))
    m = mark(170)
    img.paste(m, (W - 210, H - 220), m)
    text(d, (W - 125, H - 40), "PlateLock", 34, WHITE, font=BLACK, anchor="ms", stroke=4)
    return img


if __name__ == "__main__":
    for name, fn in [("a", variant_a), ("b", variant_b)]:
        out = ROOT / "brand" / f"thumbnail_{name}.png"
        fn().save(out, optimize=True)
        print("wrote", out, out.stat().st_size // 1024, "KB")
