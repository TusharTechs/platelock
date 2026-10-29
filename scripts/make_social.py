"""Vertical (9:16) social cut from the PlateLock share clips, captioned for muted autoplay.
    .venv/bin/python scripts/make_social.py   ->  demo/platelock_social_9x16.mp4
"""
import json
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX, WORK = ROOT / "examples", ROOT / ".runtime" / "social"
W, H = 1080, 1920
BOLD = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
GOLD, WHITE = (255, 201, 64), (255, 255, 255)

SHOTS = [("pocket-dragon", "\"Add a dragon on my phone\""),
         ("mothership", "\"Add an alien mothership\""),
         ("portal", "\"Open a portal in the sky\""),
         ("sea-serpent", "\"Add a sea monster\""),
         ("fireflies", "\"Add a spirit deer\""),
         ("lightning-road", "\"Add an explosion\"")]


def wrap_center(d, text, f, y, fill, maxw=980):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > maxw and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=f)
        d.text(((W - tw) / 2, y + i * (f.size + 14)), line, font=f, fill=fill, stroke_width=4, stroke_fill=(0, 0, 0))


def overlay(path, top, pct):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    wrap_center(d, top, ImageFont.truetype(BOLD, 64), 150, WHITE)
    wrap_center(d, "AI added ONLY this.", ImageFont.truetype(BOLD, 58), 330, GOLD)
    wrap_center(d, "original  ->  PlateLock", ImageFont.truetype(REG, 34), 1466, (200, 200, 210))
    wrap_center(d, f"{pct:.1f}% of the shot = original pixels", ImageFont.truetype(REG, 48), 1560, WHITE)
    wrap_center(d, "PlateLock  ·  LTX-2.3  ·  100% open weights", ImageFont.truetype(REG, 38), 1700, GOLD)
    img.save(path)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (slug, top) in enumerate(SHOTS):
        meta = json.loads((EX / slug / "meta.json").read_text())
        ov = WORK / f"o{i}.png"
        overlay(ov, top, meta["report"]["bit_identical_pct"])
        out = WORK / f"s{i}.mp4"
        reveal = "W*min(max((T-0.5)/0.9,0),1)"  # hold original 0.5s, wipe in the effect, then hold it
        band = "crop=1200:1080:360:0,scale=1080:972,setsar=1,format=yuv444p"
        fc = (f"[0:v]{band}[o];[1:v]split[c1][c2];[c1]{band}[c];"
              f"[c2]scale=-2:{H},crop={W}:{H},boxblur=30:3,eq=brightness=-0.25,format=yuv444p[bg];"
              f"[o][c]blend=c0_expr='if(lt(X,{reveal}),B,if(lt(X,{reveal}+5),235,A))'"
              f":c1_expr='if(lt(X,{reveal}),B,if(lt(X,{reveal}+5),128,A))'"
              f":c2_expr='if(lt(X,{reveal}),B,if(lt(X,{reveal}+5),128,A))'[w];"
              f"[bg][w]overlay=0:474[b];[b][2:v]overlay=0:0,format=yuv420p[v]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(EX / slug / "original.mp4"),
                        "-i", str(EX / slug / "comp.mp4"), "-loop", "1", "-i", str(ov),
                        "-filter_complex", fc, "-map", "[v]", "-t", "3.6", "-r", "24", "-an",
                        "-c:v", "libx264", "-crf", "20", str(out)], check=True)
        parts.append(out)
    # end card
    card = Image.new("RGB", (W, H), (10, 10, 15))
    d = ImageDraw.Draw(card)
    wrap_center(d, "PlateLock", ImageFont.truetype(BOLD, 120), 700, WHITE)
    wrap_center(d, "Add the VFX. Keep the shot.", ImageFont.truetype(BOLD, 60), 870, GOLD)
    wrap_center(d, "Open source on GitHub: TusharTechs/platelock", ImageFont.truetype(REG, 40), 1020, WHITE)
    wrap_center(d, "#OpenSourceHiggsfield", ImageFont.truetype(REG, 44), 1100, GOLD)
    card.save(WORK / "end.png")
    end = WORK / "end.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(WORK / "end.png"), "-t", "2.5", "-r", "24",
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", str(end)], check=True)
    parts.append(end)
    lst = WORK / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    final = ROOT / "demo" / "platelock_social_9x16.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
                    "-c:v", "libx264", "-crf", "21", "-c:a", "aac", "-movflags", "+faststart", str(final)], check=True)
    print("wrote", final)


if __name__ == "__main__":
    main()
