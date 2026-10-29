"""Vertical (9:16) social cut: original -> PlateLock reveals, captions for muted autoplay,
plus Kokoro (open-weights TTS) voice lines and synthesised whooshes for sound-on feeds.
    .runtime/venv-tts/bin/python scripts/make_social.py  ->  demo/platelock_social_9x16.mp4
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

SEG, END = 3.6, 3.0
LINES = ["AI video effects repaint your entire shot.",
         "PlateLock only adds the effect.",
         "Everything else stays your original footage.",
         "Bit for bit.",
         "Open weights. LTX two point three.",
         "Built in ComfyUI.",
         "PlateLock. Add the V F X. Keep the shot."]

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


AUDIO, SR = ROOT / ".runtime" / "audio", 48000


def _load(path):
    import numpy as np
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                                   "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32).copy()


def _place(track, clip, at, gain=1.0):
    a = max(0, int(at * SR))
    clip = clip[max(0, -int(at * SR)):]
    k = min(len(clip), len(track) - a)
    if k > 0:
        track[a:a + k] += clip[:k] * gain


def make_audio(n):
    """Kokoro voice lines + Mixkit SFX on every reveal + ducked music bed, mastered to -14 LUFS."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    total = n * SEG + END
    voice = np.zeros(int(total * SR) + SR, np.float32)
    sfx = np.zeros_like(voice)
    for i in range(n + 1):
        text = LINES[min(i, len(LINES) - 1)]
        room = (END if i == n else SEG) - 0.3
        speed = 1.1
        while True:
            v = np.concatenate([a for _, _, a in pipe(text, voice="af_heart", speed=speed)]).astype(np.float32)
            if len(v) / 24000 <= room or speed > 1.5:
                break
            speed += 0.1
        sf.write(WORK / f"l{i}.wav", v, 24000)
        _place(voice, _load(WORK / f"l{i}.wav"), i * SEG + 0.15, 2.0)
    fx = {k: _load(AUDIO / f"sfx_{k}.mp3") for k in ["impact_deep", "whoosh_fast", "logo_impact", "safe_lock", "sparkle"]}
    _place(sfx, fx["impact_deep"], 0.0, 0.25)
    for i in range(n):                       # whoosh peak (0.7s) on each wipe (crosses mid-frame ~0.95s in)
        _place(sfx, fx["whoosh_fast"], i * SEG + 0.25, 0.35)
    end = n * SEG
    _place(sfx, fx["logo_impact"], end - 4.6, 0.18)
    _place(sfx, fx["safe_lock"], end + 0.05, 0.7)
    _place(sfx, fx["sparkle"], end + 0.35, 0.3)
    sf.write(WORK / "voice.wav", voice[: int(total * SR)], SR)
    sf.write(WORK / "sfx.wav", sfx[: int(total * SR)], SR)
    out = WORK / "audio.wav"
    fc = (f"[2:a]atrim=start=30,asetpts=PTS-STARTPTS,aresample={SR},aformat=channel_layouts=mono,volume=0.16,"
          f"atrim=0:{total:.3f},afade=t=in:d=0.6,afade=t=out:st={total - 2:.2f}:d=2[m];[0:a]asplit[v1][v2];"
          "[m][v1]sidechaincompress=threshold=0.015:ratio=6:attack=15:release=350[md];"
          "[md][v2][1:a]amix=inputs=3:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9[a]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(WORK / "voice.wav"), "-i", str(WORK / "sfx.wav"),
                    "-i", str(AUDIO / "music_main.mp3"), "-filter_complex", fc, "-map", "[a]", "-ac", "2",
                    "-ar", str(SR), str(out)], check=True)
    return out


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
                        "-filter_complex", fc, "-map", "[v]", "-t", str(SEG), "-r", "24", "-an",
                        "-c:v", "libx264", "-crf", "20", str(out)], check=True)
        parts.append(out)
    # end card
    card = Image.new("RGBA", (W, H), (10, 10, 15, 255))
    d = ImageDraw.Draw(card)
    lk = Image.open(ROOT / "brand" / "logo-lockup.png").convert("RGBA")
    lk = lk.resize((980, int(lk.height * 980 / lk.width)), Image.LANCZOS)
    card.alpha_composite(lk, ((W - lk.width) // 2, 640))
    wrap_center(d, "Add the VFX. Keep the shot.", ImageFont.truetype(BOLD, 60), 960, GOLD)
    wrap_center(d, "Open source: github.com/TusharTechs/platelock", ImageFont.truetype(REG, 40), 1100, WHITE)
    wrap_center(d, "#OpenSourceHiggsfield", ImageFont.truetype(REG, 44), 1180, GOLD)
    card = card.convert("RGB")
    card.save(WORK / "end.png")
    end = WORK / "end.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(WORK / "end.png"), "-t", str(END), "-r", "24",
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", str(end)], check=True)
    parts.append(end)
    lst = WORK / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    silent = WORK / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(silent)], check=True)
    audio = make_audio(len(SHOTS))
    final = ROOT / "demo" / "platelock_social_9x16.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                    str(final)], check=True)
    print("wrote", final)


if __name__ == "__main__":
    main()
