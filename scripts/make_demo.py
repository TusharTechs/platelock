"""Build the PlateLock demo video (<= 2 min) from real gallery outputs.

Narration: Kokoro-82M (open weights TTS).  Visuals: ffmpeg + PIL captions.
    .runtime/venv-tts/bin/python scripts/make_demo.py [--screen path/to/app_recording.mov]
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
WORK = ROOT / ".runtime" / "demo"
W, H, FPS = 1920, 1080, 24
GOLD, PINK, WHITE, MUTED = (255, 201, 64), (255, 79, 139), (236, 235, 244), (154, 152, 173)
F_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
F_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
F_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
F_MONO = "/System/Library/Fonts/Menlo.ttc"


def font(path, size):
    return ImageFont.truetype(path, size)


def run(cmd):
    subprocess.run(cmd, check=True)


def manifest():
    return {m["slug"]: m for m in json.loads((EX / "manifest.json").read_text())}


# ------------------------------------------------------------------ narration
def tts(text, out, voice="af_heart", speed=1.05):
    import soundfile as sf
    from kokoro import KPipeline
    global _PIPE
    try:
        _PIPE
    except NameError:
        _PIPE = KPipeline(lang_code="a")
    audio = np.concatenate([a for _, _, a in _PIPE(text, voice=voice, speed=speed)])
    sf.write(out, audio, 24000)
    return len(audio) / 24000


# ------------------------------------------------------------------ overlays
def overlay_png(path, caption=None, labels=(), title=None, sub=None, big=None, stats=None, bg=None,
                logo=None, lockup=None, bug=True):
    """Full-frame RGBA overlay. labels: [(x, y, text, color)]"""
    img = Image.new("RGBA", (W, H), (10, 10, 15, 255) if bg else (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg:
        for y in range(H):  # subtle vertical gradient
            c = int(10 + 14 * (1 - y / H))
            d.line([(0, y), (W, y)], fill=(c, c - 2, c + 8, 255))
    if lockup:
        lk = Image.open(ROOT / "brand" / "logo-lockup.png").convert("RGBA")
        lk = lk.resize((1120, int(lk.height * 1120 / lk.width)), Image.LANCZOS)
        img.alpha_composite(lk, ((W - lk.width) // 2, 250))
    if logo:
        mk = Image.open(ROOT / "brand" / "logo-mark-512.png").convert("RGBA").resize((170, 170), Image.LANCZOS)
        img.alpha_composite(mk, ((W - 170) // 2, 130))
    if title:
        f = font(F_BLACK, 104)
        tw = d.textlength(title, font=f)
        d.text(((W - tw) / 2, 330 if sub else 440), title, font=f, fill=WHITE)
    if sub:
        f = font(F_BOLD, 64)
        for i, line in enumerate(sub.split("\n")):
            tw = d.textlength(line, font=f)
            d.text(((W - tw) / 2, (640 if lockup else 480) + i * 84), line, font=f, fill=GOLD)
    if big:
        for i, (txt, col, size) in enumerate(big):
            f = font(F_BLACK, size)
            tw = d.textlength(txt, font=f)
            d.text(((W - tw) / 2, 260 + sum(b[2] * 1.25 for b in big[:i])), txt, font=f, fill=col)
    if stats:
        n = len(stats)
        cw = 520
        x0 = (W - n * cw - (n - 1) * 40) / 2
        for i, (num, col, lbl) in enumerate(stats):
            x = x0 + i * (cw + 40)
            d.rounded_rectangle([x, 330, x + cw, 760], 28, fill=(19, 19, 27, 255), outline=(42, 42, 58, 255), width=2)
            size = 116
            f = font(F_BLACK, size)
            while d.textlength(num, font=f) > cw - 60 and size > 40:
                size -= 4
                f = font(F_BLACK, size)
            tw = d.textlength(num, font=f)
            d.text((x + (cw - tw) / 2, 400 + (116 - size) * 0.6), num, font=f, fill=col)
            f2 = font(F_BOLD, 34)
            for j, line in enumerate(lbl.split("\n")):
                tw = d.textlength(line, font=f2)
                d.text((x + (cw - tw) / 2, 590 + j * 46), line, font=f2, fill=WHITE if j == 0 else MUTED)
    for (x, y, txt, col) in labels:
        f = font(F_MONO, 30)
        tw = d.textlength(txt, font=f)
        d.rounded_rectangle([x, y, x + tw + 28, y + 50], 10, fill=(0, 0, 0, 190))
        d.text((x + 14, y + 8), txt, font=f, fill=col)
    if caption:
        f = font(F_BOLD, 46)
        lines = caption.split("\n")
        bh = 40 + 60 * len(lines)
        d.rectangle([0, H - bh - 40, W, H - 40], fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            tw = d.textlength(line, font=f)
            d.text(((W - tw) / 2, H - bh - 20 + i * 60), line, font=f, fill=WHITE)
    # brand bug
    if bug and not lockup:
        mk = Image.open(ROOT / "brand" / "logo-mark-512.png").convert("RGBA").resize((54, 54), Image.LANCZOS)
        img.alpha_composite(mk, (W - 84, 26))
    img.save(path)


# ------------------------------------------------------------------ segment builders
def seg_video(out, dur, inputs, filter_body, overlay):
    """inputs: list of video paths (looped); filter_body must produce [base]."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in inputs:
        cmd += ["-stream_loop", "-1", "-i", str(p)]
    cmd += ["-loop", "1", "-i", str(overlay)]
    k = len(inputs)
    fc = filter_body + f";[base][{k}:v]overlay=0:0,format=yuv420p[v]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.3f}", "-r", str(FPS),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", str(out)]
    run(cmd)


def seg_card(out, dur, overlay):
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(overlay), "-t", f"{dur:.3f}", "-r", str(FPS),
         "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)])


def montage_body(n):
    parts = [f"color=c=0x0a0a0f:s={W}x{H}[bg]"]
    parts += [fit(i, 620, 348) + f"[g{i}]" for i in range(n)]
    prev = "bg"
    for i in range(n):
        x, y = 20 + (i % 3) * 630, 150 + (i // 3) * 380
        nxt = "base" if i == n - 1 else f"o{i}"
        parts.append(f"[{prev}][g{i}]overlay={x}:{y}[{nxt}]")
        prev = nxt
    return ";".join(parts)


def seg_pan(out, dur, image, y0, y1, overlay):
    """Scroll down a tall 1440-wide screenshot, scaled to 1080p."""
    ease = f"(1-cos(PI*min(t/{dur:.3f},1)))/2"
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", image, "-loop", "1", "-i", str(overlay),
         "-filter_complex", f"[0:v]crop=1440:810:0:'{y0}+({y1}-{y0})*{ease}',scale={W}:{H}[b];"
                            "[b][1:v]overlay=0:0,format=yuv420p[v]",
         "-map", "[v]", "-t", f"{dur:.3f}", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", str(out)])


def fit(i, w, h, extra=""):
    return f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0a0a0f{extra}"


# ------------------------------------------------------------------ the film
def build(screen=None):
    WORK.mkdir(parents=True, exist_ok=True)
    m = manifest()

    def f(slug, name):
        return EX / slug / f"{name}.mp4"

    def pct(slug):
        return m[slug]["report"]["bit_identical_pct"]

    def db(slug):
        return m[slug]["report"]["raw_model_psnr_outside_effect_db"]

    have = [s for s in ["energy-orb", "pocket-dragon", "mothership", "code-on-fire", "portal", "meteor",
                        "sea-serpent", "lightning-road", "fireflies", "hologram"] if s in m]
    hero = have[:3]
    pcts = [pct(s) for s in have]
    dbs = [db(s) for s in have]
    avg_pct, min_db, max_db = sum(pcts) / len(pcts), min(dbs), max(dbs)

    segs = []

    # 1. hook: full-screen wipes original -> PlateLock
    hero = [x for x in ["mothership", "pocket-dragon", "portal"] if x in m] or hero
    per = 3.6
    wipes, ins = [], []
    for i, sl in enumerate(hero):
        ins += [f(sl, "original"), f(sl, "comp")]
        a, b = 2 * i, 2 * i + 1
        # wipe line travels right->left revealing the PlateLock comp
        pos = f"W*(0.97-0.94*min(T/{per},1))"
        wipes.append(f"[{a}:v]scale={W}:{H},trim=0:{per},setpts=PTS-STARTPTS,format=yuv444p[o{i}];"
                     f"[{b}:v]scale={W}:{H},trim=0:{per},setpts=PTS-STARTPTS,format=yuv444p[c{i}];"
                     f"[o{i}][c{i}]blend=c0_expr='if(lt(X,{pos}),A,if(lt(X,{pos}+4),235,B))'"
                     f":c1_expr='if(lt(X,{pos}),A,if(lt(X,{pos}+4),128,B))'"
                     f":c2_expr='if(lt(X,{pos}),A,if(lt(X,{pos}+4),128,B))'[w{i}]")
    segs.append(dict(
        text=("Stock footage. Plus an alien mothership, a pocket dragon, and a portal in the sky. "
              "And every pixel the effects don't need is still the original footage. Bit for bit."),
        kind="video", inputs=ins,
        body=";".join(wipes) + ";" + "".join(f"[w{i}]" for i in range(len(hero))) +
             f"concat=n={len(hero)}:v=1:a=0,format=yuv420p[base]",
        ov=dict(labels=[(40, 90, "ORIGINAL  ◂", WHITE), (W - 290, 90, "▸  PLATELOCK", GOLD)],
                caption="Add the VFX. Keep the shot.")))

    # 2. problem: what the raw model changed (amplified difference)
    z = "mothership" if "mothership" in m else hero[0]
    segs.append(dict(
        text=("That's not how AI video effects usually work. Higgsfield-style effects, and every open "
              "video-to-video model, repaint the entire frame. On the right is everything raw LTX changed, "
              "amplified eight times. Not just the ship: every building, every window, every frame. "
              f"Outside the effect, raw output scores just {min_db:.0f} to {max_db:.0f} decibels."),
        kind="video", inputs=[f(z, "raw"), f(z, "original")],
        body=f"color=c=0x0a0a0f:s={W}x{H}[bg];[0:v]scale=1920:1080,split[r1][r2];[1:v]scale=1920:1080[o];"
             "[r2][o]blend=all_mode=difference,lutrgb=r='min(val*8,255)':g='min(val*8,255)':b='min(val*8,255)',"
             "scale=940:528[d];[r1]scale=940:528[l];"
             "[bg][l]overlay=13:200[a];[a][d]overlay=W-w-13:200[base]",
        ov=dict(labels=[(40, 130, "RAW LTX OUTPUT", WHITE), (W - 560, 130, "WHAT IT CHANGED  (x8)", PINK)],
                caption="AI edits repaint everything. Even what they shouldn't.")))

    # 3. solution: comp vs x-ray
    s3 = "mothership" if "mothership" in m else hero[0]
    segs.append(dict(
        text=("PlateLock fixes this like a VFX compositor would. LTX two point three, fully open weights, "
              "invents the effect. Then PlateLock's custom nodes undo the model's colour drift, find exactly "
              "what it changed with a noise-adaptive, space-time difference matte, stabilise that matte so it "
              "never flickers, and composite only the effect onto your full-resolution original, "
              "re-grained to match your camera."),
        kind="video", inputs=[f(s3, "comp"), f(s3, "xray")],
        body=f"color=c=0x0a0a0f:s={W}x{H}[bg];{fit(0, 940, 528)}[l];{fit(1, 940, 528)}[r];"
             "[bg][l]overlay=13:200[a];[a][r]overlay=W-w-13:200[base]",
        ov=dict(labels=[(40, 130, "PLATELOCK COMP", GOLD), (W - 520, 130, "WHAT PLATELOCK KEPT", PINK)],
                caption="Only the pink area comes from the model. The rest is your shot.")))

    # 4. pipeline card
    segs.append(dict(
        text="Eleven open-source ComfyUI nodes, running inside a single LTX workflow.",
        kind="card",
        ov=dict(bg=True, logo=True, title="comfyui-platelock",
                sub="Prep · Difference Matte · Temporal Stabilize · Composite\nExtract Element · Fidelity Report · X-Ray · Share")))

    # 5. proof
    segs.append(dict(
        text=(f"And it proves it. A fidelity report node measures every shot. Across our gallery, "
              f"about {avg_pct:.0f} percent of each frame comes back bit-identical to the original. "
              f"Lossless, instead of {min_db:.0f} decibels."),
        kind="card",
        ov=dict(bg=True, stats=[(f"{avg_pct:.1f}%", GOLD, "bit-identical\nto your footage"),
                                (f"{min_db:.0f}-{max_db:.0f} dB", PINK, "raw model outside\nthe effect"),
                                ("∞ dB", GOLD, "PlateLock outside\nthe effect")])))

    # 6. app
    app_text = ("The app runs on the Comfy SDK. Drop a clip, tap an effect, and compare the original, "
                "raw LTX, PlateLock and the X-ray matte, with the fidelity numbers for every shot. "
                "Or bring a generation from any other tool, and PlateLock rescues your plate.")
    if screen:
        segs.append(dict(text=app_text, kind="video", inputs=[screen], body=f"{fit(0, W, H)}[base]",
                         ov=dict(caption="The PlateLock app · built on the Comfy SDK")))
    else:
        segs.append(dict(text=app_text, kind="pan", image=str(ROOT / ".runtime" / "app_full.png"),
                         y0=0, y1=2100, ov=dict(caption="The PlateLock app · built on the Comfy SDK", bug=False)))

    # 7. outputs: element + share
    s7 = "energy-orb" if "energy-orb" in m else hero[0]
    segs.append(dict(
        text=("You get a vertical before-and-after for your feed, and the effect as its own layer, "
              "with alpha, ready for Nuke, After Effects or Resolve."),
        kind="video", inputs=[f(s7, "share"), f(s7, "element"), f(s7, "comp")],
        body=f"color=c=0x0a0a0f:s={W}x{H}[bg];[0:v]scale=-2:960[a0];{fit(1, 1100, 618)}[a1];"
             "[bg][a0]overlay=120:40[x];[x][a1]overlay=W-w-120:150[base]",
        ov=dict(labels=[(120, 30, "9:16 SHARE", GOLD), (W - 1220, 140, "VFX ELEMENT LAYER", GOLD)],
                caption="Made to share. Made to composite.")))

    # 8. gallery montage
    mont = [x for x in ["code-on-fire", "meteor", "sea-serpent", "lightning-road", "fireflies", "hologram"] if x in m][:6]
    segs.append(dict(
        text=("Generative effects that finally respect the plate. Open weights end to end. "
              "Even this voice is Kokoro, an open-weights model."),
        kind="video", inputs=[f(s, "comp") for s in mont],
        body=montage_body(len(mont)),
        ov=dict(caption="100% open weights · LTX-2.3 · ComfyUI · MIT")))

    # 9. close
    segs.append(dict(
        text="PlateLock. Add the VFX. Keep the shot.",
        kind="card", ov=dict(bg=True, lockup=True, sub="Add the VFX. Keep the shot.\n#OpenSourceHiggsfield")))

    parts, starts, durs, total = [], [], [], 0.0
    for i, s in enumerate(segs):
        wav = WORK / f"n{i}.wav"
        dur = tts(s["text"], wav) + 0.5
        starts.append(total)
        durs.append(dur)
        total += dur
        ovp = WORK / f"o{i}.png"
        overlay_png(ovp, **s["ov"])
        vid = WORK / f"v{i}.mp4"
        if s["kind"] == "card":
            seg_card(vid, dur, ovp)
        elif s["kind"] == "pan":
            seg_pan(vid, dur, s["image"], s["y0"], s["y1"], ovp)
        else:
            seg_video(vid, dur, s["inputs"], s["body"], ovp)
        parts.append(vid)
        print(f"segment {i}: {dur:.1f}s  (total {total:.1f}s)")
    lst = WORK / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    silent = WORK / "video.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)])

    audio = mix_audio(segs, starts, durs, total)
    final = ROOT / "demo" / "platelock_demo.mp4"
    final.parent.mkdir(exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(audio), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(final)])
    print(f"wrote {final}  ({total:.1f}s)")
    if total > 120:
        print("WARNING: over 2 minutes - tighten narration")


# ------------------------------------------------------------------ sound design
SR = 48000
AUDIO = ROOT / ".runtime" / "audio"   # Mixkit music + SFX (free license), fetched by scripts/fetch_audio.sh


def load(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                                   "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32).copy()


def place(track, clip, at, gain=1.0):
    a = max(0, int(at * SR))
    clip = clip[max(0, -int(at * SR)):]
    n = min(len(clip), len(track) - a)
    if n > 0:
        track[a:a + n] += clip[:n] * gain


def mix_audio(segs, starts, durs, total):
    """Narration + SFX built sample-accurately; music bed ducked under the voice; -14 LUFS master."""
    import soundfile as sf
    n = int(total * SR) + SR
    voice, sfx = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i in range(len(segs)):
        place(voice, load(WORK / f"n{i}.wav"), starts[i], 2.0)
    fx = {k: load(AUDIO / f"sfx_{k}.mp3") for k in
          ["impact_deep", "whoosh_fast", "whoosh_cine_fast", "sparkle", "logo_impact", "safe_lock"]}
    # opening hit (peak at 0.6s) + a whoosh on each hook wipe (peak 0.7s, wipe crosses mid-frame ~1.6s in)
    place(sfx, fx["impact_deep"], 0.0, 1.0)
    for k in range(3):
        place(sfx, fx["whoosh_fast"], k * 3.6 + 0.9, 1.6)
    # cinematic whoosh on every cut (peak at 1.05s lands on the cut)
    for i in range(1, len(segs) - 1):
        place(sfx, fx["whoosh_cine_fast"], starts[i] - 1.05, 1.4)
    # sparkle on the proof stats
    for i, s in enumerate(segs):
        if s["ov"].get("stats"):
            place(sfx, fx["sparkle"], starts[i] + 0.2, 1.0)
    # logo end card: trailer impact rising into the cut (peak 4.6s), then the lock clicks shut
    last = starts[-1]
    place(sfx, fx["logo_impact"], last - 4.6, 0.8)
    place(sfx, fx["safe_lock"], last + 0.05, 3.0)
    place(sfx, fx["sparkle"], last + 0.35, 1.2)
    sfx *= 0.22  # effects accent the edit; the voice stays on top
    sf.write(WORK / "voice.wav", voice, SR)
    sf.write(WORK / "sfx.wav", sfx, SR)

    music = AUDIO / "music_main.mp3"
    out = WORK / "mix.wav"
    fade_out = max(0.0, total - 4)
    fc = (f"[2:a]aresample={SR},aformat=channel_layouts=mono,volume=0.13,atrim=0:{total:.3f},"
          f"afade=t=in:d=1.5,afade=t=out:st={fade_out:.2f}:d=4[m];"
          "[0:a]asplit[v1][v2];"
          "[m][v1]sidechaincompress=threshold=0.015:ratio=8:attack=15:release=450:makeup=1[md];"
          "[md][v2][1:a]amix=inputs=3:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9[a]")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(WORK / "voice.wav"), "-i", str(WORK / "sfx.wav"),
         "-i", str(music), "-filter_complex", fc, "-map", "[a]", "-ar", str(SR), "-ac", "2", str(out)])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=None, help="screen recording of the app")
    build(ap.parse_args().screen)
