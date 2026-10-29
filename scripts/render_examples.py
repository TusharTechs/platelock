"""Batch-render the PlateLock gallery through the Comfy Developer Platform.

    COMFY_BASE_URL=http://127.0.0.1:18189 python scripts/render_examples.py [slug ...]

Each example: plate -> LTX-2.3 EditAnything (open weights) -> PlateLock.
Outputs land in examples/<slug>/ and examples/manifest.json is rebuilt.
"""
import json
import pathlib
import sys
import time

from comfy_sdk import Comfy

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLATES = ROOT / ".runtime" / "plates"
EX = ROOT / "examples"
WF = json.loads((ROOT / "workflows" / "platelock_ltx_api.json").read_text())

SHOTS = [
    # slug, plate, emoji, title, subtitle, prompt, sensitivity, element
    ("energy-orb", "46762", "🔮", "Energy Orb", "Hand at sunset",
     "Add a glowing golden orb of swirling energy with crackling sparks, floating just above the open palm.", 0.55, "alpha"),
    ("code-on-fire", "1781", "🔥", "My Code Is On Fire", "Typing on a laptop",
     "Add roaring orange flames with drifting embers, burning on the laptop keyboard.", 0.55, "emissive"),
    ("pocket-dragon", "4915", "🐉", "Pocket Dragon", "Scrolling a phone",
     "Add a small green dragon with spread wings, standing on top of the phone.", 0.5, "alpha", 7),
    ("mothership", "49806", "🛸", "Mothership", "City aerial",
     "Add a colossal dark alien mothership with glowing blue lights, hovering in the sky above the city skyline.", 0.45, "alpha"),
    ("meteor", "4852", "☄️", "Meteor Strike", "Country road",
     "Add a flaming meteor with a bright fiery trail and smoke, crashing onto the road in the distance.", 0.5, "alpha"),
    ("portal", "4170", "🌀", "Sky Portal", "Looking up at skyscrapers",
     "Add a swirling glowing blue portal with violet sparks at its edge, opening in the sky between the buildings.", 0.5, "alpha"),
    ("sea-serpent", "1218", "🌊", "Sea Serpent", "Mountain lake",
     "Add a giant green sea monster head with a long neck, rising from the lake in the center.", 0.45, "alpha", 7),
    ("fireflies", "50847", "🦌", "Spirit Deer", "Pine forest",
     "Add a glowing blue spirit deer with shining antlers, standing on the forest path in the center.", 0.55, "alpha", 7),
    ("lightning-road", "52427", "💥", "Road Explosion", "Car on a highway",
     "Add a huge orange fireball explosion with smoke, on the road behind the car.", 0.5, "alpha", 7),
    ("hologram", "50111", "🪐", "Desk Hologram", "Studying at a desk",
     "Add a small glowing blue hologram of a rotating planet, floating above the open book.", 0.55, "alpha"),
]

OUT_NAMES = ["comp", "raw", "xray", "element", "share"]


def title_id(title):
    for nid, n in WF.items():
        if n.get("_meta", {}).get("title") == title:
            return nid
    raise KeyError(title)


def render(c, shot, seed=42):
    slug, plate, emoji, title, subtitle, prompt, sens, elem = shot[:8]
    seed = shot[8] if len(shot) > 8 else seed
    out = EX / slug
    out.mkdir(parents=True, exist_ok=True)
    wf = c.workflows.from_json(json.loads(json.dumps(WF)))
    wf.set_input(title_id("Plate (original footage)"), "file", c.assets.from_file(str(PLATES / f"{plate}.mp4")))
    wf.set_input(title_id("Effect Prompt"), "text", prompt)
    wf.set_input(title_id("Seed"), "noise_seed", seed)
    wf.set_input(title_id("Difference Matte"), "sensitivity", sens)
    wf.set_input(title_id("Extract Element"), "mode", elem)
    t0 = time.time()
    job = c.submit(wf)
    print(f"[{slug}] submitted {job.id}", flush=True)
    job.result()
    secs = round(time.time() - t0, 1)
    files = {}
    for name in OUT_NAMES:
        outs = job.get_outputs(title_id(f"Save {name}"))
        if outs:
            outs[0].to_file(out / f"{name}.mp4")
            files[name] = f"{slug}/{name}.mp4"
    report = {}
    for o in job.get_outputs(title_id("Fidelity Report")):
        if o.name.endswith(".json"):
            report = json.loads(o.to_bytes())
    # original plate trimmed to the same frames, for the compare viewer
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(PLATES / f"{plate}.mp4"), "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(out / "original.mp4")], check=True)
    files["original"] = f"{slug}/original.mp4"
    make_poster(out)
    meta = {"slug": slug, "emoji": emoji, "title": title, "subtitle": subtitle, "prompt": prompt,
            "seed": seed, "sensitivity": sens, "seconds": secs, "files": files, "report": report,
            "source": f"Mixkit clip #{plate}"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{slug}] done in {secs}s  report={report}", flush=True)
    return meta


def make_poster(out):
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "comp.mp4"), "-vf",
                    "select=eq(n\\,60),scale=640:-2", "-frames:v", "1", "-q:v", "4",
                    str(out / "poster.jpg")], check=True)


def rebuild_manifest():
    items = []
    for shot in SHOTS:
        m = EX / shot[0] / "meta.json"
        if m.exists():
            items.append(json.loads(m.read_text()))
    (EX / "manifest.json").write_text(json.dumps(items, indent=2))
    print(f"manifest: {len(items)} examples")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    c = Comfy(timeout=300.0)
    for shot in SHOTS:
        if want and shot[0] not in want:
            continue
        try:
            render(c, shot)
        except Exception as e:
            print(f"[{shot[0]}] FAILED: {type(e).__name__}: {e}", flush=True)
        rebuild_manifest()
