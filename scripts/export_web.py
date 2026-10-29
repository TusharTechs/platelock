"""Export the static PlateLock site (GitHub Pages) into docs/.

Copies app/web and web-optimised versions of the gallery outputs. Unmodified
source footage (original.mp4) is NOT exported: Mixkit's license forbids
redistributing clips as-is. Everything exported is a PlateLock derivative.
"""
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX, DOCS = ROOT / "examples", ROOT / "docs"
KEEP = ["comp", "raw", "xray", "element", "share"]


def web_encode(src, dst, width):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale={width}:-2:flags=lanczos",
                    "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p", "-an",
                    "-movflags", "+faststart", str(dst)], check=True)


def main():
    if DOCS.exists():
        shutil.rmtree(DOCS)
    shutil.copytree(ROOT / "app" / "web", DOCS)
    (DOCS / ".nojekyll").write_text("")
    items = json.loads((EX / "manifest.json").read_text())
    out_items = []
    for it in items:
        slug = it["slug"]
        dst = DOCS / "examples" / slug
        dst.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in KEEP:
            src = EX / slug / f"{name}.mp4"
            if src.exists():
                web_encode(src, dst / f"{name}.mp4", 540 if name == "share" else 960)
                files[name] = f"{slug}/{name}.mp4"
        if (EX / slug / "poster.jpg").exists():
            shutil.copy(EX / slug / "poster.jpg", dst / "poster.jpg")
        out_items.append({**it, "files": files})
        print("exported", slug)
    (DOCS / "examples" / "manifest.json").write_text(json.dumps(out_items, indent=2))
    total = sum(f.stat().st_size for f in DOCS.rglob("*") if f.is_file())
    print(f"docs/ = {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
