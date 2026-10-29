"""One-tap effect presets, written in the format the LTX-2.3 EditAnything LoRA
was trained on ("Add a/an [subject] with [attributes], [location]."). LTX
repaints the whole frame; PlateLock then keeps only what actually changed."""

PRESETS = [
    {
        "id": "ablaze", "emoji": "🔥", "name": "Set It Ablaze",
        "blurb": "Real flames that follow your shot's motion.",
        "prompt": "Add roaring orange flames with drifting embers, burning on {subject}.",
        "subject_default": "the object in the center of the frame",
        "sensitivity": 0.55, "element": "emissive",
    },
    {
        "id": "lightning", "emoji": "⚡", "name": "Lightning Hands",
        "blurb": "Crackling electric arcs, superhero-grade.",
        "prompt": "Add bright blue electric lightning arcs with a glowing white core, crackling around {subject}.",
        "subject_default": "the hand",
        "sensitivity": 0.6, "element": "emissive",
    },
    {
        "id": "orb", "emoji": "🔮", "name": "Energy Orb",
        "blurb": "A glowing orb of power forms and pulses.",
        "prompt": "Add a glowing golden orb of swirling energy with crackling sparks, floating just above {subject}.",
        "subject_default": "the open palm",
        "sensitivity": 0.55, "element": "emissive",
    },
    {
        "id": "portal", "emoji": "🌀", "name": "Open a Portal",
        "blurb": "A swirling portal tears open in your world.",
        "prompt": "Add a swirling glowing blue portal with violet sparks at its edge, opening in {subject}.",
        "subject_default": "the sky",
        "sensitivity": 0.5, "element": "alpha",
    },
    {
        "id": "meteor", "emoji": "☄️", "name": "Meteor Strike",
        "blurb": "A flaming meteor streaks in and impacts.",
        "prompt": "Add a flaming meteor with a bright fiery trail and smoke, crashing onto {subject}.",
        "subject_default": "the road in the distance",
        "sensitivity": 0.5, "element": "alpha",
    },
    {
        "id": "mothership", "emoji": "🛸", "name": "Mothership",
        "blurb": "A colossal alien ship looms overhead.",
        "prompt": "Add a colossal dark alien mothership with glowing blue lights, hovering in the sky above {subject}.",
        "subject_default": "the city skyline",
        "sensitivity": 0.45, "element": "alpha",
    },
    {
        "id": "dragon", "emoji": "🐉", "name": "Pocket Dragon",
        "blurb": "A tiny dragon crawls out and says hi.",
        "prompt": "Add a tiny cute green dragon with small wings, climbing out of {subject}.",
        "subject_default": "the phone screen",
        "sensitivity": 0.5, "element": "alpha",
    },
    {
        "id": "freeze", "emoji": "❄️", "name": "Flash Freeze",
        "blurb": "Frost creeps over everything it touches.",
        "prompt": "Add thick white frost and sparkling ice crystals with cold mist, covering {subject}.",
        "subject_default": "the surface in the frame",
        "sensitivity": 0.5, "element": "alpha",
    },
    {
        "id": "custom", "emoji": "✍️", "name": "Write Your Own",
        "blurb": "Describe any effect. PlateLock guards the rest.",
        "prompt": "{subject}",
        "subject_default": "Add glowing magical particles with soft golden light, floating through the scene.",
        "sensitivity": 0.5, "element": "alpha",
    },
]

BY_ID = {p["id"]: p for p in PRESETS}


def build_prompt(preset_id: str, subject: str | None) -> str:
    p = BY_ID.get(preset_id, BY_ID["custom"])
    return p["prompt"].format(subject=(subject or p["subject_default"]).strip())
