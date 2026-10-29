// PlateLock frontend: studio, compare viewer, gallery. Works against the
// FastAPI server, or statically (GitHub Pages) with examples/manifest.json.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = { mode: "fx", preset: "orb", presets: [], plate: null, gen: null, online: false,
                files: null, left: "original", right: "comp", zoom: 1 };

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    const r = await fetch("api/info");
    if (!r.ok) throw 0;
    const info = await r.json();
    state.online = true;
    state.presets = info.presets;
    $$(".maxs").forEach(e => (e.textContent = info.max_seconds));
    if (!info.fx_available) setMode("lock");
  } catch {
    state.online = false;
    $("#offline-note").classList.remove("hidden");
    state.presets = FALLBACK_PRESETS;
  }
  renderPresets();
  loadGallery();
}

const FALLBACK_PRESETS = [
  ["ablaze", "🔥", "Set It Ablaze", "Real flames that follow your shot's motion."],
  ["lightning", "⚡", "Lightning Hands", "Crackling electric arcs, superhero-grade."],
  ["orb", "🔮", "Energy Orb", "A glowing orb of power forms and pulses."],
  ["portal", "🌀", "Open a Portal", "A swirling portal tears open in your world."],
  ["meteor", "☄️", "Meteor Strike", "A flaming meteor streaks in and impacts."],
  ["mothership", "🛸", "Mothership", "A colossal alien ship looms overhead."],
  ["dragon", "🐉", "Pocket Dragon", "A tiny dragon crawls out and says hi."],
  ["freeze", "❄️", "Flash Freeze", "Frost creeps over everything it touches."],
  ["custom", "✍️", "Write Your Own", "Describe any effect. PlateLock guards the rest."],
].map(([id, emoji, name, blurb]) => ({ id, emoji, name, blurb, prompt: "{subject}", subject_default: "", sensitivity: 0.5 }));

// ---------------------------------------------------------------- studio
function setMode(m) {
  state.mode = m;
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === m));
  $$(".mode-fx").forEach(e => e.classList.toggle("hidden", m !== "fx"));
  $$(".mode-lock").forEach(e => e.classList.toggle("hidden", m !== "lock"));
  refreshGo();
}
$$(".tab").forEach(t => t.addEventListener("click", () => setMode(t.dataset.mode)));

function renderPresets() {
  const box = $("#presets");
  box.innerHTML = "";
  for (const p of state.presets) {
    const b = document.createElement("button");
    b.className = "preset" + (p.id === state.preset ? " on" : "");
    b.innerHTML = `<div class="e">${p.emoji}</div><div class="t">${p.name}</div><div class="b">${p.blurb}</div>`;
    b.onclick = () => {
      state.preset = p.id;
      $("#sens").value = p.sensitivity; $("#sens-out").textContent = (+p.sensitivity).toFixed(2);
      $("#subject").placeholder = p.id === "custom" ? "Describe the effect…" : `Where? default: ${p.subject_default}`;
      renderPresets(); previewPrompt();
    };
    box.appendChild(b);
  }
  previewPrompt();
}
function previewPrompt() {
  const p = state.presets.find(x => x.id === state.preset);
  if (!p) return;
  const subj = $("#subject").value.trim() || p.subject_default;
  $("#prompt-preview").textContent = "→ " + p.prompt.replace("{subject}", subj);
}
$("#subject").addEventListener("input", previewPrompt);
$("#sens").addEventListener("input", e => ($("#sens-out").textContent = (+e.target.value).toFixed(2)));

function wireDrop(labelSel, inputSel, videoSel, key) {
  const label = $(labelSel), input = $(inputSel), video = $(videoSel);
  const take = f => {
    if (!f) return;
    state[key] = f;
    video.src = URL.createObjectURL(f);
    video.classList.remove("hidden");
    video.play();
    refreshGo();
  };
  input.addEventListener("change", () => take(input.files[0]));
  label.addEventListener("dragover", e => { e.preventDefault(); label.classList.add("over"); });
  label.addEventListener("dragleave", () => label.classList.remove("over"));
  label.addEventListener("drop", e => { e.preventDefault(); label.classList.remove("over"); take(e.dataTransfer.files[0]); });
}
wireDrop("#drop-plate", "#file-plate", "#prev-plate", "plate");
wireDrop("#drop-gen", "#file-gen", "#prev-gen", "gen");

function refreshGo() {
  const ok = state.online && state.plate && (state.mode === "fx" || state.gen);
  $("#go").disabled = !ok;
}

$("#go").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("plate", state.plate);
  if (state.mode === "lock") fd.append("generated", state.gen);
  fd.append("preset", state.preset);
  fd.append("subject", $("#subject").value);
  fd.append("sensitivity", $("#sens").value);
  fd.append("element_mode", $("#element").value);
  fd.append("seed", String(Math.floor(Math.random() * 1e6)));
  $("#go").disabled = true;
  $("#status").classList.remove("hidden");
  setStatus(0.02, "Uploading your shot…");
  try {
    const r = await fetch("api/jobs", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { id } = await r.json();
    poll(id);
  } catch (e) {
    setStatus(0, "⚠️ " + e.message);
    refreshGo();
  }
});

function setStatus(p, msg) {
  $("#bar").style.width = Math.max(2, p * 100) + "%";
  $("#msg").textContent = msg;
}

async function poll(id) {
  const j = await (await fetch(`api/jobs/${id}`)).json();
  if (j.status === "done") {
    setStatus(1, `${j.message} (${j.seconds}s)`);
    refreshGo();
    showResult(j.files, j.report, j.prompt || "Locked generation");
    return;
  }
  if (j.status === "error") { setStatus(0, "⚠️ " + j.message); refreshGo(); return; }
  setStatus(j.progress || 0.05, j.message);
  setTimeout(() => poll(id), 1200);
}

// ---------------------------------------------------------------- result viewer
const LABELS = { original: "ORIGINAL", raw: "RAW LTX", comp: "PLATELOCK", xray: "X-RAY MATTE", element: "VFX ELEMENT" };

function showResult(files, report, title, base = "", instant = false) {
  state.files = Object.fromEntries(Object.entries(files).map(([k, v]) => [k, base + v]));
  $("#result").classList.remove("hidden");
  $("#result-title").textContent = title;
  const hasOrig = !!state.files.original;
  $$('#left-seg button[data-left="original"]').forEach(b => b.classList.toggle("hidden", !hasOrig));
  setLeft(hasOrig ? "original" : "raw"); setRight("comp");
  renderFidelity(report);
  renderDownloads();
  if (!location.search.includes("noscroll")) $("#result").scrollIntoView({ behavior: instant ? "auto" : "smooth", block: "start" });
}

function syncPlay() {
  const L = $("#v-left"), R = $("#v-right");
  const go = () => { L.currentTime = R.currentTime = 0; L.play(); R.play(); };
  let n = 0;
  [L, R].forEach(v => (v.oncanplay = () => { if (++n === 2) go(); }));
}
function setLeft(k) {
  if (!state.files?.[k]) return;
  state.left = k;
  $("#v-left").poster = state.files[k].replace(/\.mp4$/, ".jpg");
  $("#v-left").src = state.files[k];
  $("#tag-l").textContent = LABELS[k];
  $$("#left-seg button").forEach(b => b.classList.toggle("on", b.dataset.left === k));
  syncPlay(); $("#v-right").load();
}
function setRight(k) {
  if (!state.files?.[k]) return;
  state.right = k;
  $("#v-right").poster = state.files[k].replace(/\.mp4$/, ".jpg");
  $("#v-right").src = state.files[k];
  $("#tag-r").textContent = LABELS[k];
  $$("#right-seg button").forEach(b => b.classList.toggle("on", b.dataset.right === k));
  syncPlay(); $("#v-left").load();
}
$$("#left-seg button").forEach(b => (b.onclick = () => setLeft(b.dataset.left)));
$$("#right-seg button").forEach(b => (b.onclick = () => setRight(b.dataset.right)));

// keep the two videos frame-locked
setInterval(() => {
  const L = $("#v-left"), R = $("#v-right");
  if (!L.paused && Math.abs(L.currentTime - R.currentTime) > 0.06) L.currentTime = R.currentTime;
}, 250);

// drag divider
(() => {
  const box = $("#compare");
  let drag = false;
  const set = x => {
    const r = box.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, (x - r.left) / r.width));
    $("#v-left").style.clipPath = `inset(0 ${100 - p * 100}% 0 0)`;
    $("#handle").style.left = p * 100 + "%";
  };
  box.addEventListener("pointerdown", e => { drag = true; box.setPointerCapture(e.pointerId); set(e.clientX); });
  box.addEventListener("pointermove", e => drag && set(e.clientX));
  box.addEventListener("pointerup", () => (drag = false));
})();

$("#play").onclick = () => {
  const L = $("#v-left"), R = $("#v-right");
  if (R.paused) { L.play(); R.play(); } else { L.pause(); R.pause(); }
};
$("#zoom").onclick = () => {
  state.zoom = state.zoom === 1 ? 2 : state.zoom === 2 ? 3 : 1;
  $$("#compare video").forEach(v => (v.style.transform = `scale(${state.zoom})`));
  $("#zoom").textContent = `🔍 ${state.zoom}×`;
};

function renderFidelity(r) {
  const box = $("#fidelity");
  if (!r) { box.innerHTML = ""; return; }
  const pl = r.platelock_psnr_outside_effect_db;
  const lossless = pl === "inf" || pl === Infinity;
  box.innerHTML = `
    <div class="stat"><div class="big gold">${(+r.bit_identical_pct).toFixed(1)}%</div>
      <div class="lbl">of your shot is <b>bit-identical</b> to the original (${r.resolution}, ${r.frames} frames)</div></div>
    <div class="stat"><div class="big pink">${(+r.raw_model_psnr_outside_effect_db).toFixed(1)} dB</div>
      <div class="lbl">raw model fidelity <b>outside</b> the effect — what you'd get without PlateLock</div></div>
    <div class="stat"><div class="big gold">${lossless ? "∞ dB" : (+pl).toFixed(1) + " dB"}</div>
      <div class="lbl">PlateLock fidelity outside the effect${lossless ? " — <b>lossless</b>" : ""}</div></div>`;
}

function renderDownloads() {
  const f = state.files, names = { comp: "Final comp", share: "9:16 share clip", element: "VFX element", xray: "X-ray matte", raw: "Raw LTX output" };
  $("#downloads").innerHTML = Object.entries(names).filter(([k]) => f[k])
    .map(([k, n]) => `<a href="${f[k]}" download>⬇ ${n}</a>`).join("");
}

// ---------------------------------------------------------------- gallery
async function loadGallery() {
  const grid = $("#gallery-grid");
  let items = [];
  for (const url of ["api/examples", "examples/manifest.json", "../examples/manifest.json"]) {
    try { const r = await fetch(url); if (r.ok) { items = await r.json(); if (items.length) { items._base = url.replace(/(api\/examples|manifest\.json)$/, ""); break; } } } catch {}
  }
  if (!items.length) { grid.innerHTML = `<p class="muted">No examples yet.</p>`; return; }
  const base = items._base === "" ? "examples/" : items._base;
  grid.innerHTML = "";
  for (const it of items) {
    const d = document.createElement("div");
    d.className = "g-item";
    d.innerHTML = `<video src="${base}${it.files.comp}" poster="${base}${it.slug}/poster.jpg" muted loop playsinline autoplay preload="metadata"></video>
      <div class="cap"><span class="pct">${(+it.report.bit_identical_pct).toFixed(1)}% locked</span>
      <b>${it.emoji || ""} ${it.title}</b><span>${it.subtitle || ""}</span></div>`;
    const open = (instant = false) => {
      showResult(it.files, it.report, `${it.emoji || ""} ${it.title}`, base, instant);
      $("#result-prompt").textContent = it.prompt ? `prompt: “${it.prompt}”` : "";
      history.replaceState(null, "", `#ex=${it.slug}`);
    };
    d.onclick = () => open();
    if (location.hash === `#ex=${it.slug}`) setTimeout(() => open(true), 300);
    grid.appendChild(d);
  }
}

boot();
