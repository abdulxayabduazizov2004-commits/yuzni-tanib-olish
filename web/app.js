/* FaceID Vision — frontend */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const initials = (n) => (n || "?").split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
const hue = (s) => [...(s || "")].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 360, 200);
const avatarBg = (name) => `linear-gradient(135deg, hsl(${hue(name)} 70% 45%), hsl(${(hue(name) + 50) % 360} 70% 35%))`;

const api = async (url, opts = {}) => {
  const o = { ...opts };
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers = { "Content-Type": "application/json" }; delete o.json; }
  const r = await fetch(url, o);
  const data = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (r.status === 401 && data?.auth === false && state.me) { location.reload(); throw new Error(data.error); }
  if (!r.ok) throw new Error(data?.error || r.statusText);
  return data;
};

const ATTR_ICON = { glasses: "👓", sunglasses: "🕶️", cap: "🧢", hat: "👒", headscarf: "🧕", beard: "🧔", mustache: "👨", mask: "😷", headphones: "🎧", earrings: "💎", smile: "😊" };
const attrChips = (attrs) => (attrs || []).map((a) => `<span class="attr">${ATTR_ICON[a.key] || "•"} ${esc(a.label)}</span>`).join("");
const state = { meta: null, users: [], page: "camera", evType: "", editPid: null, camOn: false, dets: [], frameSize: [1, 1], photoVer: Date.now() };

/* ---------------- toast ---------------- */
function toast(msg, color = "var(--accent)") {
  const t = document.createElement("div");
  t.className = "toast"; t.style.setProperty("--c", color); t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => { t.style.transition = ".3s"; t.style.opacity = 0; setTimeout(() => t.remove(), 300); }, 3800);
}

/* ---------------- navigation ---------------- */
const can = (page) => !!state.me?.pages.includes(page);
const isAdmin = () => state.me?.role === "admin";
function go(page) {
  if (!can(page)) page = state.me?.pages[0] || "camera";
  state.page = page;
  $$("nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
  if (page === "users") loadUsers();
  if (page === "dashboard") loadDashboard();
  if (page === "settings") loadSettings();
  if (page === "accounts") loadAccounts();
}
$$("nav a").forEach((a) => a.addEventListener("click", () => go(a.dataset.page)));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => { clearForm(); go(b.dataset.goto); }));

/* ---------------- modal ---------------- */
function openModal(html) { $("#modal").innerHTML = html; $("#modal-bg").classList.remove("hidden"); }
function closeModal() { $("#modal-bg").classList.add("hidden"); }
$("#modal-bg").addEventListener("click", (e) => { if (e.target.id === "modal-bg" || e.target.closest(".close")) closeModal(); });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#modal-bg").classList.contains("hidden")) closeModal(); else cancelReg();
});

/* ================= CAMERA ================= */
const video = $("#video"), overlay = $("#overlay"), octx = overlay.getContext("2d");
const grab = document.createElement("canvas"), gctx = grab.getContext("2d");

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    state.camOn = true;
    $("#cam-placeholder").classList.add("hidden");
    $("#cam-dot").classList.add("on"); $("#cam-state").textContent = "Kamera faol";
    loop();
  } catch (e) {
    toast("Kamera ochilmadi: " + e.message, "var(--bad)");
    $("#cam-placeholder p").textContent = "Ruxsat berilmadi. Brauzer manzil satridagi kamera belgisini bosib ruxsat bering.";
  }
}
$("#btn-start-cam").addEventListener("click", startCamera);

async function loop() {
  if (!state.camOn) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) { requestAnimationFrame(loop); return; }
  const W = Math.min(960, vw), H = Math.round(vh * W / vw);
  grab.width = W; grab.height = H;
  gctx.save(); gctx.translate(W, 0); gctx.scale(-1, 1); gctx.drawImage(video, 0, 0, W, H); gctx.restore();
  try {
    const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.82));
    const res = await api("/api/frame", { method: "POST", body: blob });
    state.dets = res.detections; state.frameSize = res.size;
    drawOverlay(); renderInFrame(); renderReg(res.registration);
    $("#fps").textContent = `${res.fps.toFixed(1)} kadr/s`;
    res.events.forEach(addLog);
    if (res.registration_done) {
      toast(`${res.registration_done.name} muvaffaqiyatli ro'yxatdan o'tdi ✓`, "var(--ok)");
      state.photoVer = Date.now(); refreshCount();
    }
  } catch (e) { await new Promise((r) => setTimeout(r, 500)); }
  requestAnimationFrame(loop);
}

function roundRect(c, x, y, w, h, r) { c.beginPath(); c.roundRect ? c.roundRect(x, y, w, h, r) : c.rect(x, y, w, h); }
function corners(c, x, y, w, h, color, lw) {
  const L = Math.min(w, h) * 0.22;
  c.strokeStyle = color; c.lineWidth = lw; c.lineCap = "round";
  c.beginPath();
  c.moveTo(x, y + L); c.lineTo(x, y); c.lineTo(x + L, y);
  c.moveTo(x + w - L, y); c.lineTo(x + w, y); c.lineTo(x + w, y + L);
  c.moveTo(x + w, y + h - L); c.lineTo(x + w, y + h); c.lineTo(x + w - L, y + h);
  c.moveTo(x + L, y + h); c.lineTo(x, y + h); c.lineTo(x, y + h - L);
  c.stroke();
}
function drawOverlay() {
  const rect = overlay.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  overlay.width = rect.width * dpr; overlay.height = rect.height * dpr;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0); octx.clearRect(0, 0, rect.width, rect.height);
  const [fw, fh] = state.frameSize;
  const k = Math.min(rect.width / fw, rect.height / fh), ox = (rect.width - fw * k) / 2, oy = (rect.height - fh * k) / 2;
  const T = (b) => [ox + b[0] * k, oy + b[1] * k, b[2] * k, b[3] * k];
  for (const d of state.dets) {
    const reg = d.state === "reg";
    const color = reg ? "#f59e0b" : d.kind === "body" ? "#38bdf8" : d.pid ? "#22c55e" : "#f43f5e";
    if (d.body && !reg && d.kind === "face") {
      const [bx, by, bw, bh] = T(d.body);
      octx.setLineDash([6, 6]); octx.strokeStyle = color + "66"; octx.lineWidth = 1.5;
      roundRect(octx, bx, by, bw, bh, 10); octx.stroke(); octx.setLineDash([]);
    }
    const [x, y, w, h] = T(d.box);
    octx.fillStyle = color + "14"; roundRect(octx, x, y, w, h, 12); octx.fill();
    corners(octx, x, y, w, h, color, 3);
    const label = reg ? "Ro'yxatga olinmoqda" : d.pid ? d.name : "Noma'lum";
    let sub = reg ? "" : `${Math.round(Math.max(d.score, 0) * 100)}% ishonch` + (d.kind === "body" ? " · tana" : d.via === "yuz+tana" ? " · yuz+tana" : "");
    if (d.state === "learned") sub += " · o'rgandi ✦";
    // ism - katta shriftda (ramka kengligiga qarab 20..34px)
    const fs = Math.round(Math.max(20, Math.min(34, w * 0.13)));
    const NF = `800 ${fs}px Inter, -apple-system, sans-serif`, SF = "500 13px Inter, sans-serif";
    octx.font = NF; const nw = octx.measureText(label).width;
    octx.font = SF; const sw = octx.measureText(sub).width;
    const ph = fs + (sub ? 30 : 16), pw = Math.max(nw, sw) + 28;
    const py = y - ph - 10 > 0 ? y - ph - 10 : y + h + 10;
    octx.fillStyle = "rgba(10,12,20,.86)"; roundRect(octx, x, py, pw, ph, 12); octx.fill();
    octx.fillStyle = color; roundRect(octx, x, py, 5, ph, 3); octx.fill();
    octx.fillStyle = "#fff"; octx.font = NF; octx.textBaseline = "top"; octx.fillText(label, x + 15, py + 8);
    if (sub) { octx.fillStyle = "#b4bdd6"; octx.font = SF; octx.fillText(sub, x + 15, py + fs + 12); }
    // atributlar (ko'zoynak, kepka, soqol...) - ramka ostida
    if (d.attrs?.length && !reg) {
      octx.font = "600 13px Inter, -apple-system, sans-serif";
      let cx = x, cy = py > y ? py + ph + 8 : y + h + 10;
      for (const a of d.attrs) {
        const t = `${ATTR_ICON[a.key] || "•"} ${a.label}`, tw = octx.measureText(t).width + 18;
        if (cx + tw > x + Math.max(w, 260)) { cx = x; cy += 30; }
        octx.fillStyle = "rgba(124,92,255,.85)"; roundRect(octx, cx, cy, tw, 25, 13); octx.fill();
        octx.fillStyle = "#fff"; octx.fillText(t, cx + 9, cy + 6);
        cx += tw + 6;
      }
    }
    octx.textBaseline = "alphabetic";
  }
}
window.addEventListener("resize", drawOverlay);

function renderInFrame() {
  const ds = state.dets.filter((d) => d.state !== "reg");
  $("#inframe-count").textContent = ds.length;
  $("#inframe").innerHTML = ds.length ? ds.map((d) => {
    const n = d.pid ? d.name : "Noma'lum";
    const av = d.pid ? `<div class="avatar" style="background-image:url('/api/users/${d.pid}/photo?v=${state.photoVer}'),${avatarBg(n)}"></div>`
                     : `<div class="avatar" style="background:rgba(244,63,94,.18);color:#fda4af">?</div>`;
    const tag = !d.pid ? `<span class="tag bad">noma'lum</span>` : d.kind === "body" ? `<span class="tag body">tana</span>` : `<span class="tag">${esc(d.via || "yuz")}</span>`;
    return `<div class="row">${av}<div class="meta"><b class="big-name">${esc(n)}</b><div class="bar"><i style="width:${Math.max(0, Math.min(100, d.score * 100))}%"></i></div>${d.attrs?.length ? `<div class="attrs">${attrChips(d.attrs)}</div>` : ""}</div>${tag}</div>`;
  }).join("") : `<div class="empty">Hech kim yo'q</div>`;
}

function addLog(line) {
  const t = line.slice(0, 8), m = line.slice(9);
  const cls = /^Aniqlandi|^Tayyor/.test(m) ? "ok" : /^O'rgandi/.test(m) ? "learn" : /Noma'lum|Xato|Diqqat|bekor/.test(m) ? "bad" : "";
  const el = document.createElement("div"); el.className = cls;
  el.innerHTML = `<time>${esc(t)}</time>${esc(m)}`;
  $("#log").prepend(el);
  while ($("#log").children.length > 200) $("#log").lastChild.remove();
  if (cls === "learn") toast(m, "var(--accent)");
  else if (/^Aniqlandi/.test(m)) toast(m, "var(--ok)");
  else if (/Diqqat|Noma'lum shaxs/.test(m)) toast(m, "var(--warn)");
}
$("#btn-clear-log").addEventListener("click", () => ($("#log").innerHTML = ""));

function renderReg(r) {
  const ov = $("#reg-overlay");
  if (!r) { ov.classList.add("hidden"); return; }
  ov.classList.remove("hidden");
  const pct = r.collected / r.total;
  $("#reg-ring").style.strokeDashoffset = 326.7 * (1 - pct);
  $("#reg-pct").textContent = Math.round(pct * 100) + "%";
  $("#reg-name").textContent = (r.existing ? "Qo'shimcha o'rgatish · " : "Ro'yxatdan o'tkazish · ") + r.name + ` · qadam ${r.step}/${r.steps}`;
  $("#reg-msg").textContent = r.message;
  $("#reg-steps").innerHTML = Array.from({ length: r.steps }, (_, i) => `<i class="${i + 1 < r.step ? "done" : i + 1 === r.step ? "cur" : ""}"></i>`).join("");
}
async function cancelReg() { await api("/api/registration/cancel", { method: "POST" }); }
$("#btn-reg-cancel").addEventListener("click", cancelReg);

/* switches */
async function setSetting(key, val) { await api("/api/settings", { method: "PUT", json: { [key]: val } }); }
$("#sw-learn").addEventListener("change", (e) => setSetting("auto_learn", e.target.checked));
$("#sw-body").addEventListener("change", (e) => setSetting("body_enabled", e.target.checked));
$("#sw-attr").addEventListener("change", (e) => setSetting("attributes_enabled", e.target.checked));

/* analyze photo */
$("#btn-analyze").addEventListener("click", () => $("#analyze-file").click());
$("#analyze-file").addEventListener("change", async (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  const fd = new FormData(); fd.append("image", await shrink(f));
  openModal(`<div class="modal-head"><h2>Rasm tahlil qilinmoqda...</h2><div class="spinner"></div></div>`);
  try {
    const r = await api("/api/analyze", { method: "POST", body: fd });
    const list = r.results.length ? r.results.map((x) => `<div class="result-row"><b class="big-name">${esc(x.name)}</b><span class="tag ${x.name === "Noma'lum" ? "bad" : ""}">${Math.round(x.score * 100)}%</span><div class="attrs">${(x.attrs || []).map((a) => `<span class="attr">${esc(a)}</span>`).join("")}</div></div>`).join("") : `<span class="muted">Hech kim topilmadi</span>`;
    openModal(`<div class="modal-head"><h2>Rasmdan aniqlash natijasi</h2><button class="close">×</button></div>
      <img class="result-img" src="${r.image}"><div class="actions">${list}</div>`);
  } catch (err) { closeModal(); toast(err.message, "var(--bad)"); }
});

/* ================= REGISTRATION ================= */
function buildForm() {
  $("#reg-form").innerHTML = state.meta.fields.map((f) => {
    const full = ["notes"].includes(f.key) ? " full" : "";
    const req = f.required ? " <em>*</em>" : "";
    let input;
    if (f.key === "gender") input = `<select name="gender"><option value="">Tanlang</option><option>Erkak</option><option>Ayol</option></select>`;
    else if (f.key === "notes") input = `<textarea name="notes" placeholder="Qo'shimcha ma'lumot"></textarea>`;
    else if (f.key === "birth_date") input = `<input type="date" name="birth_date">`;
    else if (f.key === "email") input = `<input type="email" name="email" placeholder="ism@misol.uz">`;
    else if (f.key === "phone") input = `<input type="tel" name="phone" placeholder="+998 90 123 45 67">`;
    else input = `<input name="${f.key}" ${f.required ? "required" : ""}>`;
    return `<div class="field${full}"><label>${esc(f.label.replace(/ \(YYYY-MM-DD\)/, ""))}${req}</label>${input}</div>`;
  }).join("");
}
const formData = () => Object.fromEntries(new FormData($("#reg-form")).entries());
function validForm() {
  let ok = true;
  state.meta.fields.filter((f) => f.required).forEach((f) => {
    const el = $(`[name=${f.key}]`, $("#reg-form")); const bad = !el.value.trim();
    el.classList.toggle("invalid", bad); if (bad) ok = false;
  });
  if (!ok) toast("Majburiy maydonlarni to'ldiring", "var(--bad)");
  return ok;
}
function clearForm() {
  $("#reg-form").reset(); $$(".invalid").forEach((e) => e.classList.remove("invalid"));
  state.editPid = null; $("#reg-title").textContent = "Yangi foydalanuvchi";
  $("#opt-camera").classList.remove("hidden"); $("#opt-images").classList.remove("hidden"); $("#btn-save-edit").classList.add("hidden");
}
function fillForm(u) {
  clearForm();
  state.meta.fields.forEach((f) => { const el = $(`[name=${f.key}]`, $("#reg-form")); if (el) el.value = u[f.key] || ""; });
  state.editPid = u.id; $("#reg-title").textContent = "Tahrirlash: " + u.name;
  $("#opt-camera").classList.add("hidden"); $("#opt-images").classList.add("hidden"); $("#btn-save-edit").classList.remove("hidden");
  go("register");
}
$("#btn-clear-form").addEventListener("click", clearForm);
$("#btn-save-edit").addEventListener("click", async () => {
  if (!validForm()) return;
  try { await api(`/api/users/${state.editPid}`, { method: "PUT", json: formData() }); toast("Saqlandi", "var(--ok)"); clearForm(); go("users"); }
  catch (e) { toast(e.message, "var(--bad)"); }
});
$("#opt-camera").addEventListener("click", async () => {
  if (!validForm()) return;
  try {
    await api("/api/registration/start", { method: "POST", json: { profile: formData() } });
    clearForm(); go("camera"); if (!state.camOn) startCamera();
  } catch (e) { toast(e.message, "var(--bad)"); }
});
const optImg = $("#opt-images");
optImg.addEventListener("click", (e) => { if (!validForm()) e.preventDefault(); });
["dragover", "dragenter"].forEach((ev) => optImg.addEventListener(ev, (e) => { e.preventDefault(); optImg.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => optImg.addEventListener(ev, () => optImg.classList.remove("drag")));
optImg.addEventListener("drop", (e) => { e.preventDefault(); if (validForm()) registerWithFiles([...e.dataTransfer.files]); });
$("#reg-files").addEventListener("change", (e) => { const fs = [...e.target.files]; e.target.value = ""; if (fs.length) registerWithFiles(fs); });

async function registerWithFiles(files) {
  try {
    const u = await api("/api/users", { method: "POST", json: formData() });
    clearForm();
    await uploadImages(u.id, files, true, u.name);
  } catch (e) { toast(e.message, "var(--bad)"); }
}
// Katta (telefon) rasmlarni brauzerda kichraytiradi: tezroq yuklanadi va proksi cheklovlariga sig'adi
async function shrink(file, max = 1600) {
  try {
    const bmp = await createImageBitmap(file);
    const k = Math.min(1, max / Math.max(bmp.width, bmp.height));
    if (k === 1 && file.size < 1.5e6) return file;
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
    return new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
  } catch { return file; }
}
async function uploadImages(pid, files, deleteIfEmpty, name) {
  toast(`${files.length} ta rasm qayta ishlanmoqda...`);
  const BATCH = 4;  // proksi orqali kichik guruhlarda yuboriladi
  const r = { added: 0, skipped: 0, deleted: false };
  for (let i = 0; i < files.length; i += BATCH) {
    const fd = new FormData();
    for (const f of files.slice(i, i + BATCH)) fd.append("images", await shrink(f));
    const q = deleteIfEmpty && i + BATCH >= files.length ? "?delete_if_empty=1" : "";
    const x = await api(`/api/users/${pid}/images${q}`, { method: "POST", body: fd });
    r.added += x.added; r.skipped += x.skipped; r.deleted = !!x.deleted;
  }
  if (r.deleted) toast(`${name}: rasmlarda yuz topilmadi, saqlanmadi`, "var(--bad)");
  else toast(`${name || ""}: ${r.added} ta rasm o'rganildi${r.skipped ? `, ${r.skipped} ta o'tkazib yuborildi` : ""}`, "var(--ok)");
  state.photoVer = Date.now(); refreshCount();
  if (state.page === "users") loadUsers();
}

/* ================= USERS ================= */
async function refreshCount() { if (!isAdmin()) return; state.users = await api("/api/users"); $("#nav-users").textContent = state.users.length; }
async function loadUsers() { await refreshCount(); renderUsers(); }
function renderUsers() {
  const q = $("#user-search").value.trim().toLowerCase();
  const list = state.users.filter((u) => !q || Object.values(u).join(" ").toLowerCase().includes(q));
  $("#users-sub").textContent = `${state.users.length} ta ro'yxatdan o'tgan foydalanuvchi`;
  $("#users-grid").innerHTML = list.length ? list.map((u) => `
    <div class="card ucard" data-id="${u.id}">
      <div class="cover" style="background-image:url('/api/users/${u.id}/photo?v=${state.photoVer}'),${avatarBg(u.name)}">${u.faces ? "" : initials(u.name)}</div>
      <div class="body"><h3>${esc(u.name)}</h3><div class="sub">${esc([u.position, u.department].filter(Boolean).join(" · ") || "—")}</div>
        <div class="stats"><span class="tag violet">${u.faces} yuz</span><span class="tag body">${u.bodies} tana</span><span class="tag">${u.seen} marta</span></div></div>
    </div>`).join("") : `<div class="card empty">Foydalanuvchi topilmadi</div>`;
  $$(".ucard").forEach((c) => c.addEventListener("click", () => openUser(c.dataset.id)));
}
$("#user-search").addEventListener("input", renderUsers);

function openUser(pid) {
  const u = state.users.find((x) => x.id === pid); if (!u) return;
  const rows = state.meta.fields.filter((f) => !["first_name", "last_name"].includes(f.key) && u[f.key])
    .map((f) => `<dt>${esc(f.label.replace(/ \(.*\)/, ""))}</dt><dd>${esc(u[f.key])}</dd>`).join("");
  openModal(`
    <div class="modal-head"><h2>${esc(u.name)}</h2><button class="close">×</button></div>
    <div class="profile">
      <div class="photo" style="background-image:url('/api/users/${u.id}/photo?v=${state.photoVer}'),${avatarBg(u.name)}"></div>
      <div>
        <div class="stats" style="display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap">
          <span class="tag violet">${u.faces} yuz namunasi</span><span class="tag body">${u.bodies} tana namunasi</span><span class="tag">${u.seen} marta aniqlangan</span></div>
        <dl class="kv">${rows}<dt>Oxirgi marta</dt><dd>${esc(u.last_seen || "—")}</dd><dt>Ro'yxatdan o'tgan</dt><dd>${esc(u.created_at)}</dd></dl>
        <div class="actions">
          <button class="btn primary sm" id="u-train">Kameradan o'rgatish</button>
          <label class="btn ghost sm">Rasm qo'shish<input type="file" id="u-files" accept="image/*" multiple hidden></label>
          <button class="btn ghost sm" id="u-samples">Namunalar</button>
          <button class="btn ghost sm" id="u-edit">Tahrirlash</button>
          <button class="btn danger sm" id="u-del">O'chirish</button>
        </div>
      </div>
    </div>`);
  $("#u-edit").onclick = () => { closeModal(); fillForm(u); };
  $("#u-train").onclick = async () => {
    closeModal(); await api("/api/registration/start", { method: "POST", json: { pid } });
    go("camera"); if (!state.camOn) startCamera();
  };
  $("#u-files").onchange = (e) => { const fs = [...e.target.files]; if (fs.length) { closeModal(); uploadImages(pid, fs, false, u.name); } };
  $("#u-del").onclick = async () => {
    if (!confirm(`${u.name} va uning barcha ma'lumotlari o'chirilsinmi?`)) return;
    await api(`/api/users/${pid}`, { method: "DELETE" }); closeModal(); toast("O'chirildi", "var(--bad)"); loadUsers();
  };
  $("#u-samples").onclick = async () => {
    openModal(`<div class="modal-head"><h2>${esc(u.name)} — namunalar</h2><div class="spinner"></div></div>`);
    const s = await api(`/api/users/${pid}/samples`);
    const src = { reg: "ro'yxat", auto: "avto-o'rganish", photo: "rasm" };
    openModal(`<div class="modal-head"><h2>${esc(u.name)} — o'rganilgan yuzlar (${s.length})</h2><button class="close">×</button></div>
      <div class="samples">${s.map((x) => `<figure><img src="${x.image}"><figcaption>${src[x.source] || x.source}<br>${esc(x.created_at.slice(5, 16))}</figcaption></figure>`).join("") || '<div class="empty">Namuna yo\'q</div>'}</div>`);
  };
}

/* ================= DASHBOARD ================= */
const KPIS = [
  ["users", "Jami foydalanuvchilar", "#7c5cff", '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'],
  ["unique_today", "Bugun kelganlar", "#22c55e", '<path d="M20 6L9 17l-5-5"/>'],
  ["recognized_today", "Bugungi aniqlashlar", "#22d3ee", '<circle cx="12" cy="12" r="3"/><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/>'],
  ["unknown_today", "Noma'lum shaxslar", "#f43f5e", '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>'],
  ["new_today", "Bugun ro'yxatdan o'tgan", "#f59e0b", '<path d="M12 5v14M5 12h14"/>'],
  ["samples", "O'rganilgan namunalar", "#a78bfa", '<path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/>'],
];
const EV_STYLE = { recognized: "", unknown: "bad", learned: "violet", registered: "warn", imported: "body", updated: "body", deleted: "bad" };

async function loadDashboard() {
  const d = await api("/api/dashboard" + (state.evType ? `?type=${state.evType}` : ""));
  const s = d.summary; s.samples = s.face_samples + s.body_samples;
  $("#kpis").innerHTML = KPIS.map(([k, l, c, ic]) => `
    <div class="card kpi" style="--c:${c}"><div class="ic" style="background:${c}22;color:${c}"><svg viewBox="0 0 24 24">${ic}</svg></div>
      <div class="v">${s[k] ?? 0}</div><div class="l">${l}</div></div>`).join("");
  $("#dash-time").textContent = `Yangilandi: ${new Date().toLocaleTimeString("uz-UZ")} · baza hajmi ${s.db_size_mb.toFixed(1)} MB · ${s.face_samples} yuz / ${s.body_samples} tana vektori`;
  barChart($("#chart-days"), d.daily.map((x) => x[0].slice(5).replace("-", ".")), [[d.daily.map((x) => x[1]), "#7c5cff"], [d.daily.map((x) => x[2]), "#f43f5e"]]);
  barChart($("#chart-hours"), d.hourly.map((_, i) => String(i).padStart(2, "0")), [[d.hourly, "#22d3ee"]], 3);

  const types = state.meta.event_types;
  $("#ev-chips").innerHTML = [["", "Barchasi"], ...Object.entries(types)].map(([k, v]) => `<span class="chip ${state.evType === k ? "active" : ""}" data-t="${k}">${esc(v)}</span>`).join("");
  $$("#ev-chips .chip").forEach((c) => (c.onclick = () => { state.evType = c.dataset.t; loadDashboard(); }));
  $("#ev-body").innerHTML = d.events.length ? d.events.map((e) => {
    const snap = ["recognized", "unknown"].includes(e.type) ? `<img class="thumb" loading="lazy" src="/api/events/${e.id}/snapshot" data-eid="${e.id}" onerror="this.style.visibility='hidden'">` : `<div class="thumb"></div>`;
    const name = e.user_name || (e.type === "unknown" ? "Noma'lum" : "—");
    return `<tr><td>${snap}</td><td><b>${esc(name)}</b></td><td><span class="tag ${EV_STYLE[e.type] ?? ""}">${esc(types[e.type] || e.type)}</span></td>
      <td class="muted">${esc(e.method || "")}</td><td>${["recognized", "unknown"].includes(e.type) && e.note ? `<div class="attrs">${e.note.split(", ").map((a) => `<span class="attr">${esc(a)}</span>`).join("")}</div>` : `<span class="muted">${esc(["recognized", "unknown"].includes(e.type) ? "" : e.note || "")}</span>`}</td><td>${e.score != null ? Math.round(e.score * 100) + "%" : ""}</td><td class="muted">${esc(e.created_at)}</td></tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">Hodisalar yo'q</td></tr>`;
  $$("#ev-body .thumb[data-eid]").forEach((img) => (img.onclick = () => openModal(`<div class="modal-head"><h2>Kadrdagi surat</h2><button class="close">×</button></div><img class="result-img" src="${img.src}">`)));

  $("#top-users").innerHTML = d.top.length ? d.top.map((u, i) => `
    <div class="row"><div class="rank">${i + 1}</div><div class="avatar" style="background-image:url('/api/users/${u.id}/photo?v=${state.photoVer}'),${avatarBg(u.name)}"></div>
      <div class="meta"><b>${esc(u.name)}</b><br><small>${esc(u.last_seen || "hali ko'rilmagan")}</small></div><div class="n">${u.seen}</div></div>`).join("")
    : `<div class="empty">Foydalanuvchilar yo'q</div>`;
}

function barChart(el, labels, series, every = 1) {
  const W = el.clientWidth || 600, H = el.clientHeight || 220, pl = 34, pr = 8, pt = 10, pb = 26;
  const max = Math.max(1, ...series.flatMap((s) => s[0]));
  const step = Math.max(1, Math.ceil(max / 4)), vmax = step * 4;
  const slot = (W - pl - pr) / labels.length, bw = Math.max(3, Math.min(26, slot * 0.62 / series.length));
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const y = H - pb - (H - pt - pb) * i / 4;
    g += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="#262c40" stroke-dasharray="${i ? "3 4" : ""}"/><text x="${pl - 8}" y="${y + 4}" text-anchor="end" fill="#8a92ab" font-size="11">${step * i}</text>`;
  }
  labels.forEach((lab, i) => {
    const x0 = pl + i * slot + (slot - bw * series.length) / 2;
    series.forEach(([vals, color], j) => {
      const v = vals[i], h = (H - pt - pb) * v / vmax;
      if (v > 0) g += `<rect x="${x0 + j * bw}" y="${H - pb - h}" width="${bw - 2}" height="${h}" rx="4" fill="${color}"><title>${lab}: ${v}</title></rect>`;
    });
    if (i % every === 0) g += `<text x="${pl + i * slot + slot / 2}" y="${H - 8}" text-anchor="middle" fill="#8a92ab" font-size="11">${lab}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${g}</svg>`;
}
setInterval(() => { if (state.page === "dashboard" && $("#modal-bg").classList.contains("hidden")) loadDashboard(); }, 4000);
window.addEventListener("resize", () => { if (state.page === "dashboard") loadDashboard(); });

/* ================= SETTINGS ================= */
const SETTINGS = [
  ["face_match", "Yuz tanish chegarasi", "Begonalar tanilib qolsa — oshiring; sizni tanimasa — kamaytiring.", 0.25, 0.7, 0.01],
  ["face_learn", "Avtomatik o'rganish ishonchi", "Faqat shundan yuqori ishonchda yangi namuna saqlanadi.", 0.4, 0.85, 0.01],
  ["face_novelty", "Yangi ko'rinish chegarasi", "Kichikroq — kamroq, lekin xilma-xilroq namunalar.", 0.6, 0.95, 0.01],
  ["body_match", "Tana tanish chegarasi", "Yuz ko'rinmaganda tana bo'yicha tanish uchun.", 0.5, 0.95, 0.01],
  ["face_gallery_max", "Yuz namunalari (max)", "Har bir odam uchun saqlanadigan yuz vektorlari soni.", 30, 400, 10],
  ["body_gallery_max", "Tana namunalari (max)", "Kiyim o'zgargani sari eskilari almashtiriladi.", 10, 200, 5],
  ["learn_interval", "O'rganish oralig'i (s)", "Bir odamni ikki marta o'rganish orasidagi minimal vaqt.", 0.5, 10, 0.5],
  ["min_face_px", "Minimal yuz o'lchami (px)", "Kichik/uzoqdagi yuzlardan o'rganilmaydi.", 30, 200, 5],
];
async function loadSettings() {
  const s = await api("/api/settings");
  $("#settings").innerHTML = SETTINGS.map(([k, t, d, min, max, st]) => `
    <div class="setting"><div><b>${t}</b><p>${d}</p></div>
      <input type="range" min="${min}" max="${max}" step="${st}" value="${s[k]}" data-k="${k}"><output>${s[k]}</output></div>`).join("");
  $$("#settings input").forEach((inp) => {
    inp.oninput = () => (inp.nextElementSibling.textContent = inp.value);
    inp.onchange = async () => { await setSetting(inp.dataset.k, Number(inp.value)); toast("Sozlama saqlandi", "var(--ok)"); };
  });
}

/* ================= HISOBLAR (admin) ================= */
async function loadAccounts() {
  const { accounts, roles } = await api("/api/accounts");
  $("#acc-body").innerHTML = accounts.map((a) => {
    const me = a.id === state.me.id;
    return `<tr data-id="${a.id}" data-name="${esc(a.full_name)}">
      <td><div class="acc-user"><div class="avatar" style="background:${avatarBg(a.full_name)}">${esc(initials(a.full_name))}</div>
        <div><b>${esc(a.full_name)}</b>${me ? '<span class="you">SIZ</span>' : ""}</div></div></td>
      <td class="muted">${esc(a.username)}</td>
      <td><select class="acc-role" ${me ? "disabled" : ""}>${Object.entries(roles).map(([k, v]) => `<option value="${k}" ${k === a.role ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></td>
      <td><label class="switch"><input type="checkbox" class="acc-active" ${a.active ? "checked" : ""} ${me ? "disabled" : ""}><span></span>${a.active ? "Faol" : "Bloklangan"}</label></td>
      <td class="muted">${esc(a.last_login || "—")}</td><td class="muted">${esc(a.created_at)}</td>
      <td style="white-space:nowrap"><button class="btn ghost sm acc-pw">Parol</button> ${me ? "" : '<button class="btn danger sm acc-del">O\'chirish</button>'}</td></tr>`;
  }).join("");
  $$("#acc-body tr").forEach((tr) => {
    const id = tr.dataset.id, name = tr.dataset.name;
    const put = async (body, msg) => {
      try { await api(`/api/accounts/${id}`, { method: "PUT", json: body }); toast(msg, "var(--ok)"); }
      catch (e) { toast(e.message, "var(--bad)"); }
      loadAccounts();
    };
    $(".acc-role", tr).onchange = (e) => put({ role: e.target.value }, `${name}: rol o'zgartirildi`);
    $(".acc-active", tr).onchange = (e) => put({ active: e.target.checked }, `${name}: ${e.target.checked ? "faollashtirildi" : "bloklandi"}`);
    $(".acc-pw", tr).onclick = () => passwordModal(id, name);
    const del = $(".acc-del", tr);
    if (del) del.onclick = async () => {
      if (!confirm(`${name} hisobi o'chirilsinmi?`)) return;
      try { await api(`/api/accounts/${id}`, { method: "DELETE" }); toast("Hisob o'chirildi", "var(--bad)"); }
      catch (e) { toast(e.message, "var(--bad)"); }
      loadAccounts();
    };
  });
}

function passwordModal(id, name) {
  const own = !id || Number(id) === state.me.id;
  openModal(`<div class="modal-head"><h2>${own ? "Parolni o'zgartirish" : esc(name) + " — yangi parol"}</h2><button class="close">×</button></div>
    <form id="pw-form" class="auth-form" style="max-width:420px">
      ${own ? '<div class="field"><label>Joriy parol</label><input type="password" name="old" required autocomplete="current-password"></div>' : ""}
      <div class="field"><label>Yangi parol</label><input type="password" name="new" required minlength="6" autocomplete="new-password"></div>
      <div class="field"><label>Yangi parolni takrorlang</label><input type="password" name="new2" required autocomplete="new-password"></div>
      <div class="auth-error" id="pw-error"></div>
      <button class="btn primary" type="submit">Saqlash</button></form>`);
  $("#pw-form").onsubmit = async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    if (f.new !== f.new2) { $("#pw-error").textContent = "Parollar mos kelmadi"; return; }
    try {
      if (own) await api("/api/auth/password", { method: "POST", json: { old: f.old, new: f.new } });
      else await api(`/api/accounts/${id}`, { method: "PUT", json: { password: f.new } });
      closeModal(); toast("Parol o'zgartirildi", "var(--ok)");
    } catch (err) { $("#pw-error").textContent = err.message; }
  };
}

/* ================= KIRISH / RO'YXATDAN O'TISH ================= */
function authTab(tab) {
  $$(".auth-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $(".auth-tabs").classList.toggle("right", tab === "signup");
  $("#login-form").classList.toggle("hidden", tab !== "login");
  $("#signup-form").classList.toggle("hidden", tab !== "signup");
  $("#login-error").textContent = "";
  $("#signup-error").textContent = "";
}
$$("[data-tab]").forEach((b) => b.addEventListener("click", () => authTab(b.dataset.tab)));
$$(".eye").forEach((b) => b.addEventListener("click", () => { const i = b.previousElementSibling; i.type = i.type === "password" ? "text" : "password"; }));
$("#signup-form [name=password]").addEventListener("input", (e) => {
  const v = e.target.value;
  const sc = [v.length >= 6, v.length >= 10, /[A-Z]/.test(v) && /[a-z]/.test(v), /\d/.test(v), /[^A-Za-z0-9]/.test(v)].filter(Boolean).length;
  $("#pw-bar").style.width = `${sc * 20}%`;
  $("#pw-bar").style.background = ["#f43f5e", "#f43f5e", "#f59e0b", "#f59e0b", "#22c55e", "#22c55e"][sc];
});
async function submitAuth(form, url, errEl, check = () => null) {
  const f = Object.fromEntries(new FormData(form));
  const err = check(f);
  if (err) { $(errEl).textContent = err; return; }
  const btn = $("button[type=submit]", form);
  btn.disabled = true;
  try {
    const r = await api(url, { method: "POST", json: { ...f, remember: !!f.remember } });
    if (r.first_admin) toast("Siz birinchi foydalanuvchisiz — administrator huquqi berildi", "var(--ok)");
    await startApp(r.user);
  } catch (e) { $(errEl).textContent = e.message; }
  btn.disabled = false;
}
$("#login-form").addEventListener("submit", (e) => { e.preventDefault(); submitAuth(e.target, "/api/auth/login", "#login-error"); });
$("#signup-form").addEventListener("submit", (e) => {
  e.preventDefault();
  submitAuth(e.target, "/api/auth/register", "#signup-error", (f) => (f.password !== f.password2 ? "Parollar mos kelmadi" : null));
});

function showAuth(hasAccounts) {
  $("#app").classList.add("hidden");
  $("#auth").classList.remove("hidden");
  if (!hasAccounts) {
    authTab("signup");
    $("#signup-hint").textContent = "Tizimda hali hisob yo'q. Birinchi yaratilgan hisob administrator bo'ladi.";
  } else authTab("login");
}

async function startApp(me) {
  state.me = me;
  $("#auth").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $$("nav a").forEach((a) => a.classList.toggle("hidden", !can(a.dataset.page)));
  $("#me-name").textContent = me.full_name;
  $("#me-role").textContent = me.role_label;
  $("#me-avatar").textContent = initials(me.full_name);
  $("#me-avatar").style.background = avatarBg(me.full_name);
  // oddiy foydalanuvchi kamerada boshqaruv tugmalarini ko'rmaydi
  $$("#page-camera .head-actions, #btn-reg-cancel").forEach((el) => el.classList.toggle("hidden", !isAdmin()));
  $(".cam-foot").classList.toggle("hidden", !can("camera"));
  state.meta = await api("/api/meta");
  if (isAdmin()) {
    buildForm();
    const s = await api("/api/settings");
    $("#sw-learn").checked = s.auto_learn;
    $("#sw-body").checked = s.body_enabled;
    $("#sw-attr").checked = s.attributes_enabled;
    if (!Object.keys(state.meta.attributes || {}).length) $("#sw-attr").closest("label").classList.add("hidden");
    refreshCount();
  }
  go(me.pages[0]);
  if (can("camera")) {
    try {
      const perm = await navigator.permissions?.query({ name: "camera" });
      if (perm?.state === "granted") startCamera();
    } catch { /* ruxsat holatini aniqlab bo'lmadi */ }
  }
}

$("#btn-me-menu").addEventListener("click", (e) => { e.stopPropagation(); $("#me-menu").classList.toggle("hidden"); });
document.addEventListener("click", () => $("#me-menu").classList.add("hidden"));
$("#btn-change-pw").addEventListener("click", () => passwordModal(null));
$("#btn-logout").addEventListener("click", async () => {
  state.camOn = false;
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  await api("/api/auth/logout", { method: "POST" });
  location.reload();
});

/* ================= INIT ================= */
(async () => {
  const r = await api("/api/auth/me");
  if (r.user) startApp(r.user); else showAuth(r.has_accounts);
})();
