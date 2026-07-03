/* EmbodiedGen V2 — interactions. Vanilla, no deps (model-viewer loaded separately). */
(function () {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 1. Nav: transparent over hero -> solid after scroll */
  const nav = document.getElementById("nav");
  const onScroll = () => nav.classList.toggle("is-solid", window.scrollY > 40);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  /* 2. Reveal on scroll */
  const reveals = document.querySelectorAll(".reveal");
  if (reduce || !("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("is-in"));
  } else {
    const io = new IntersectionObserver(
      (es, obs) => es.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("is-in"); obs.unobserve(e.target); }
      }),
      { rootMargin: "0px 0px -10% 0px", threshold: 0.12 }
    );
    reveals.forEach((el) => io.observe(el));
  }

  /* 3. Nav scroll-spy */
  const links = Array.from(document.querySelectorAll(".nav__link[data-sect]"));
  const map = new Map();
  links.forEach((l) => {
    const id = l.dataset.sect;
    const t = document.getElementById(id === "top" ? "top" : id);
    if (t) map.set(t, l);
  });
  if ("IntersectionObserver" in window && map.size) {
    const sio = new IntersectionObserver(
      (es) => es.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((l) => l.classList.remove("is-active"));
          const a = map.get(e.target); if (a) a.classList.add("is-active");
        }
      }),
      { rootMargin: "-45% 0px -50% 0px" }
    );
    map.forEach((_, t) => sio.observe(t));
  }

  /* 4. Asset 3D viewer */
  const mv = document.getElementById("mv");
  const gallery = document.getElementById("gallery");
  if (mv && gallery) {
    const BASE = "assets/models/";
    let assets = [];
    let afford = {};
    let cur = 0;
    let mode = "visual";
    const affCard = document.getElementById("affCard");
    const metaCard = document.getElementById("metaCard");
    const fmtCard = document.getElementById("fmtCard");
    const affParts = document.getElementById("affParts");
    const affDetail = document.getElementById("affDetail");

    const el = {
      name: document.getElementById("m-name"),
      cat: document.getElementById("m-cat"),
      desc: document.getElementById("m-desc"),
      h: document.getElementById("m-h"),
      m: document.getElementById("m-m"),
      f: document.getElementById("m-f"),
      a: document.getElementById("m-a"),
    };

    function load() {
      const a = assets[cur];
      const file = mode === "collision" ? a.collision
                 : mode === "affordance" ? a.name + "_afford.glb"
                 : a.model;
      mv.src = BASE + file;
      mv.setAttribute("alt", a.label + " — " + mode);
      const isAff = mode === "affordance";
      if (affCard) affCard.hidden = !isAff;
      if (metaCard) metaCard.hidden = isAff;
      if (fmtCard) fmtCard.hidden = isAff;
      if (isAff) renderAfford();
    }

    function renderAfford() {
      const a = assets[cur];
      const data = afford[a.name];
      if (!data || !affParts) return;
      affParts.innerHTML = data.parts.map((p, i) =>
        `<li><button class="part${i === 0 ? " is-active" : ""}" data-i="${i}">
           <span class="part__dot" style="background:${p.color}"></span>
           <span class="part__name">${p.part_name}</span>
           <span class="part__grasp ${p.graspable ? "ok" : "no"}">${p.graspable ? "graspable" : "—"}</span>
         </button></li>`).join("");
      affParts.querySelectorAll(".part").forEach((b) =>
        b.addEventListener("click", () => {
          affParts.querySelectorAll(".part").forEach((x) => x.classList.toggle("is-active", x === b));
          showPart(+b.dataset.i);
        }));
      showPart(0);
    }
    function showPart(i) {
      const p = afford[assets[cur].name].parts[i];
      const labels = (p.labels || []).map((l) => `<span class="pd__chip">${l}</span>`).join("");
      affDetail.innerHTML =
        `<div class="pd__labels">${labels}</div>
         <p class="pd__desc">${p.description || ""}</p>`;
    }

    function fillMeta() {
      const a = assets[cur];
      el.name.textContent = a.label;
      el.cat.textContent = a.category;
      el.desc.textContent = a.description || "—";
      el.h.textContent = a.height ? (+a.height).toFixed(2) + " m" : "—";
      el.m.textContent = a.mass ? (+a.mass).toFixed(2) + " kg" : "—";
      el.f.textContent = a.friction ? (+a.friction).toFixed(2) : "—";
      el.a.textContent = a.aesthetic ? (+a.aesthetic).toFixed(1) + " / 10" : "—";
    }

    function select(i) {
      cur = i;
      document.querySelectorAll(".thumb").forEach((t, k) =>
        t.classList.toggle("is-active", k === i));
      load(); fillMeta();
    }

    let locked = false;
    let autoTimer = null;
    function setMode(m) {
      mode = m;
      document.querySelectorAll(".mode").forEach((x) =>
        x.classList.toggle("is-active", x.dataset.mode === m));
      load();
    }
    const AUTO_ORDER = ["visual", "collision", "affordance"];
    function startAuto() {
      if (reduce || locked || autoTimer) return;
      autoTimer = setInterval(() => {
        const next = AUTO_ORDER[(AUTO_ORDER.indexOf(mode) + 1) % AUTO_ORDER.length];
        setMode(next);
      }, 4000);
    }
    function lockModes() {
      locked = true;
      if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    }
    document.querySelectorAll(".mode").forEach((b) =>
      b.addEventListener("click", () => { lockModes(); setMode(b.dataset.mode); }));

    // Heavy init (model-viewer lib ~1MB + first GLB) is deferred until the
    // Assets section approaches the viewport, to keep initial page load light.
    let inited = false;
    function initViewer() {
      if (inited) return;
      inited = true;
      if (!window.__mvLoaded) {
        window.__mvLoaded = true;
        const s = document.createElement("script");
        s.type = "module";
        s.src = "https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js";
        document.head.appendChild(s);
      }
      fetch(BASE + "affordance.json").then((r) => r.json()).then((j) => { afford = j; }).catch(() => {});
      fetch(BASE + "assets.json")
        .then((r) => r.json())
        .then((data) => {
          assets = data;
          gallery.innerHTML = data.map((a, i) =>
            `<button class="thumb${i === 0 ? " is-active" : ""}" role="tab" data-i="${i}">
               <span class="thumb__dot"></span>${a.label}</button>`).join("");
          gallery.querySelectorAll(".thumb").forEach((t) =>
            t.addEventListener("click", () => select(+t.dataset.i)));
          select(0);
          startAuto();
        })
        .catch((e) => { gallery.innerHTML = "<span class='viewer__hint'>assets.json failed to load (serve over http).</span>"; console.error(e); });

      if (reduce) mv.removeAttribute("auto-rotate");
    }

    // Trigger viewer init on the FIRST of: (a) user scrolls near Assets, or
    // (b) the page finishes loading + goes idle — so the first asset is warmed
    // in the background right after the hero (not competing with first paint),
    // and is ready before the user scrolls. initViewer() is idempotent.
    const assetsSect = document.getElementById("assets");
    if ("IntersectionObserver" in window && assetsSect) {
      const aio = new IntersectionObserver((es, obs) => {
        if (es.some((e) => e.isIntersecting)) { obs.disconnect(); initViewer(); }
      }, { rootMargin: "400px 0px" });
      aio.observe(assetsSect);
    }
    const warm = () => (window.requestIdleCallback || ((cb) => setTimeout(cb, 1)))(initViewer);
    if (document.readyState === "complete") warm();
    else window.addEventListener("load", warm, { once: true });
    setTimeout(warm, 3000); // fallback if the load event is delayed by media
  }

  /* ---- 5. Vibe Coding edit sessions ---- */
  const vibeHist = document.getElementById("vibeHist");
  const vibeVid = document.getElementById("vibeVid");
  const vibeBadge = document.getElementById("vibeBadge");
  if (vibeHist && vibeVid) {
    const sessions = {
      kitchen: {
        prefix: "scene_kitchen_S",
        turns: [
          { cmd: "Generate a simple kitchen", d: "init" },
          { cmd: "Add a dining table with four chairs", d: "add", t: "+ table · 4 chairs" },
          { cmd: "Put some fruit on the table", d: "add", t: "+ fruit" },
        ],
      },
      living: {
        prefix: "scene_living_room_S",
        turns: [
          { cmd: "Generate a simple living room", d: "init" },
          { cmd: "Set up a TV, vase, table and chair", d: "add", t: "+ tv · vase · table · chair" },
          { cmd: "Set the table for chess", d: "add", t: "+ chessboard" },
        ],
      },
    };
    let sess = "kitchen";

    function play(stage) {
      const s = sessions[sess];
      vibeVid.src = "assets/video/" + s.prefix + stage + ".mp4";
      vibeVid.load();
      if (!reduce) { const p = vibeVid.play(); if (p && p.catch) p.catch(() => {}); }
      vibeBadge.textContent = "S" + stage;
      vibeHist.querySelectorAll(".turn").forEach((t, i) =>
        t.classList.toggle("is-active", i === stage));
    }
    function render() {
      const s = sessions[sess];
      vibeHist.innerHTML = s.turns.map((tn, i) => {
        const dcls = tn.d === "add" ? "add" : tn.d === "del" ? "del" : "init";
        const dtxt = tn.d === "init" ? "init" : tn.t;
        return `<li><button class="turn${i === 0 ? " is-active" : ""}" data-s="${i}">
          <span class="turn__cmd">${tn.cmd}</span>
          <span class="turn__delta turn__delta--${dcls}">${dtxt}</span></button></li>`;
      }).join("");
      vibeHist.querySelectorAll(".turn").forEach((b) =>
        b.addEventListener("click", () => play(+b.dataset.s)));
      play(0);
    }
    document.querySelectorAll(".vtab").forEach((b) =>
      b.addEventListener("click", () => {
        sess = b.dataset.sess;
        document.querySelectorAll(".vtab").forEach((x) => x.classList.toggle("is-active", x === b));
        render();
      }));
    render();
  }

  /* ---- 5a. Lazy-load + pause offscreen videos (network + CPU perf) ----
     Section videos use preload="none" + data-autoplay: they only fetch and
     play when scrolled into view, and pause when scrolled away. The hero keeps
     native autoplay for instant first paint but is paused offscreen too. */
  const dav = Array.from(document.querySelectorAll("video[data-autoplay]"));
  if (dav.length) {
    const playV = (v) => { if (reduce) return; const p = v.play(); if (p && p.catch) p.catch(() => {}); };
    if ("IntersectionObserver" in window) {
      const vio = new IntersectionObserver((es) => es.forEach((e) => {
        if (e.isIntersecting) playV(e.target); else e.target.pause();
      }), { rootMargin: "300px 0px", threshold: 0.1 });
      dav.forEach((v) => vio.observe(v));
    } else {
      dav.forEach(playV);
    }
  }

  /* ---- 5b. Result count-up (animate the after-value from its baseline) ---- */
  const vals = Array.from(document.querySelectorAll(".result__val"));
  if (vals.length) {
    const run = (el) => {
      const from = parseFloat(el.dataset.from);
      const to = parseFloat(el.dataset.to);
      if (reduce || !isFinite(from) || !isFinite(to)) { el.textContent = to.toFixed(1); return; }
      const dur = 1100, t0 = performance.now();
      const tick = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
        el.textContent = (from + (to - from) * e).toFixed(1);
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = to.toFixed(1);
      };
      requestAnimationFrame(tick);
    };
    if (reduce || !("IntersectionObserver" in window)) {
      vals.forEach((el) => (el.textContent = parseFloat(el.dataset.to).toFixed(1)));
    } else {
      const cio = new IntersectionObserver((es, obs) => es.forEach((e) => {
        if (e.isIntersecting) { run(e.target); obs.unobserve(e.target); }
      }), { threshold: 0.6 });
      vals.forEach((el) => cio.observe(el));
    }
  }

  /* ---- 6. BibTeX copy (one button per cite block) ---- */
  document.querySelectorAll(".cite__copy").forEach((btn) => {
    const code = btn.parentElement.querySelector(".cite__code");
    if (!code) return;
    btn.addEventListener("click", () => {
      const txt = code.textContent;
      const done = () => {
        btn.textContent = "Copied";
        btn.classList.add("is-done");
        setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("is-done"); }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done).catch(() => {});
      } else {
        const r = document.createRange(); r.selectNode(code);
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
        try { document.execCommand("copy"); done(); } catch (e) {}
        s.removeAllRanges();
      }
    });
  });
})();
