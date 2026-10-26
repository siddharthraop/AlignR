// AlignR app: the scorecard, the Draft button, and each draft beside the published topic. No framework.
(() => {
  const CFG = window.ALIGNR;
  const qs = new URLSearchParams(location.search);
  const API = qs.get("api") || CFG.api;
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  const STAGES = ["find_siblings", "load_facts", "build_schema", "generate_topic", "checks", "revise", "open_pr"];
  let csh = {};
  let helpId = "scorecard";
  let run = null;
  let active = 0;

  // ---------------------------------------------------------------- theme
  const setTheme = (t) => { document.documentElement.dataset.theme = t; try { localStorage.setItem("alignr-theme", t); } catch (e) {} };
  try { const t = localStorage.getItem("alignr-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(dark ? "light" : "dark");
  });

  // ---------------------------------------------------------------- help panel (context-sensitive help from the DITA help set)
  fetch(CFG.help + "csh.json").then((r) => (r.ok ? r.json() : {})).then((j) => { csh = j; }).catch(() => {});
  function openHelp(id = helpId) {
    const t = csh[id];
    $("#help-title").textContent = t ? t.title : "Help";
    $("#help-short").textContent = t ? t.shortdesc : "Open the help to read about this part of AlignR.";
    $("#help-steps").innerHTML = t ? (t.steps || []).map((s) => `<li>${esc(s)}</li>`).join("") : "";
    $("#help-related").innerHTML = t && t.related && t.related.length
      ? `<h3>Related</h3>${t.related.map((r) => `<a href="${esc(CFG.help + r.href)}" target="_blank" rel="noopener">${esc(r.title)}</a>`).join("")}` : "";
    $("#help-open").href = t ? CFG.help + t.href : CFG.help;
    $("#help").classList.add("open"); $("#help").setAttribute("aria-hidden", "false");
    $("#help-btn").setAttribute("aria-expanded", "true"); $("#help-close").focus();
  }
  function closeHelp() {
    $("#help").classList.remove("open"); $("#help").setAttribute("aria-hidden", "true");
    $("#help-btn").setAttribute("aria-expanded", "false");
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-help]");
    if (b) { helpId = b.dataset.help; openHelp(helpId); }
  });
  $("#help-btn").addEventListener("click", () => ($("#help").classList.contains("open") ? closeHelp() : openHelp()));
  $("#help-close").addEventListener("click", closeHelp);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeHelp();
    if (e.key === "?" && !/input|select|textarea/i.test(document.activeElement.tagName)) openHelp();
  });

  // ---------------------------------------------------------------- scorecard
  const setRow = (m, value, ok, note = "") => {
    const tr = $(`#scorecard tr[data-m="${m}"]`);
    $(".v", tr).innerHTML = value + (note ? `<br><span class="meta">${note}</span>` : "");
    const st = $(".st", tr);
    st.className = "st " + (ok === true ? "met" : ok === false ? "missed" : "review");
    st.innerHTML = ok === true ? "Met" : ok === false ? "Missed" : "<span>Pending</span>";
  };

  async function scorecard() {
    const [web, git, merges] = await Promise.all([
      fetch(API + "/api/scorecard").then((r) => r.json()).catch(() => null),
      fetch(CFG.runs + "latest-github.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      fetch(CFG.runs + "merges.jsonl", { cache: "no-store" }).then((r) => (r.ok ? r.text() : "")).catch(() => ""),
    ]);
    const latest = (web && web.latest) || git;
    const from = [];
    if (latest) from.push(`Latest model run: ${latest.run} (${latest.source === "github" ? "Git path" : "web"}, ${latest.model}).`);
    if (git) from.push(`Latest Git-path run: ${git.run}.`);
    $("#sc-from").textContent = from.join(" ");
    const runsAll = (web ? web.runs : 0) + (git ? 1 : 0);
    const gramFails = (web ? web.grammar_failures_reaching_writer : 0) + (git ? git.metrics.grammar_failures_reaching_writer : 0);
    if (runsAll) setRow("grammar", String(gramFails), gramFails === 0, `over ${runsAll} logged run${runsAll > 1 ? "s" : ""}`);
    else setRow("grammar", "no runs yet", null);
    if (latest) {
      const m = latest.metrics;
      setRow("values", `${m.values_sourced_pct}%`, m.values_sourced_pct === 100, `${m.values_sourced} of ${m.values_total} values`);
      const per = latest.topics.filter((t) => t.has_sibling).map((t) => `${t.slot} ${pct(t.compare.edit_ratio)}`).join(" · ");
      setRow("edit", `mean ${pct(m.edit_ratio_mean)}, max ${pct(m.edit_ratio_max)}`, m.edit_ratio_met, esc(per));
    } else { setRow("values", "no runs yet", null); setRow("edit", "no runs yet", null); }
    if (git && git.issue_to_prs_seconds != null) {
      const s = git.issue_to_prs_seconds;
      setRow("time", `${Math.floor(s / 60)} min ${s % 60} s`, s < 900,
        `<a href="${esc(CFG.repo)}/issues/${esc(git.issue)}">issue #${esc(git.issue)}</a> → ${git.prs.length} draft pull requests`);
    } else setRow("time", "no issue run yet", null, `<a href="${esc(CFG.repo)}/issues/new?template=new-generation-set.yml">Open an issue</a> to start one`);
    const rows = merges.split("\n").filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (e) { return null; } }).filter(Boolean);
    if (rows.length) {
      const open = rows.reduce((a, r) => a + r.open_questions, 0);
      setRow("merge", String(open), open === 0, `over ${rows.length} merged draft pull request${rows.length > 1 ? "s" : ""}`);
    } else setRow("merge", "no merges yet", null, "The open-questions status blocks a merge while any draft comment is open");
    const withCost = (web ? web.runs_with_cost : 0) + (git && git.usage && "cost_usd" in git.usage ? 1 : 0);
    if (runsAll) setRow("cost", `${Math.round((100 * withCost) / runsAll)}%`, withCost === runsAll,
      `${withCost} of ${runsAll} runs; latest $${latest ? latest.usage.cost_usd.toFixed(4) : "—"}`);
    else setRow("cost", "no runs yet", null);
    return web;
  }

  // ---------------------------------------------------------------- drafting
  function progressView(topics) {
    $("#progress").innerHTML = `<div class="prog">${topics.map((t) => `<div class="name">${esc(t)}</div>
      <ol class="progress" data-topic="${esc(t)}">${STAGES.map((s) => `<li data-s="${s}">${s}</li>`).join("")}</ol>`).join("")}</div>`;
  }
  function mark(e) {
    const ol = $(`ol.progress[data-topic="${CSS.escape(e.topic)}"]`);
    if (!ol) return;
    const idx = STAGES.indexOf(e.stage === "withheld" ? "open_pr" : e.stage);
    ol.querySelectorAll("li").forEach((li, i) => {
      if (i < idx) { li.classList.add("done"); li.classList.remove("now"); }
      if (i === idx) li.classList.add("now");
    });
    if (e.stage === "open_pr" || e.stage === "withheld") ol.querySelectorAll("li").forEach((li) => { li.classList.remove("now"); li.classList.add("done"); });
  }

  async function draft() {
    const btn = $("#go");
    btn.disabled = true;
    $("#status").innerHTML = `<div class="notice">Drafting ${esc($("#gen").selectedOptions[0].textContent)}…</div>`;
    $("#progress").innerHTML = "";
    try {
      const r = await fetch(API + "/api/draft", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generation: $("#gen").value }) });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        const d = j.detail || {};
        $("#status").innerHTML = `<div class="notice err">${esc(d.message || d || r.statusText)}</div>`;
        if (d.latest) show(await (await fetch(`${API}/api/runs/${d.latest}`)).json());
        return;
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 1);
          if (!line.trim()) continue;
          const e = JSON.parse(line);
          if (e.stage === "plan") progressView(e.topics);
          else if (e.type === "stage") mark(e);
          else if (e.type === "error") $("#status").innerHTML = `<div class="notice err">${esc(e.message)}</div>`;
          else if (e.type === "result") {
            $("#status").innerHTML = `<div class="notice">Done in ${e.run.seconds} s. Each draft is below, beside the topic Google published.</div>`;
            show(e.run);
            scorecard();
          }
        }
      }
    } catch (err) {
      $("#status").innerHTML = `<div class="notice err">The AlignR API didn't answer (${esc(err.message)}). Try again in a minute.</div>`;
    } finally { btn.disabled = false; quota(); }
  }

  // ---------------------------------------------------------------- the run
  function viewHtml(blocks) {
    return (blocks || []).map((b) => {
      if (b.kind === "title") return `<div class="t">${esc(b.text)}</div>`;
      if (b.kind === "shortdesc") return `<p class="sd">${esc(b.text)}</p>`;
      if (b.kind === "section") return `<div class="sec">${esc(b.text)}</div>`;
      if (b.kind === "note") return `<div class="note">${esc(b.text)}</div>`;
      if (b.kind === "question") return `<div class="q">${esc(b.text)}</div>`;
      if (b.kind === "table") return `<div class="table-wrap"><table><thead><tr>${b.head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
        <tbody>${b.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
      return `<p>${esc(b.text)}</p>`;
    }).join("");
  }

  function show(r) {
    run = r; active = 0;
    $("#result").classList.remove("hidden");
    $("#run-id").textContent = r.run;
    $("#zip").href = `${API}/api/runs/${encodeURIComponent(r.run)}/zip`;
    const m = r.metrics;
    $("#band").innerHTML = [
      [r.model, `model · ${r.started.replace("T", " ").slice(0, 16)} UTC`],
      [`${r.seconds} s`, "to draft the set"],
      [`${m.grammar_failures_reaching_writer}`, "grammar failures reaching a writer"],
      [`${m.values_sourced_pct}%`, "values with a key or a source"],
      [pct(m.edit_ratio_mean), "mean edit ratio, topics with a sibling"],
      [`${m.open_questions}`, "open questions for people"],
      [`$${r.usage.cost_usd.toFixed(4)}`, `${r.usage.calls} model calls · logged`],
    ].map(([b, s]) => `<div><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");
    $("#tabs").innerHTML = r.topics.map((t, i) => `<button role="tab" type="button" data-i="${i}" aria-selected="${i === 0}">
      ${esc(t.topic_id)}<small>${t.stub ? "stub · no sibling" : "edit ratio " + pct(t.compare.edit_ratio)}</small></button>`).join("");
    topic(0);
  }

  function topic(i) {
    active = i;
    document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.i === i)));
    const t = run.topics[i];
    const g = t.grounding, c = t.compare || {};
    const pub = c.published_view ? viewHtml(c.published_view) : `<p class="meta">Not published.</p>`;
    $("#topic").innerHTML = `
      <div class="checks">
        <div><b>Grammar</b>${t.grammar.valid ? "Valid" : "Invalid"}: ${esc(t.grammar.grammar || "")}</div>
        <div><b>Rules</b>${(t.rules.errors || []).length} errors, ${(t.rules.warnings || []).length} warnings</div>
        <div><b>Grounding</b>${g.grounded} of ${g.total} values have a key or a source</div>
        <div><b>Model</b>${t.usage.calls} call(s), ${t.revisions} revision(s), $${t.usage.cost_usd.toFixed(4)}</div>
        <div><b>Built from</b>${t.siblings.length ? t.siblings.map((s) => `<code>${esc(s.split("/").pop())}</code>`).join(", ") : "No sibling: a stub with questions"}</div>
        <div><b>Edit ratio</b><span class="ratio">${pct(c.edit_ratio)}</span> ${t.has_sibling ? "(target under 40%)" : "(no sibling: not counted)"}</div>
      </div>
      ${t.fixes.length ? `<p class="meta">Safe fixes in code: ${t.fixes.map(esc).join("; ")}</p>` : ""}
      <div class="split">
        <div class="view"><h3><span>AlignR draft</span><span>${t.questions.length} open question(s)</span></h3>${viewHtml(c.draft_view)}</div>
        <div class="view"><h3><span>Published by Google</span><a href="${esc(run.facts_page)}" target="_blank" rel="noopener">TPU7x page</a></h3>${pub}
          <p class="meta">As adapted in dita-corpus (<code>${esc(t.published_href || "")}</code>), CC BY 4.0. Held out: AlignR never read it while drafting.</p></div>
      </div>
      <details><summary>DITA XML of the draft</summary><pre class="src">${esc(t.xml)}</pre></details>
      <details><summary>Report, as it appears on the draft pull request</summary><pre class="src">${esc(t.report || "")}</pre></details>
      <details><summary>Checks on each attempt</summary><pre class="src">${esc(JSON.stringify(t.history, null, 1))}</pre></details>`;
  }
  $("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-i]"); if (b) topic(+b.dataset.i); });
  $("#tabs").addEventListener("keydown", (e) => {
    if (!run || !["ArrowRight", "ArrowLeft"].includes(e.key)) return;
    const n = (active + (e.key === "ArrowRight" ? 1 : -1) + run.topics.length) % run.topics.length;
    topic(n); $(`#tabs button[data-i="${n}"]`).focus();
  });

  async function quota() {
    const h = await fetch(API + "/api/health").then((r) => r.json()).catch(() => null);
    $("#quota").textContent = h ? `${Math.max(0, h.daily_cap - h.drafts_today)} of ${h.daily_cap} public drafts left today · model ${h.model}` : "API not reachable";
  }

  $("#go").addEventListener("click", draft);
  scorecard().then(async () => {
    const latest = await fetch(API + "/api/runs/latest").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if (latest) {
      $("#status").innerHTML = `<div class="notice">Showing the latest run, from ${esc(latest.started.replace("T", " ").slice(0, 16))} UTC. Select Draft to run it again.</div>`;
      show(latest);
    }
  });
  quota();
})();
