"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n ?? 0).toLocaleString();

const state = {
  sessionId: null,
  sessions: [],
  messages: [],      // {id, role, content, turn?, status?, error?, retry?}
  summary: null,
  facts: [],
  stats: null,
  health: null,
  selectedId: null,  // assistant message whose turn the inspector shows
  pending: false,
  tab: "turn",
};

// ---------- api ----------

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

async function* readSSE(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield { event, data: data ? JSON.parse(data) : null };
    }
  }
}

// ---------- markdown (escaped first; small subset) ----------

function esc(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return s
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function markdown(src) {
  const parts = esc(src).split("```");
  let html = "";
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      // Code fence (also handles an unclosed fence while streaming).
      const code = part.replace(/^[\w+-]*\n/, "");
      html += `<pre><code>${code.replace(/\n$/, "")}</code></pre>`;
      return;
    }
    let list = null;
    let para = [];
    const flushPara = () => { if (para.length) { html += `<p>${para.map(inline).join("<br>")}</p>`; para = []; } };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
    for (const line of part.split("\n")) {
      const ul = line.match(/^\s*[-*]\s+(.*)/);
      const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      const h = line.match(/^#{1,6}\s+(.*)/);
      if (ul || ol) {
        flushPara();
        const tag = ul ? "ul" : "ol";
        if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; }
        html += `<li>${inline((ul || ol)[1])}</li>`;
      } else if (h) {
        flushPara(); closeList();
        html += `<h4>${inline(h[1])}</h4>`;
      } else if (!line.trim()) {
        flushPara(); closeList();
      } else {
        closeList();
        para.push(line);
      }
    }
    flushPara(); closeList();
  });
  return html;
}

// ---------- helpers ----------

function newSessionId() {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

function ago(ts) {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function sessionTitle(s) {
  const t = (s.first_message || "Untitled").replace(/\s+/g, " ").trim();
  return t.length > 60 ? t.slice(0, 57) + "…" : t;
}

function el(tag, attrs = {}, html = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  if (html) node.innerHTML = html;
  return node;
}

function toast(text, action) {
  const t = $("toast");
  t.innerHTML = `<span>${esc(text)}</span>`;
  if (action) {
    const b = el("button", { class: "link-btn", type: "button" }, esc(action.label));
    b.addEventListener("click", () => { t.hidden = true; action.run(); });
    t.append(b);
  }
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 6000);
}

// ---------- sessions ----------

function renderSessions() {
  const list = $("session-list");
  list.innerHTML = "";
  if (!state.sessions.length) {
    list.append(el("p", { class: "session-empty" }, "Conversations you start will appear here."));
  }
  for (const s of state.sessions) {
    const item = el("button", {
      class: "session-item", type: "button",
      "aria-current": String(s.session_id === state.sessionId),
      onclick: () => { openSession(s.session_id); closeDrawer(); },
    });
    item.innerHTML = `<span class="s-title">${esc(sessionTitle(s))}</span>
      <span class="s-meta">${s.turns} ${s.turns === 1 ? "turn" : "turns"} · ${ago(s.updated_at)}</span>`;
    list.append(item);
  }
  const st = state.stats;
  $("global-stats").innerHTML = st
    ? `<strong>${fmt(st.triples)}</strong> facts across <strong>${fmt(st.sessions)}</strong> conversations<br>
       <strong>${fmt(st.answer_input_tokens)}</strong> tokens sent · ~<strong>${fmt(st.full_history_tokens)}</strong> without memory`
    : "";
}

async function refreshSidebar() {
  [state.sessions, state.stats] = await Promise.all([api("/api/sessions"), api("/api/stats")]);
  renderSessions();
}

async function openSession(id) {
  state.sessionId = id;
  history.replaceState(null, "", `#${id}`);
  const data = await api(`/api/sessions/${id}`);
  state.messages = data.messages;
  state.summary = data.summary;
  const last = [...state.messages].reverse().find((m) => m.turn);
  state.selectedId = last ? last.id : null;
  renderAll();
  scrollToEnd(false);
}

function newChat() {
  if (state.pending) return;
  state.sessionId = newSessionId();
  history.replaceState(null, "", `#${state.sessionId}`);
  state.messages = [];
  state.summary = null;
  state.selectedId = null;
  renderAll();
  closeDrawer();
  $("input").focus();
}

// ---------- chat ----------

function renderHeader() {
  const s = state.sessions.find((x) => x.session_id === state.sessionId);
  const firstUser = state.messages.find((m) => m.role === "user");
  $("chat-title").textContent = s ? sessionTitle(s) : firstUser ? sessionTitle({ first_message: firstUser.content }) : "New conversation";

  const turns = state.messages.filter((m) => m.turn && m.turn.answer_input_tokens != null);
  if (!turns.length) {
    $("chat-meter").textContent = "Memory is shared across all conversations";
    return;
  }
  const sent = turns.reduce((a, m) => a + m.turn.answer_input_tokens, 0);
  const full = turns.reduce((a, m) => a + m.turn.full_history_tokens, 0);
  const pct = full > sent ? Math.round((1 - sent / full) * 100) : 0;
  $("chat-meter").innerHTML =
    `${turns.length} ${turns.length === 1 ? "turn" : "turns"} · ${fmt(sent)} tokens sent · ~${fmt(full)} with full history` +
    (pct > 0 ? ` · <b>${pct}% saved</b>` : "");
}

function chipText(turn) {
  const n = turn.memory.triples.length;
  const parts = [`${n} ${n === 1 ? "fact" : "facts"} recalled`];
  if (turn.answer_input_tokens != null) parts.push(`${fmt(turn.answer_input_tokens)} tok sent`);
  let html = parts.map(esc).join(" · ");
  if (turn.added && turn.added.length) html += ` <span class="learned">+${turn.added.length} learned</span>`;
  return html;
}

function renderMessage(m) {
  const node = el("article", { class: `msg ${m.role}`, "data-id": m.id });
  if (m.id === state.selectedId && m.id !== latestTurnId()) node.classList.add("selected");
  const body = el("div", { class: "body" });

  if (m.role === "user") {
    body.textContent = m.content;
  } else if (!m.content && m.status && m.status !== "error") {
    body.innerHTML = `<div class="skeleton" aria-label="Waiting for reply"><span></span><span></span></div>`;
  } else {
    body.innerHTML = markdown(m.content || "");
  }
  node.append(body);

  if (m.role !== "assistant") return node;

  if (m.status === "recalling" || m.status === "answering") {
    node.append(el("div", { class: "status-line" }, `<span class="dot"></span>${m.status === "recalling" ? "Recalling memory" : "Answering"}`));
  } else if (m.status === "saving") {
    node.append(el("div", { class: "status-line" }, `<span class="dot"></span>Saving to memory`));
  } else if (m.status === "error") {
    const line = el("div", { class: "status-line error" }, esc(m.error));
    if (m.retry) {
      const b = el("button", { class: "link-btn", type: "button" }, "Retry");
      b.addEventListener("click", () => retry(m));
      line.append(b);
    }
    node.append(line);
  }
  if (m.turn && m.status !== "recalling") {
    node.append(el("button", {
      class: "turn-chip", type: "button", "aria-label": "Show memory for this reply",
      onclick: () => selectTurn(m.id, true),
    }, chipText(m.turn)));
  }
  return node;
}

function renderMessages() {
  const box = $("messages");
  box.innerHTML = "";
  if (!state.messages.length) {
    const empty = el("div", { class: "empty-chat" });
    empty.innerHTML = `<h2>Start a conversation</h2>
      <p>Tell it something worth remembering, then start a new chat and ask about it.
      Each reply shows which facts were recalled and how many tokens that cost compared with resending every past message.</p>`;
    const sug = el("div", { class: "suggestions" });
    for (const text of [
      "I'm Ada. I lead the payments team at Kora and I like short, direct answers.",
      "I'm moving from Lagos to Nairobi in March and need to plan the move.",
      "What do you know about me so far?",
    ]) {
      sug.append(el("button", { class: "suggestion", type: "button", onclick: () => { $("input").value = text; autosize(); $("input").focus(); } }, esc(text)));
    }
    empty.append(sug);
    box.append(empty);
    return;
  }
  for (const m of state.messages) box.append(renderMessage(m));
}

function updateMessage(m) {
  const old = document.querySelector(`.msg[data-id="${m.id}"]`);
  if (old) old.replaceWith(renderMessage(m));
}

function scrollToEnd(smooth = true) {
  const box = $("messages");
  box.scrollTo({ top: box.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

function nearBottom() {
  const box = $("messages");
  return box.scrollHeight - box.scrollTop - box.clientHeight < 120;
}

async function send(text) {
  text = text.trim();
  if (!text || state.pending || !state.health?.ok) return;
  state.pending = true;
  setComposerBusy(true);

  const user = { id: `u-${Date.now()}`, role: "user", content: text };
  const reply = { id: `pending-${Date.now()}`, role: "assistant", content: "", status: "recalling" };
  state.messages.push(user, reply);
  renderMessages();
  renderHeader();
  scrollToEnd();

  let frame = 0;
  try {
    const res = await fetch(`/api/sessions/${state.sessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    for await (const { event, data } of readSSE(res)) {
      if (event === "memory") {
        reply.turn = { memory: data };
        reply.status = "answering";
        selectTurn(reply.id);
        updateMessage(reply);
      } else if (event === "delta") {
        reply.content += data.text;
        if (!frame) {
          frame = requestAnimationFrame(() => {
            frame = 0;
            const stick = nearBottom();
            updateMessage(reply);
            if (stick) scrollToEnd(false);
          });
        }
      } else if (event === "answer") {
        cancelAnimationFrame(frame); frame = 0;
        const oldId = reply.id;
        Object.assign(reply, { id: data.message_id, content: data.content, status: "saving" });
        reply.turn = pickTurn(data);
        user.id = `u-${data.message_id}`;
        if (state.selectedId === oldId) state.selectedId = reply.id;
        const node = document.querySelector(`.msg[data-id="${oldId}"]`);
        if (node) node.dataset.id = reply.id;
        updateMessage(reply);
        renderHeader();
        renderInspector();
      } else if (event === "saved") {
        reply.turn = pickTurn(data);
        reply.status = null;
        updateMessage(reply);
      } else if (event === "error") {
        reply.status = "error";
        reply.error = data.message;
        // Nothing was stored if the failure came before the answer; allow a clean retry.
        reply.retry = typeof reply.id === "string";
        reply.retryText = text;
        updateMessage(reply);
      }
    }
    if (reply.status && reply.status !== "error") {
      reply.status = "error";
      reply.error = "The connection closed before the reply finished.";
      updateMessage(reply);
    }
  } catch (err) {
    cancelAnimationFrame(frame);
    Object.assign(reply, { status: "error", error: err.message, retry: true, retryText: text });
    updateMessage(reply);
  } finally {
    state.pending = false;
    setComposerBusy(false);
    renderHeader();
    await Promise.all([refreshSidebar(), loadFacts(), loadSummary()]).catch(() => {});
    renderInspector();
  }
}

function pickTurn(d) {
  return {
    memory: d.memory, answer_input_tokens: d.answer_input_tokens, full_history_tokens: d.full_history_tokens,
    extraction_tokens: d.extraction_tokens, added: d.added, removed: d.removed,
  };
}

function retry(m) {
  const i = state.messages.indexOf(m);
  if (i < 1) return;
  state.messages.splice(i - 1, 2);
  renderMessages();
  send(m.retryText);
}

function setComposerBusy(busy) {
  const disabled = busy || !state.health?.ok;
  $("send").disabled = disabled;
  $("send").textContent = busy ? "Sending" : "Send";
  $("input").disabled = !state.health?.ok;
}

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight + 2, 200) + "px";
}

// ---------- inspector ----------

function selectTurn(id, reveal = false) {
  const prev = document.querySelector(".msg.selected");
  if (prev) prev.classList.remove("selected");
  state.selectedId = id;
  const node = document.querySelector(`.msg[data-id="${id}"]`);
  if (node && id !== latestTurnId()) node.classList.add("selected");
  setTab("turn");
  if (reveal && window.matchMedia("(max-width: 820px)").matches) setView("memory");
}

function latestTurnId() {
  const last = [...state.messages].reverse().find((m) => m.role === "assistant");
  return last && last.id;
}

function setTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const p of ["turn", "facts", "summary"]) $(`tab-${p}`).hidden = p !== tab;
  renderInspector();
}

function triplesTable(rows) {
  const head = `<div class="head"><span>Subject</span><span>Predicate</span><span>Object</span></div>`;
  return `<div class="triples" role="table">${head}${rows
    .map(([s, p, o]) => `<div role="row"><span>${esc(s)}</span><span class="p">${esc(p)}</span><span>${esc(o)}</span></div>`)
    .join("")}</div>`;
}

function renderTurn() {
  const panel = $("tab-turn");
  const m = state.messages.find((x) => x.id === state.selectedId);
  if (!m || !m.turn) {
    panel.innerHTML = `<p class="hint">Send a message to see which facts get recalled for it, what the model learns from the reply, and how many input tokens that costs compared with resending the full history.</p>`;
    return;
  }
  const t = m.turn;
  const mem = t.memory;
  const budget = state.health?.budget;
  let html = "";

  // Token comparison
  html += `<section class="section"><h2 class="section-title">Input tokens</h2>`;
  if (t.answer_input_tokens == null) {
    html += `<p class="hint">Measured once the reply arrives.</p>`;
  } else {
    const sent = t.answer_input_tokens, full = t.full_history_tokens;
    const max = Math.max(sent, full, 1);
    html += `<div class="bars">
        <span class="label">With memory</span>
        <div class="track"><div class="fill sent" style="width:${(sent / max) * 100}%"></div></div>
        <span class="val">${fmt(sent)}</span>
        <span class="label">Full history</span>
        <div class="track"><div class="fill" style="width:${(full / max) * 100}%"></div></div>
        <span class="val">~${fmt(full)}</span>
      </div>`;
    if (full > sent) {
      html += `<p class="verdict"><b>${Math.round((1 - sent / full) * 100)}% fewer</b> input tokens than resending every past message.</p>`;
    } else {
      html += `<p class="verdict">History is still short, so memory costs ${fmt(sent - full)} more tokens than resending it. Savings start once conversations add up.</p>`;
    }
    html += `<p class="sub">Memory block ${fmt(mem.tokens)}${budget ? ` of ${fmt(budget)} budget` : ""}` +
      (t.extraction_tokens ? ` · extraction ${fmt(t.extraction_tokens)}` : "") + `</p>`;
  }
  html += `</section>`;

  // Recalled
  html += `<section class="section"><h2 class="section-title">Recalled facts <span class="aside">${mem.triples.length}</span></h2>`;
  html += mem.triples.length ? triplesTable(mem.triples) : `<p class="hint">No stored facts matched this message.</p>`;
  html += `</section>`;

  if (mem.session_summary) {
    html += `<section class="section"><h2 class="section-title">Conversation summary used</h2><p class="prose">${esc(mem.session_summary)}</p></section>`;
  }
  if (mem.related_summaries.length) {
    html += `<section class="section"><h2 class="section-title">Related conversations</h2><ul class="related">${mem.related_summaries
      .map((s) => `<li>${esc(s)}</li>`).join("")}</ul></section>`;
  }

  // Learned
  html += `<section class="section"><h2 class="section-title">Learned from this reply</h2>`;
  if (m.status === "saving" || m.status === "answering" || m.status === "recalling") {
    html += `<div class="skeleton"><span></span><span></span></div>`;
  } else if (!t.added?.length && !t.removed?.length) {
    html += `<p class="hint">Nothing new worth storing.</p>`;
  } else {
    html += `<ul class="changes">` +
      (t.added || []).map((x) => `<li class="add"><span class="sign" aria-label="added">+</span><span>${esc(x.join(" · "))}</span></li>`).join("") +
      (t.removed || []).map((x) => `<li class="rm"><span class="sign" aria-label="removed">−</span><span>${esc(x.join(" · "))}</span></li>`).join("") +
      `</ul>`;
  }
  html += `</section>`;
  panel.innerHTML = html;
}

function renderFacts() {
  $("facts-count").textContent = state.facts.length || "";
  const list = $("facts-list");
  const q = $("facts-filter").value.trim().toLowerCase();
  const rows = q
    ? state.facts.filter((f) => `${f.subject} ${f.predicate} ${f.object}`.includes(q))
    : state.facts;

  if (!state.facts.length) {
    list.innerHTML = `<p class="hint">No facts yet. After each reply the model extracts subject, predicate, object triples, and they show up here.</p>`;
    return;
  }
  if (!rows.length) {
    list.innerHTML = `<p class="hint">No facts match “${esc(q)}”.</p>`;
    return;
  }
  const groups = new Map();
  for (const f of rows) {
    if (!groups.has(f.subject)) groups.set(f.subject, []);
    groups.get(f.subject).push(f);
  }
  list.innerHTML = "";
  for (const [subject, facts] of groups) {
    const g = el("section", { class: "fact-group" });
    g.append(el("h3", {}, esc(subject)));
    for (const f of facts) {
      const row = el("div", { class: "fact-row" });
      row.innerHTML = `<span class="p">${esc(f.predicate)}</span><span class="o">${esc(f.object)}</span>
        <span class="n" title="Times mentioned">${f.mentions > 1 ? "×" + f.mentions : ""}</span>`;
      const del = el("button", { class: "icon-btn", type: "button", "aria-label": `Forget: ${f.subject} ${f.predicate} ${f.object}` },
        `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>`);
      del.addEventListener("click", () => forget(f, row));
      row.append(del);
      g.append(row);
    }
    list.append(g);
  }
}

async function forget(f, row) {
  row.classList.add("removing");
  const body = { subject: f.subject, predicate: f.predicate, object: f.object };
  try {
    await api("/api/facts", { method: "DELETE", body });
  } catch (err) {
    row.classList.remove("removing");
    toast(`Couldn't forget that fact: ${err.message}`);
    return;
  }
  await Promise.all([loadFacts(), refreshSidebar()]);
  toast(`Forgot “${f.subject} ${f.predicate} ${f.object}”`, {
    label: "Undo",
    run: async () => {
      await api("/api/facts", { method: "POST", body: { ...body, session_id: f.session_id } });
      await Promise.all([loadFacts(), refreshSidebar()]);
    },
  });
}

function renderSummary() {
  const panel = $("tab-summary");
  panel.innerHTML = state.summary
    ? `<section class="section"><h2 class="section-title">This conversation</h2><p class="prose">${esc(state.summary)}</p>
       <p class="sub" style="margin-top:12px">Rewritten after every reply and sent instead of the raw transcript.</p></section>`
    : `<p class="hint">After the first reply, a short running summary of this conversation appears here. It's what gets sent in place of the full transcript.</p>`;
}

function renderInspector() {
  if (state.tab === "turn") renderTurn();
  else if (state.tab === "facts") renderFacts();
  else renderSummary();
  $("facts-count").textContent = state.facts.length || "";
}

async function loadFacts() {
  state.facts = await api("/api/facts");
  if (state.tab === "facts") renderFacts();
  $("facts-count").textContent = state.facts.length || "";
}

async function loadSummary() {
  if (!state.sessionId) return;
  const data = await api(`/api/sessions/${state.sessionId}`);
  state.summary = data.summary;
}

// ---------- layout ----------

function setView(view) {
  $("app").dataset.view = view;
  for (const b of document.querySelectorAll(".view-switch button")) b.setAttribute("aria-selected", String(b.dataset.view === view));
}

function openDrawer() { $("app").classList.add("drawer-open"); $("scrim").hidden = false; }
function closeDrawer() { $("app").classList.remove("drawer-open"); $("scrim").hidden = true; }

function renderAll() {
  renderSessions();
  renderHeader();
  renderMessages();
  renderInspector();
}

function renderNotice() {
  const n = $("notice");
  if (state.health && !state.health.ok) {
    n.innerHTML = `No Anthropic credentials found. Set <code>ANTHROPIC_API_KEY</code> and restart <code>deduct serve</code>. You can still browse past conversations and stored facts.`;
    n.hidden = false;
  } else {
    n.hidden = true;
  }
}

// ---------- boot ----------

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("input").value;
  if (!text.trim() || state.pending) return;
  $("input").value = "";
  autosize();
  send(text);
});
$("input").addEventListener("input", autosize);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
$("new-chat").addEventListener("click", newChat);
$("open-sessions").addEventListener("click", openDrawer);
$("close-sessions").addEventListener("click", closeDrawer);
$("scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => setTab(b.dataset.tab));
for (const b of document.querySelectorAll(".view-switch button")) b.addEventListener("click", () => setView(b.dataset.view));
$("facts-filter").addEventListener("input", renderFacts);

(async function boot() {
  state.health = await api("/api/health").catch((e) => ({ ok: false, error: e.message }));
  renderNotice();
  setComposerBusy(false);
  await Promise.all([refreshSidebar(), loadFacts()]).catch(() => {});
  const fromHash = location.hash.slice(1);
  if (/^[A-Za-z0-9_-]{1,64}$/.test(fromHash)) {
    await openSession(fromHash);
  } else if (state.sessions.length) {
    await openSession(state.sessions[0].session_id);
  } else {
    newChat();
  }
})();
