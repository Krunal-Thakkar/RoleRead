const API_BASE = "/api";
const MAX_JOBS = 5;

function getSessionId() {
  let id = localStorage.getItem("career_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("career_session_id", id);
  }
  return id;
}

const SESSION_ID = getSessionId();

function apiHeaders(extra = {}) {
  return { "X-Session-Id": SESSION_ID, ...extra };
}

async function apiFetch(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: apiHeaders(options.headers || {}),
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

// ---- State ----
let resumeUploaded = false;
let jobCount = 0;

// ---- Elements ----
const resumeInput = document.getElementById("resume-input");
const resumeUploadBtn = document.getElementById("resume-upload-btn");
const resumeStatus = document.getElementById("resume-status");
const resumeSkills = document.getElementById("resume-skills");

const jobInput = document.getElementById("job-input");
const jobUploadBtn = document.getElementById("job-upload-btn");
const jobStatus = document.getElementById("job-status");
const jobList = document.getElementById("job-list");

const chatLocked = document.getElementById("chat-locked");
const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");

function setStatus(el, message, isError = false) {
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function renderChips(container, items) {
  container.innerHTML = "";
  items.forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = s;
    container.appendChild(chip);
  });
}

function addJobCard(job) {
  const card = document.createElement("div");
  card.className = "job-card";
  const title = document.createElement("div");
  title.className = "job-title";
  title.textContent = `${job.label} — ${job.filename}`;
  card.appendChild(title);
  const chips = document.createElement("div");
  chips.className = "chips";
  card.appendChild(chips);
  jobList.appendChild(card);
  renderChips(chips, job.skills);
}

function maybeUnlockChat() {
  if (resumeUploaded && jobCount > 0) {
    chatLocked.classList.add("hidden");
    chatWindow.classList.remove("hidden");
    chatForm.classList.remove("hidden");
  }
}

function updateJobLimitUI() {
  if (jobCount >= MAX_JOBS) {
    jobInput.disabled = true;
    jobUploadBtn.disabled = true;
    setStatus(jobStatus, `Maximum of ${MAX_JOBS} job descriptions reached.`);
  }
}

resumeUploadBtn.addEventListener("click", async () => {
  const file = resumeInput.files[0];
  if (!file) {
    setStatus(resumeStatus, "Choose a resume file first.", true);
    return;
  }
  setStatus(resumeStatus, "Processing...");
  resumeUploadBtn.disabled = true;
  try {
    const formData = new FormData();
    formData.append("file", file);
    const data = await apiFetch("/resume", { method: "POST", body: formData });
    resumeUploaded = true;
    setStatus(resumeStatus, `Processed: ${data.filename} (${data.chunk_count} chunks)`);
    renderChips(resumeSkills, data.skills);
    maybeUnlockChat();
  } catch (err) {
    setStatus(resumeStatus, `Error: ${err.message}`, true);
  } finally {
    resumeUploadBtn.disabled = false;
  }
});

async function uploadOneJob(file) {
  const formData = new FormData();
  formData.append("file", file);
  const data = await apiFetch("/jobs", { method: "POST", body: formData });
  jobCount += 1;
  addJobCard(data);
  return data;
}

jobUploadBtn.addEventListener("click", async () => {
  if (jobCount >= MAX_JOBS) {
    updateJobLimitUI();
    return;
  }
  const files = Array.from(jobInput.files || []);
  if (!files.length) {
    setStatus(jobStatus, "Choose one or more job description files first.", true);
    return;
  }
  const allowed = files.slice(0, MAX_JOBS - jobCount);
  const skipped = files.length - allowed.length;

  jobUploadBtn.disabled = true;
  let succeeded = 0;
  for (const file of allowed) {
    setStatus(jobStatus, `Processing ${file.name} (${succeeded + 1}/${allowed.length})...`);
    try {
      const data = await uploadOneJob(file);
      succeeded += 1;
      setStatus(jobStatus, `Processed: ${data.filename} as ${data.label}`);
    } catch (err) {
      setStatus(jobStatus, `Error on ${file.name}: ${err.message}`, true);
      break; // stop the batch on first failure, keep whatever succeeded so far
    }
  }
  if (skipped > 0) {
    setStatus(
      jobStatus,
      `Processed ${succeeded} job(s). Skipped ${skipped} — max of ${MAX_JOBS} reached.`,
      succeeded === 0
    );
  }
  jobInput.value = "";
  maybeUnlockChat();
  updateJobLimitUI();
  jobUploadBtn.disabled = jobCount >= MAX_JOBS;
});

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// A markdown table row: `| cell | cell |` (at least two pipe-delimited cells).
const _TABLE_ROW_RE = /^\|(.+)\|$/;
// The header/body separator row, e.g. `| --- | :---: |`.
const _TABLE_SEP_RE = /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/;

function _splitTableRow(line) {
  const inner = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return inner.split("|").map((c) => c.trim());
}

function renderMarkdown(text) {
  const lines = text.split("\n").map(escapeHtml);
  let html = "";
  let inList = false;
  let listBuffer = [];

  function closeList() {
    if (!inList) return;
    const items = listBuffer.map((s) => `<li>${s}</li>`).join("");
    html += `<ul>${items}</ul>`;
    inList = false;
    listBuffer = [];
  }

  const applyInline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // All list markers (numbered "1." or bulleted "-"/"*") render as plain bullets:
  // the LLM's numbered output isn't reliably sequential/well-formed markdown (e.g.
  // blank lines between items, or restarting at 1), so bullets sidestep that entirely.
  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = applyInline(rawLine);

    // Markdown table: a `| ... |` row immediately followed by a `| --- | --- |` separator row.
    if (_TABLE_ROW_RE.test(rawLine.trim()) && lines[i + 1] && _TABLE_SEP_RE.test(lines[i + 1].trim())) {
      closeList();
      const headerCells = _splitTableRow(rawLine).map(applyInline);
      let j = i + 2;
      const bodyRows = [];
      while (j < lines.length && _TABLE_ROW_RE.test(lines[j].trim())) {
        bodyRows.push(_splitTableRow(lines[j]).map(applyInline));
        j += 1;
      }
      const thead = `<thead><tr>${headerCells.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
      const tbody = `<tbody>${bodyRows
        .map((row) => `<tr>${row.map((c) => `<td>${c}</td>`).join("")}</tr>`)
        .join("")}</tbody>`;
      html += `<div class="table-wrap"><table>${thead}${tbody}</table></div>`;
      i = j - 1;
      continue;
    }

    const listMatch = line.match(/^(?:\d+\.|[-*])\s+(.*)$/);
    if (listMatch) {
      inList = true;
      listBuffer.push(listMatch[1]);
    } else if (line.trim() === "") {
      // A blank line only ends the list if what follows isn't another list item
      // (the LLM often inserts a blank line between list items).
      const next = lines.slice(i + 1).find((l) => l.trim() !== "");
      if (!inList || !next || !/^(?:\d+\.|[-*])\s+/.test(next)) {
        closeList();
      }
    } else {
      closeList();
      html += `<p>${line}</p>`;
    }
  }
  closeList();
  return html;
}

function addMessage(role, text, sources) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  if (role === "assistant") {
    msg.innerHTML = renderMarkdown(text);
  } else {
    msg.textContent = text;
  }
  if (sources && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = "Sources: " + sources.map((s) => s.source).join(", ");
    msg.appendChild(src);
  }
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return msg;
}

function addThinkingIndicator() {
  const msg = document.createElement("div");
  msg.className = "msg assistant thinking";
  msg.innerHTML = '<span class="thinking-dots"><span></span><span></span><span></span></span>';
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return msg;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  addMessage("user", question);
  chatInput.value = "";
  chatInput.disabled = true;
  chatSendBtn.disabled = true;
  const thinkingEl = addThinkingIndicator();
  try {
    const data = await apiFetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    thinkingEl.remove();
    addMessage("assistant", data.answer, data.sources);
  } catch (err) {
    thinkingEl.remove();
    addMessage("assistant", `Error: ${err.message}`);
  } finally {
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
  }
});

// Restore existing session's documents on reload.
(async function restoreSession() {
  try {
    const data = await apiFetch("/documents");
    if (data.resume) {
      resumeUploaded = true;
      setStatus(resumeStatus, `Processed: ${data.resume.filename}`);
      renderChips(resumeSkills, data.resume.skills);
    }
    data.jobs.forEach((job) => {
      jobCount += 1;
      addJobCard(job);
    });
    maybeUnlockChat();
    updateJobLimitUI();
  } catch (_) {
    // No prior session data; ignore.
  }
})();
