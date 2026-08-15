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

jobUploadBtn.addEventListener("click", async () => {
  if (jobCount >= MAX_JOBS) {
    updateJobLimitUI();
    return;
  }
  const file = jobInput.files[0];
  if (!file) {
    setStatus(jobStatus, "Choose a job description file first.", true);
    return;
  }
  setStatus(jobStatus, "Processing...");
  jobUploadBtn.disabled = true;
  try {
    const formData = new FormData();
    formData.append("file", file);
    const data = await apiFetch("/jobs", { method: "POST", body: formData });
    jobCount += 1;
    setStatus(jobStatus, `Processed: ${data.filename} as ${data.label}`);
    addJobCard(data);
    jobInput.value = "";
    maybeUnlockChat();
    updateJobLimitUI();
  } catch (err) {
    setStatus(jobStatus, `Error: ${err.message}`, true);
  } finally {
    jobUploadBtn.disabled = jobCount >= MAX_JOBS;
  }
});

function addMessage(role, text, sources) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.textContent = text;
  if (sources && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = "Sources: " + sources.map((s) => s.source).join(", ");
    msg.appendChild(src);
  }
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  addMessage("user", question);
  chatInput.value = "";
  chatInput.disabled = true;
  try {
    const data = await apiFetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    addMessage("assistant", data.answer, data.sources);
  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
  } finally {
    chatInput.disabled = false;
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
