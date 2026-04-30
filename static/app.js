const API_BASE = "http://localhost:8000";

document.querySelectorAll("nav a").forEach((tab) => {
  tab.addEventListener("click", (e) => {
    e.preventDefault();
    document
      .querySelectorAll("nav a")
      .forEach((t) => t.classList.remove("active"));
    document
      .querySelectorAll("section")
      .forEach((s) => s.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.section).classList.add("active");
  });
});

async function postFile(endpoint, fileInput, resultEl) {
  const file = fileInput.files[0];
  if (!file) {
    resultEl.innerHTML = "<p>No file selected.</p>";
    return;
  }

  resultEl.innerHTML = "<p>Loading...</p>";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    resultEl.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
  } catch (err) {
    resultEl.innerHTML = `<p class="error">Error: ${err.message}</p>`;
  }
}

function runPipeline() {
  postFile(
    "/pipeline",
    document.getElementById("pipeline-file"),
    document.getElementById("pipeline-result"),
  );
}
function runDetect() {
  postFile(
    "/detect",
    document.getElementById("detect-file"),
    document.getElementById("detect-result"),
  );
}
function runRecognize() {
  postFile(
    "/recognize",
    document.getElementById("recognize-file"),
    document.getElementById("recognize-result"),
  );
}
