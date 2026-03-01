#include "webui/ui_bundle.h"

namespace inferflux {
namespace webui {

const std::string &UiHtml() {
  static const std::string kHtml = R"(<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>InferFlux WebUI</title>
    <style>{{css}}</style>
  </head>
  <body>
    <div class="container">
      <header>
        <div>
          <h1>InferFlux WebUI</h1>
          <p class="subtitle">Serving model: <strong>{{backend}}</strong></p>
        </div>
        <div class="api-key">
          <label for="apiKey">API Key</label>
          <input id="apiKey" placeholder="Bearer dev-key-123" />
          <button onclick="saveApiKey()">Save</button>
        </div>
      </header>
      <section class="split">
        <div class="left">
          <label for="modelSelect">Models</label>
          <select id="modelSelect"></select>
          <button onclick="refreshModels()">Refresh</button>
          <label for="prompt">Prompt</label>
          <textarea id="prompt" rows="6">Hello, InferFlux!</textarea>
          <div class="btn-row">
            <button onclick="sendCompletion()">Completion</button>
            <button onclick="sendChat()">Chat</button>
          </div>
        </div>
        <div class="right">
          <label>Output</label>
          <pre id="output"></pre>
        </div>
      </section>
      <section>
        <h2>Chat History</h2>
        <ul id="history"></ul>
      </section>
    </div>
    <script>{{js}}</script>
  </body>
</html>
)";
  return kHtml;
}

const std::string &UiCss() {
  static const std::string kCss = R"CSS(
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  padding: 24px;
  background: #0f172a;
  color: #f8fafc;
}
.container {
  max-width: 1000px;
  margin: 0 auto;
}
header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 24px;
}
.subtitle {
  color: #cbd5f5;
}
.api-key {
  display: flex;
  align-items: flex-end;
  gap: 8px;
}
.api-key input {
  width: 240px;
}
.split {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}
.left,
.right {
  flex: 1;
  min-width: 300px;
}
textarea,
input,
select {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #334155;
  padding: 12px;
  background: #0f172a;
  color: #f8fafc;
  font-size: 16px;
  margin-bottom: 12px;
}
.btn-row {
  display: flex;
  gap: 12px;
}
button {
  border: none;
  border-radius: 6px;
  padding: 10px 16px;
  cursor: pointer;
  background: #38bdf8;
  color: #0f172a;
  font-weight: 600;
}
pre {
  white-space: pre-wrap;
  background: #020617;
  border-radius: 8px;
  padding: 16px;
  min-height: 200px;
  border: 1px solid #1e293b;
}
ul#history {
  list-style: none;
  padding: 0;
  border: 1px solid #1e293b;
  border-radius: 8px;
  max-height: 240px;
  overflow-y: auto;
}
ul#history li {
  padding: 8px 12px;
  border-bottom: 1px solid #1e293b;
}
ul#history li:last-child {
  border-bottom: none;
}
)CSS";
  return kCss;
}

const std::string &UiJs() {
  static const std::string kJs = R"JS(
const apiKeyInput = document.getElementById("apiKey");
function headers() {
  const token =
    localStorage.getItem("inferflux_api_key") ||
    apiKeyInput.value ||
    "Bearer dev-key-123";
  return { "Content-Type": "application/json", Authorization: token };
}
function saveApiKey() {
  localStorage.setItem("inferflux_api_key", apiKeyInput.value);
}
async function refreshModels() {
  const select = document.getElementById("modelSelect");
  select.innerHTML = "";
  try {
    const res = await fetch("/v1/models", { headers: headers() });
    const data = await res.json();
    data.data.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.id;
      select.appendChild(opt);
    });
  } catch (err) {
    document.getElementById("output").textContent =
      "Failed to load models: " + err;
  }
}
async function sendCompletion() {
  const prompt = document.getElementById("prompt").value;
  const model = document.getElementById("modelSelect").value;
  const res = await fetch("/v1/completions", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ prompt, model, max_tokens: 128 }),
  });
  const text = await res.text();
  document.getElementById("output").textContent = text;
}
async function sendChat() {
  const prompt = document.getElementById("prompt").value;
  const model = document.getElementById("modelSelect").value;
  const history = document.getElementById("history");
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 128,
    }),
  });
  const data = await res.json();
  const choice = data.choices?.[0]?.message?.content || JSON.stringify(data);
  const liUser = document.createElement("li");
  liUser.textContent = "User: " + prompt;
  const liAssistant = document.createElement("li");
  liAssistant.textContent = "Assistant: " + choice;
  history.prepend(liAssistant);
  history.prepend(liUser);
  document.getElementById("output").textContent = JSON.stringify(
    data,
    null,
    2
  );
}
window.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("inferflux_api_key");
  if (saved) {
    apiKeyInput.value = saved;
  }
  refreshModels();
});
)JS";
  return kJs;
}

} // namespace webui
} // namespace inferflux
