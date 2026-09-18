function render(data) {
  const status = document.querySelector("#status");
  const intent = document.querySelector("#intent");
  const confidence = document.querySelector("#confidence");
  const bar = document.querySelector("#confidenceBar");
  const candidates = document.querySelector("#candidates");
  if (data?.error) {
    status.textContent = `Local predictor unavailable: ${data.error}`;
    intent.textContent = "—";
    confidence.textContent = "Start local_predictor first.";
    bar.value = 0;
    candidates.textContent = "{}";
    return;
  }
  const answer = data?.answers?.intent || data?.intent || {};
  const probs = answer.probabilities || data?.probabilities || {};
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const top = sorted[0];
  intent.textContent = top ? top[0] : "unknown";
  const score = top ? Number(top[1]) : 0;
  confidence.textContent = `confidence ${(score * 100).toFixed(1)}% · ${new Date(data.updatedAt || Date.now()).toLocaleTimeString()}`;
  bar.value = score;
  candidates.textContent = JSON.stringify(Object.fromEntries(sorted.slice(0, 5)), null, 2);
  status.textContent = "Local prediction ready";
}

async function refresh() {
  const value = await chrome.storage.local.get({ lastPrediction: { error: "no_prediction" } });
  render(value.lastPrediction);
}

async function saveFeedback(label) {
  const value = await chrome.storage.local.get({ lastPrediction: null, feedback: [] });
  if (!value.lastPrediction || value.lastPrediction.error) return;
  const feedback = Array.isArray(value.feedback) ? value.feedback : [];
  feedback.push({ label, predicted: value.lastPrediction.intent || value.lastPrediction.answers?.intent?.choice || null, at: Date.now() });
  await chrome.storage.local.set({ feedback: feedback.slice(-500) });
  document.querySelector("#feedback").textContent = "Saved locally";
}

document.querySelector("#refresh").addEventListener("click", refresh);
document.querySelector("#correct").addEventListener("click", () => saveFeedback("correct"));
document.querySelector("#wrong").addEventListener("click", () => saveFeedback("wrong"));
chrome.storage.onChanged.addListener(refresh);
refresh();
