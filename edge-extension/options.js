const defaults = { predictorEndpoint: "http://127.0.0.1:8765/v1/predict", enabled: true, collectHistory: true };

async function load() {
  const value = await chrome.storage.local.get(defaults);
  document.querySelector("#endpoint").value = value.predictorEndpoint;
  document.querySelector("#enabled").checked = value.enabled;
  document.querySelector("#collectHistory").checked = value.collectHistory;
}

document.querySelector("#save").addEventListener("click", async () => {
  const endpoint = document.querySelector("#endpoint").value.trim();
  if (!endpoint.startsWith("http://127.0.0.1:")) {
    document.querySelector("#status").textContent = "Only a 127.0.0.1 endpoint is allowed.";
    return;
  }
  await chrome.storage.local.set({
    predictorEndpoint: endpoint,
    enabled: document.querySelector("#enabled").checked,
    collectHistory: document.querySelector("#collectHistory").checked,
  });
  document.querySelector("#status").textContent = "Saved.";
});

load();
