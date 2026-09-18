import { redactHistoryItem, redactObservation, redactText, redactUrl } from "./redact.js";

const DEFAULT_ENDPOINT = "http://127.0.0.1:8765/v1/predict";
const HISTORY_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const MAX_HISTORY = 32;
const MAX_EVENTS = 64;

async function settings() {
  const stored = await chrome.storage.local.get({
    predictorEndpoint: DEFAULT_ENDPOINT,
    enabled: true,
    collectHistory: true,
  });
  return stored;
}

async function eventBuffer(tabId) {
  const key = `events:${tabId}`;
  const value = await chrome.storage.session.get({ [key]: [] });
  return { key, events: Array.isArray(value[key]) ? value[key] : [] };
}

async function appendEvent(tabId, event) {
  const { key, events } = await eventBuffer(tabId);
  events.push({ ...event, t: Date.now() });
  await chrome.storage.session.set({ [key]: events.slice(-MAX_EVENTS) });
}

async function historySnapshot() {
  const cfg = await settings();
  if (!cfg.collectHistory) return [];
  const items = await chrome.history.search({
    text: "",
    startTime: Date.now() - HISTORY_WINDOW_MS,
    maxResults: MAX_HISTORY,
  });
  return items.map(redactHistoryItem);
}

async function currentObservation(tabId) {
  const tab = await chrome.tabs.get(tabId);
  const { events } = await eventBuffer(tabId);
  const recentHistory = await historySnapshot();
  return redactObservation({
    url: tab.url || "",
    title: tab.title || "",
    elements: [
      { role: "tab", id: String(tab.id), text: redactText(tab.title || "", 160) },
      { role: "browser_history", id: "recent", text: `${recentHistory.length} recent entries` },
    ],
    events: [
      ...recentHistory.map((item) => ({ action: "history_visit", target: item.url })),
      ...events.map((event) => ({ action: event.action, target: event.target })),
    ],
  });
}

async function predict(tabId, reason = "navigation") {
  const cfg = await settings();
  if (!cfg.enabled) return;
  const observation = await currentObservation(tabId);
  const payload = { observation, source: "edge-extension", reason, createdAt: Date.now() };
  let result;
  try {
    const response = await fetch(cfg.predictorEndpoint || DEFAULT_ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    result = response.ok ? await response.json() : { error: `predictor_http_${response.status}` };
  } catch (error) {
    result = { error: "predictor_unavailable", detail: String(error?.message || error) };
  }
  await chrome.storage.local.set({
    lastObservation: observation,
    lastPrediction: { ...result, reason, updatedAt: Date.now() },
  });
}

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.storage.local.set({ predictorEndpoint: DEFAULT_ENDPOINT, enabled: true, collectHistory: true });
  if (chrome.sidePanel?.setPanelBehavior) {
    await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
  }
});

chrome.runtime.onStartup.addListener(async () => {
  const tabs = await chrome.tabs.query({ active: true });
  for (const tab of tabs) {
    if (tab.id !== undefined) await predict(tab.id, "browser_startup");
  }
});

chrome.action.onClicked.addListener(async (tab) => {
  if (chrome.sidePanel?.open && tab?.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
  if (tab?.id !== undefined) await predict(tab.id, "action_click");
});

chrome.webNavigation.onCommitted.addListener(async (details) => {
  if (details.frameId !== 0 || details.tabId < 0) return;
  await appendEvent(details.tabId, { action: "navigation", target: redactUrl(details.url) });
  await predict(details.tabId, "navigation");
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  await chrome.storage.session.remove(`events:${tabId}`);
});
