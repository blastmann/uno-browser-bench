import assert from "node:assert/strict";

const listeners = {};
const localState = new Map();
const sessionState = new Map();

globalThis.chrome = {
  storage: {
    local: {
      async get(defaults) {
        const result = { ...defaults };
        for (const [key, value] of localState) result[key] = value;
        return result;
      },
      async set(values) {
        for (const [key, value] of Object.entries(values)) localState.set(key, value);
      },
    },
    session: {
      async get(defaults) {
        const result = { ...defaults };
        for (const [key, value] of sessionState) result[key] = value;
        return result;
      },
      async set(values) {
        for (const [key, value] of Object.entries(values)) sessionState.set(key, value);
      },
      async remove(key) {
        sessionState.delete(key);
      },
    },
  },
  history: {
    async search() {
      return [{ url: "https://github.com/example/repo?private=removed", title: "Synthetic history", lastVisitTime: Date.now() }];
    },
  },
  tabs: {
    async query() {
      return [{ id: 42 }];
    },
    async get() {
      return { id: 42, url: "https://example.com/", title: "Synthetic startup page" };
    },
    onRemoved: { addListener(listener) { listeners.tabRemoved = listener; } },
  },
  runtime: {
    onInstalled: { addListener(listener) { listeners.installed = listener; } },
    onStartup: { addListener(listener) { listeners.startup = listener; } },
  },
  action: { onClicked: { addListener(listener) { listeners.clicked = listener; } } },
  webNavigation: { onCommitted: { addListener(listener) { listeners.committed = listener; } } },
  sidePanel: {
    async setPanelBehavior() {},
    async open() {},
  },
};

await import("../../edge-extension/service_worker.js?startup-test");
assert.equal(typeof listeners.startup, "function", "service worker must register onStartup");
await listeners.startup();
await new Promise((resolve) => setTimeout(resolve, 100));

const audit = await fetch("http://127.0.0.1:8765/debug/last").then((response) => response.json());
assert.ok(audit.request_count >= 1, "startup handler must reach localhost predictor");
assert.equal(audit.last.source, "edge-extension");
assert.equal(audit.last.reason, "browser_startup");
assert.equal(audit.last.network_calls, 0);
console.log("extension startup hook test passed", JSON.stringify({ reason: audit.last.reason, events: audit.last.events_count }));
