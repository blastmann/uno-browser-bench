import assert from "node:assert/strict";
import { redactObservation, redactUrl, redactText } from "../../edge-extension/redact.js";

assert.equal(redactUrl("https://example.com/path/user-123?email=a@example.com#secret"), "https://example.com/path/user-123");
assert.equal(redactText("Call +86 138 1234 5678 or a@example.com order 123456"), "Call <phone> or <email> order <num>");
const value = redactObservation({
  url: "https://mail.example.com/inbox?token=secret",
  title: "Reply to alice@example.com order 123456",
  elements: [{ role: "button", id: "send-123456", text: "Send" }],
  events: [{ action: "input", target: "composer", value: "Phone 13812345678" }],
});
assert.equal(value.url, "https://mail.example.com/inbox");
assert.equal(value.title, "Reply to <email> order <num>");
assert.equal(value.elements[0].id, "send-<num>");
assert.equal(value.events[0].value, "Phone <phone>");
console.log("extension redaction tests passed");
