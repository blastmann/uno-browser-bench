#!/usr/bin/env python3
"""Local-only predictor service for the Edge extension.

This MVP deliberately has no network model call.  It provides a deterministic
heuristic baseline and the stable JSON protocol that NanoJev/decider adapters
can implement later.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


INTENTS = [
    "research_topic",
    "inspect_repository",
    "compare_products",
    "reply_to_email",
    "filter_and_export",
    "engage_with_post",
    "read_and_share_article",
    "dismiss_noise",
    "manage_settings",
]

KEYWORDS = {
    "research_topic": ("search", "result", "research", "query", "documentation", "guide", "搜索", "研究"),
    "inspect_repository": ("github", "repository", "repo", "issue", "readme", "release", "代码", "仓库"),
    "compare_products": ("product", "price", "spec", "review", "compare", "cart", "商品", "价格", "比较"),
    "reply_to_email": ("email", "inbox", "sender", "reply", "message", "mail", "邮件", "回复"),
    "filter_and_export": ("dashboard", "filter", "export", "csv", "report", "table", "报表", "导出"),
    "engage_with_post": ("post", "comment", "like", "social", "share", "社区", "评论"),
    "read_and_share_article": ("article", "news", "byline", "read", "save", "新闻", "文章"),
    "dismiss_noise": ("cookie", "notice", "consent", "loading", "interstitial", "banner", "弹窗"),
    "manage_settings": ("settings", "privacy", "options", "toggle", "permission", "设置", "隐私"),
}

_AUDIT_LOCK = threading.Lock()
_AUDIT = {"request_count": 0, "last": None}


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exp = [math.exp(value - peak) for value in values]
    total = sum(exp)
    return [value / total for value in exp]


def text_of(observation: dict[str, Any]) -> str:
    pieces = [observation.get("title", ""), observation.get("url", "")]
    for element in observation.get("elements", []) or []:
        pieces.extend([element.get("role", ""), element.get("text", ""), element.get("id", "")])
    for event in observation.get("events", []) or []:
        pieces.extend([event.get("action", ""), event.get("target", ""), event.get("value", "")])
    return " ".join(str(piece).lower() for piece in pieces)


def predict(observation: dict[str, Any]) -> dict[str, Any]:
    text = text_of(observation)
    scores = []
    for intent in INTENTS:
        hits = sum(len(re.findall(re.escape(keyword.lower()), text)) for keyword in KEYWORDS[intent])
        scores.append(0.15 * hits)
    # Avoid an overconfident zero-feature answer; unknown is represented by a
    # diffuse distribution and can be added by a future typed-decision model.
    if max(scores, default=0.0) == 0.0:
        scores = [0.0 for _ in scores]
    probabilities = dict(zip(INTENTS, softmax(scores)))
    choice = max(probabilities, key=probabilities.get)
    return {
        "model": "heuristic-v0",
        "answers": {"intent": {"type": "choice", "choice": choice, "confidence": probabilities[choice], "probabilities": probabilities}},
        "intent": choice,
        "probabilities": probabilities,
        "privacy": {"network_calls": 0, "bind": "127.0.0.1", "input": "redacted observation"},
    }


def record_audit(payload: dict[str, Any], observation: dict[str, Any], result: dict[str, Any]) -> None:
    """Keep only non-content protocol evidence for local runtime verification."""
    with _AUDIT_LOCK:
        _AUDIT["request_count"] += 1
        _AUDIT["last"] = {
            "received_at": time.time(),
            "source": payload.get("source"),
            "reason": payload.get("reason"),
            "observation_keys": sorted(str(key) for key in observation.keys()),
            "elements_count": len(observation.get("elements", []) or []),
            "events_count": len(observation.get("events", []) or []),
            "has_title": bool(observation.get("title")),
            "has_url": bool(observation.get("url")),
            "prediction": result.get("intent"),
            "network_calls": 0,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "BrowserIntentLocal/0.1"

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._json(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._json(200, {"ok": True, "model": "heuristic-v0", "bind": "127.0.0.1"})
            return
        if self.path == "/debug/last":
            with _AUDIT_LOCK:
                self._json(200, {"request_count": _AUDIT["request_count"], "last": _AUDIT["last"], "privacy": "content-free local audit"})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/predict":
            self._json(404, {"error": "not_found"})
            return
        try:
            size = int(self.headers.get("content-length", "0"))
            if size > 2_000_000:
                raise ValueError("request_too_large")
            payload = json.loads(self.rfile.read(size))
            observation = payload.get("observation", payload if isinstance(payload, dict) else {})
            result = predict(observation)
            record_audit(payload, observation, result)
            result["updatedAt"] = payload.get("createdAt")
            self._json(200, result)
        except Exception as error:  # keep the extension protocol deterministic
            self._json(400, {"error": "invalid_request", "detail": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("The MVP intentionally binds only to 127.0.0.1")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "model": "heuristic-v0"}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
