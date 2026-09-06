#!/usr/bin/env python3
"""Generate a deterministic Browser Observation semantic benchmark."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DOMAINS = [
    "ecommerce", "search", "news", "github", "email", "social", "admin", "noise"
]

PRODUCTS = [
    ("MacBook Pro 14", "M5 Pro", "$1,999"),
    ("iPhone 17 Pro", "A19 Pro", "$1,199"),
    ("ThinkPad X1 Carbon", "Core Ultra 7", "$1,649"),
    ("Sony WH-1000XM6", "ANC headphones", "$449"),
]

def elements(domain: str, i: int) -> list[dict]:
    if domain == "ecommerce":
        name, attr, price = PRODUCTS[i % len(PRODUCTS)]
        return [
            {"id": 1, "role": "heading", "text": name},
            {"id": 2, "role": "text", "text": attr},
            {"id": 3, "role": "text", "text": price},
            {"id": 4, "role": "button", "text": "Add to cart"},
            {"id": 5, "role": "button", "text": "Buy now"},
        ]
    if domain == "search":
        q = ["wireless mechanical keyboard", "best local LLM GPU", "RTX 4070 Ti SUPER"] [i % 3]
        return [
            {"id": 1, "role": "textbox", "text": q},
            {"id": 2, "role": "button", "text": "Search"},
            {"id": 3, "role": "link", "text": "Result 1: A practical guide"},
            {"id": 4, "role": "link", "text": "Result 2: Independent comparison"},
        ]
    if domain == "news":
        return [
            {"id": 1, "role": "heading", "text": "NVIDIA announces a new efficient GPU"},
            {"id": 2, "role": "text", "text": "Technology · 2 hours ago"},
            {"id": 3, "role": "text", "text": "The report covers performance and power efficiency."},
            {"id": 4, "role": "button", "text": "Share"},
        ]
    if domain == "github":
        return [
            {"id": 1, "role": "heading", "text": "ifm-ai/uno"},
            {"id": 2, "role": "text", "text": "Unlocking Lossless Speedups in LLMs via Discrete Diffusion"},
            {"id": 3, "role": "link", "text": "Issues"},
            {"id": 4, "role": "button", "text": "Star"},
            {"id": 5, "role": "button", "text": "Fork"},
        ]
    if domain == "email":
        return [
            {"id": 1, "role": "heading", "text": "Deployment results for browser semantic layer"},
            {"id": 2, "role": "text", "text": "From: research@example.com"},
            {"id": 3, "role": "text", "text": "Please review the latency table before Friday."},
            {"id": 4, "role": "button", "text": "Reply"},
            {"id": 5, "role": "button", "text": "Archive"},
        ]
    if domain == "social":
        return [
            {"id": 1, "role": "heading", "text": "A small model can be a browser semantic layer"},
            {"id": 2, "role": "text", "text": "128 likes · 23 comments"},
            {"id": 3, "role": "button", "text": "Like"},
            {"id": 4, "role": "button", "text": "Comment"},
        ]
    if domain == "admin":
        return [
            {"id": 1, "role": "heading", "text": "Orders dashboard"},
            {"id": 2, "role": "text", "text": "42 pending · $18,420 revenue"},
            {"id": 3, "role": "button", "text": "Export CSV"},
            {"id": 4, "role": "button", "text": "Filter"},
        ]
    return [
        {"id": 1, "role": "text", "text": "Loading…"},
        {"id": 2, "role": "text", "text": "cookie consent analytics pixel"},
        {"id": 3, "role": "button", "text": "Continue"},
    ]

def entity_label(domain: str, i: int) -> dict:
    if domain == "ecommerce":
        name, attr, price = PRODUCTS[i % len(PRODUCTS)]
        return {"type": "product", "name": name, "attributes": {"spec": attr, "price": price}, "actions": ["add_to_cart", "buy_now"]}
    if domain == "search":
        q = ["wireless mechanical keyboard", "best local LLM GPU", "RTX 4070 Ti SUPER"][i % 3]
        return {"type": "search_query", "name": q, "attributes": {}, "actions": ["search"]}
    if domain == "news":
        return {"type": "article", "name": "NVIDIA announces a new efficient GPU", "attributes": {"section": "Technology"}, "actions": ["share"]}
    if domain == "github":
        return {"type": "repository", "name": "ifm-ai/uno", "attributes": {"topic": "discrete diffusion"}, "actions": ["open_issues", "star", "fork"]}
    if domain == "email":
        return {"type": "email", "name": "Deployment results for browser semantic layer", "attributes": {"sender": "research@example.com"}, "actions": ["reply", "archive"]}
    if domain == "social":
        return {"type": "post", "name": "A small model can be a browser semantic layer", "attributes": {"likes": "128"}, "actions": ["like", "comment"]}
    if domain == "admin":
        return {"type": "dashboard", "name": "Orders dashboard", "attributes": {"pending": "42", "revenue": "$18,420"}, "actions": ["export", "filter"]}
    return {"type": "noise", "name": "Loading…", "attributes": {}, "actions": ["continue"]}

def trajectory(domain: str, i: int, n: int) -> tuple[list[dict], dict]:
    if domain == "ecommerce":
        events = [
            {"t": 0, "action": "input", "target": "search_box", "value": "iPhone 17 Pro"},
            {"t": 1, "action": "click", "target": "search_button"},
            {"t": 2, "action": "click", "target": "iPhone 17 Pro"},
            {"t": 3, "action": "scroll", "target": "specifications"},
            {"t": 4, "action": "back", "target": "browser"},
            {"t": 5, "action": "click", "target": "MacBook Pro 14"},
        ]
        label = {"intent": "compare_products", "entities": ["iPhone 17 Pro", "MacBook Pro 14"], "steps": ["search_product", "inspect_product_a", "inspect_specs", "inspect_product_b"]}
    elif domain == "search":
        events = [
            {"t": 0, "action": "input", "target": "search_box", "value": "local LLM GPU"},
            {"t": 1, "action": "click", "target": "search_button"},
            {"t": 2, "action": "click", "target": "Result 1"},
            {"t": 3, "action": "back", "target": "browser"},
            {"t": 4, "action": "click", "target": "Result 2"},
        ]
        label = {"intent": "research_topic", "entities": ["local LLM GPU", "Result 1", "Result 2"], "steps": ["search", "open_result", "return", "open_result"]}
    elif domain == "github":
        events = [
            {"t": 0, "action": "open", "target": "ifm-ai/uno"},
            {"t": 1, "action": "click", "target": "Issues"},
            {"t": 2, "action": "click", "target": "issue_42"},
            {"t": 3, "action": "back", "target": "repository"},
            {"t": 4, "action": "click", "target": "Star"},
        ]
        label = {"intent": "inspect_repository", "entities": ["ifm-ai/uno", "issue_42"], "steps": ["open_repository", "open_issues", "inspect_issue", "star_repository"]}
    elif domain == "email":
        events = [
            {"t": 0, "action": "open", "target": "deployment results email"},
            {"t": 1, "action": "click", "target": "Reply"},
            {"t": 2, "action": "input", "target": "composer", "value": "Reviewed; latency table looks good."},
            {"t": 3, "action": "click", "target": "Send"},
        ]
        label = {"intent": "reply_to_email", "entities": ["deployment results email"], "steps": ["open_email", "compose_reply", "send_reply"]}
    elif domain == "admin":
        events = [
            {"t": 0, "action": "open", "target": "Orders dashboard"},
            {"t": 1, "action": "click", "target": "Filter"},
            {"t": 2, "action": "input", "target": "status", "value": "pending"},
            {"t": 3, "action": "click", "target": "Apply"},
            {"t": 4, "action": "click", "target": "Export CSV"},
        ]
        label = {"intent": "filter_and_export", "entities": ["Orders dashboard", "pending"], "steps": ["open_dashboard", "open_filter", "set_filter", "apply_filter", "export"]}
    elif domain == "social":
        events = [
            {"t": 0, "action": "open", "target": "semantic layer post"},
            {"t": 1, "action": "click", "target": "Like"},
            {"t": 2, "action": "click", "target": "Comment"},
            {"t": 3, "action": "input", "target": "comment_box", "value": "Interesting benchmark."},
        ]
        label = {"intent": "engage_with_post", "entities": ["semantic layer post"], "steps": ["open_post", "like_post", "open_comments", "write_comment"]}
    elif domain == "news":
        events = [
            {"t": 0, "action": "open", "target": "NVIDIA article"},
            {"t": 1, "action": "scroll", "target": "article_body"},
            {"t": 2, "action": "click", "target": "Share"},
        ]
        label = {"intent": "read_and_share_article", "entities": ["NVIDIA article"], "steps": ["open_article", "read_article", "share_article"]}
    else:
        events = [
            {"t": 0, "action": "wait", "target": "page"},
            {"t": 1, "action": "dismiss", "target": "cookie_banner"},
            {"t": 2, "action": "click", "target": "Continue"},
        ]
        label = {"intent": "dismiss_noise", "entities": ["cookie_banner"], "steps": ["wait", "dismiss_banner", "continue"]}
    # Repeat harmless scroll/wait events to test long-context degradation.
    while len(events) < n:
        events.append({"t": len(events), "action": "scroll", "target": "page_footer" if len(events) % 2 else "page_body"})
    return events[:n], label

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    lengths = [5, 10, 20, 50, 100, 200]
    for i in range(args.count):
        domain = DOMAINS[i % len(DOMAINS)]
        obs = {"url": f"https://example.test/{domain}/{i}", "title": elements(domain, i)[0]["text"], "elements": elements(domain, i)}
        ent = entity_label(domain, i)
        n = lengths[i % len(lengths)]
        ev, traj = trajectory(domain, i, n)
        rows.append({"id": i, "domain": domain, "observation": obs, "entity_gold": {"page_type": domain, "entities": [ent]}, "events": ev, "trajectory_gold": traj, "trajectory_length": n})
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "count": len(rows), "seed": args.seed, "domains": DOMAINS}, ensure_ascii=False))

if __name__ == "__main__":
    main()
