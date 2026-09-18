#!/usr/bin/env python3
"""Generate a deterministic, leakage-controlled Browser Observation benchmark.

The legacy dataset labels the completed trajectory and pads long examples with
filler scrolls.  This generator keeps observation input separate from labels,
uses varied templates, and adds a session-level next_intent target for the
browser-startup prediction experiment.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
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
LENGTHS = (5, 10, 20, 50, 100, 200)


SCENARIOS = [
    ("search", "search_a", "research_topic", "Local inference latency", "search_query", ["search", "open_result", "compare_sources"]),
    ("search", "search_b", "research_topic", "Windows GPU benchmark", "search_query", ["search", "open_result", "save_reference"]),
    ("search", "search_c", "research_topic", "Browser extension permissions", "search_query", ["search", "open_result", "read_document"]),
    ("github", "repo_a", "inspect_repository", "open-source/browser-intent", "repository", ["open_repository", "open_issues", "inspect_issue", "inspect_code"]),
    ("github", "repo_b", "inspect_repository", "ifm-ai/uno", "repository", ["open_repository", "inspect_readme", "open_release", "copy_clone_command"]),
    ("github", "repo_c", "inspect_repository", "Mapika/decider", "repository", ["open_repository", "inspect_readme", "open_issue", "inspect_source"]),
    ("ecommerce", "shop_a", "compare_products", "Noise-cancelling headphones", "product", ["search_product", "open_product_a", "inspect_specs", "open_product_b", "compare_prices"]),
    ("ecommerce", "shop_b", "compare_products", "Developer laptops", "product", ["search_product", "open_product_a", "inspect_specs", "open_product_b", "compare_reviews"]),
    ("ecommerce", "shop_c", "compare_products", "USB microphones", "product", ["search_product", "open_product_a", "filter_products", "open_product_b", "compare_prices"]),
    ("email", "mail_a", "reply_to_email", "Deployment results", "email", ["open_email", "read_email", "compose_reply", "send_reply"]),
    ("email", "mail_b", "reply_to_email", "Review request", "email", ["open_email", "read_attachment", "compose_reply", "send_reply"]),
    ("email", "mail_c", "reply_to_email", "Meeting follow-up", "email", ["open_email", "read_email", "compose_reply", "send_reply"]),
    ("admin", "admin_a", "filter_and_export", "Orders dashboard", "dashboard", ["open_dashboard", "open_filter", "set_filter", "apply_filter", "export"]),
    ("admin", "admin_b", "filter_and_export", "Usage report", "dashboard", ["open_dashboard", "choose_date_range", "apply_filter", "export"]),
    ("admin", "admin_c", "filter_and_export", "Error inventory", "dashboard", ["open_dashboard", "set_status", "sort_table", "export"]),
    ("social", "social_a", "engage_with_post", "A small browser model", "post", ["open_post", "read_post", "open_comments", "write_comment"]),
    ("social", "social_b", "engage_with_post", "Local AI deployment", "post", ["open_post", "like_post", "open_comments", "write_comment"]),
    ("social", "social_c", "engage_with_post", "Extension privacy design", "post", ["open_post", "read_post", "share_post"]),
    ("news", "news_a", "read_and_share_article", "Efficient GPU architecture", "article", ["open_article", "read_article", "open_related", "share_article"]),
    ("news", "news_b", "read_and_share_article", "Browser automation research", "article", ["open_article", "read_article", "save_article", "share_article"]),
    ("news", "news_c", "read_and_share_article", "New developer tools", "article", ["open_article", "read_article", "share_article"]),
    ("noise", "noise_a", "dismiss_noise", "Consent prompt", "banner", ["wait", "dismiss_banner", "continue"]),
    ("noise", "noise_b", "dismiss_noise", "Interstitial notice", "banner", ["wait", "close_notice", "continue"]),
    ("settings", "settings_a", "manage_settings", "Browser privacy settings", "settings", ["open_settings", "change_setting", "save_settings"]),
    ("settings", "settings_b", "manage_settings", "Extension options", "settings", ["open_options", "change_setting", "save_settings"]),
]


def observation(domain: str, title: str, sample_id: int, variant: int) -> dict[str, Any]:
    # URLs are deliberately opaque: the intent must be recoverable from the
    # page title/elements and events, not from a label-bearing path.
    texts = {
        "search": ["Search results", "Search box", "Result A", "Result B"],
        "github": ["Code hosting page", "README", "Issues", "Releases", "Files"],
        "ecommerce": ["Product listing", "Price", "Specifications", "Reviews", "Compare"],
        "email": ["Inbox message", "Sender", "Message body", "Reply", "Archive"],
        "admin": ["Data dashboard", "Date range", "Filter", "Export", "Rows"],
        "social": ["Social post", "Author", "Comments", "Like", "Share"],
        "news": ["Article page", "Byline", "Article body", "Save", "Share"],
        "noise": ["Notice", "Continue", "Close", "Privacy", "Loading"],
        "settings": ["Settings page", "Privacy", "Toggle", "Save", "Reset"],
    }[domain]
    return {
        "url": f"https://example.test/page/{sample_id}",
        "title": title,
        "elements": [
            {"id": f"e{variant}-{i}", "role": "heading" if i == 0 else ("button" if i in {2, 3} else "text"), "text": text}
            for i, text in enumerate([title, *texts[1:]], 1)
        ],
    }


def base_events(scenario: tuple, sample_id: int) -> tuple[list[dict[str, Any]], list[str]]:
    domain, template, intent, title, entity_type, steps = scenario
    target = title if sample_id % 2 else f"{title} / option {sample_id % 4 + 1}"
    values = {
        "search": "local browser intent benchmark",
        "ecommerce": "developer laptop comparison",
        "email": "Reviewed the request; I will follow up shortly.",
        "admin": "pending",
        "social": "Useful comparison, saving this for later.",
        "settings": "enabled",
    }
    actions = []
    for index, step in enumerate(steps):
        action = "open" if step.startswith("open_") else "click"
        if step in {"search", "compose_reply", "write_comment", "set_filter", "choose_date_range", "change_setting"}:
            action = "input"
        if step == "read_article" or step == "read_email" or step == "read_attachment" or step == "inspect_code" or step == "inspect_source":
            action = "read"
        target_name = step.removeprefix("open_").removeprefix("inspect_").removeprefix("read_")
        event: dict[str, Any] = {"t": index, "action": action, "target": target_name or step}
        if action == "input":
            event["value"] = values.get(domain, target)
        actions.append(event)
    # Realistic, non-label-bearing navigation noise. It is interspersed and
    # bounded; there is no long filler tail that overwhelms the gold steps.
    noise = [
        {"action": "scroll", "target": "content"},
        {"action": "focus", "target": "main_panel"},
        {"action": "hover", "target": "navigation"},
        {"action": "back", "target": "browser"},
        {"action": "open", "target": "related_panel"},
    ]
    return actions, steps


def make_trace(scenario: tuple, sample_id: int, length: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[str]]:
    events, steps = base_events(scenario, sample_id)
    if length < len(events):
        events = events[:length]
    while len(events) < length:
        item = dict(rng.choice([
            {"action": "scroll", "target": "content"},
            {"action": "focus", "target": "main_panel"},
            {"action": "hover", "target": "navigation"},
            {"action": "open", "target": "related_panel"},
            {"action": "back", "target": "browser"},
        ]))
        item["t"] = len(events)
        events.append(item)
    # Keep step-bearing events at the front for short prefixes, then add
    # neutral activity. This intentionally tests whether confidence degrades
    # when the observation grows without changing the task.
    return events[:length], steps


def next_intent_for(scenario_index: int, sample_id: int) -> str:
    # A deterministic synthetic browser-startup target. It is not derived
    # from the current row's URL; it represents the next session action.
    return SCENARIOS[(scenario_index + 3 + sample_id % 5) % len(SCENARIOS)][2]


def generate(count_per_scenario: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    row_id = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        for sample in range(count_per_scenario):
            # Each scenario has all six lengths, so a model can be evaluated
            # at 5/10/20/50/100/200 events with the same template distribution.
            length = LENGTHS[(sample + scenario_index) % len(LENGTHS)]
            events, steps = make_trace(scenario, row_id, length, rng)
            domain, template, intent, title, entity_type, _ = scenario
            rows.append({
                "id": row_id,
                "source": "synthetic-v2",
                "domain": domain,
                "template_id": template,
                "observation": observation(domain, title, row_id, sample),
                "events": events,
                "entity_gold": {"page_type": domain, "entities": [{"type": entity_type, "name": title, "attributes": {"template": template}, "actions": steps}]},
                "trajectory_gold": {"intent": intent, "entities": [title], "steps": steps},
                "next_intent_gold": next_intent_for(scenario_index, row_id),
                "trajectory_length": length,
                "label_policy": "current_task_and_next_session_task_are_separate",
            })
            row_id += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.count_per_scenario <= 0:
        raise SystemExit("--count-per-scenario must be positive")
    rows = generate(args.count_per_scenario, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "seed": args.seed, "scenarios": len(SCENARIOS), "lengths": LENGTHS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
