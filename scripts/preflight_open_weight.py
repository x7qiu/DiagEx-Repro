"""Read public model/weight metadata and initialize a bounded evaluation ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.model_policy import OPEN_WEIGHT_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=40, follow_redirects=True)

    def get(url):
        response = client.get(url)
        response.raise_for_status()
        return response

    models = {m["id"]: m for m in get("https://openrouter.ai/api/v1/models").json()["data"]}
    verified = {}
    for slug in [*OPEN_WEIGHT_MODELS, "anthropic/claude-opus-4.7"]:
        if slug not in models:
            continue
        m = models[slug]
        if "image" not in m["architecture"]["input_modalities"]:
            continue
        endpoints = get("https://openrouter.ai/api/v1/models/" + slug + "/endpoints").json()[
            "data"
        ]["endpoints"]
        endpoints = [
            e
            for e in endpoints
            if {"tools", "tool_choice"} <= set(e.get("supported_parameters", []))
        ]
        if not endpoints:
            continue
        weight = {}
        if slug in OPEN_WEIGHT_MODELS:
            repo = OPEN_WEIGHT_MODELS[slug]
            repo_id = repo.removeprefix("https://huggingface.co/")
            info = get("https://huggingface.co/api/models/" + repo_id).json()
            revision = info["sha"]
            license_response = get(repo + "/resolve/" + revision + "/LICENSE")
            card = get(repo + "/resolve/" + revision + "/README.md").text
            license_text = license_response.text
            if "Apache License" not in license_text or "Version 2.0" not in license_text:
                raise ValueError("Unverified commercial license: " + slug)
            if not any(f["rfilename"].endswith(".safetensors") for f in info.get("siblings", [])):
                raise ValueError("No published weight files: " + slug)
            prefix = args.out / slug.replace("/", "--")
            Path(str(prefix) + ".LICENSE.txt").write_text(license_text)
            Path(str(prefix) + ".model-card.md").write_text(card)
            weight = {
                "weights_url": repo,
                "revision": revision,
                "license": "Apache-2.0",
                "license_sha256": hashlib.sha256(license_response.content).hexdigest(),
                "license_url": repo + "/blob/" + revision + "/LICENSE",
            }
        atomic_write_json(args.out / (slug.replace("/", "--") + ".endpoints.json"), endpoints)
        input_keys = ("prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h")
        verified[slug] = {
            **weight,
            "verified_at": time.time(),
            "valid_until": time.time() + 86400,
            "source": "https://openrouter.ai/api/v1/models/" + slug + "/endpoints",
            "context_length": max(e["context_length"] for e in endpoints),
            "input_per_token": max(
                float(e["pricing"].get(k, 0)) for e in endpoints for k in input_keys
            ),
            "output_per_token": max(float(e["pricing"]["completion"]) for e in endpoints),
            "request_usd": max(float(e["pricing"].get("request", 0)) for e in endpoints),
            "provider_tags": [e["tag"] for e in endpoints],
            "capabilities": {"image": True, "tools": True, "live_smoke_verified": False},
        }
    atomic_write_json(args.out / "verified-prices.json", {"models": verified})
    ledger = args.out / "spending.json"
    if not ledger.exists():
        atomic_write_json(
            ledger,
            {
                "limit_usd": 50,
                "category_limits": {"extraction": 20, "context": 10, "graph": 15, "reserve": 5},
                "requests": [],
            },
        )
    print(
        json.dumps({"verified_models": list(verified), "ledger": str(ledger), "inference_calls": 0})
    )


if __name__ == "__main__":
    main()
