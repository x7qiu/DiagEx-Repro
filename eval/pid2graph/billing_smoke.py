"""One bounded source-only training crop request to verify live billing capture."""
import argparse
import base64
import json
import os
import time
from pathlib import Path

from diagex.config import LLMConfig
from diagex.llm.billing import summarize
from diagex.llm.client import LLMClient

from .connection_accounting import configure as configure_connection_accounting
from .data import save_new, sha256
from .epoch_train import sealed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "ledger", "prices", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Use a new smoke output; never silently repeat a paid request")
    data = sealed(a.training, "training_sha256")
    row = data["samples"][0]
    source = next(s for s in data["sources"] if s["id"] == row["source_id"])
    if source["split"] != "train":
        raise ValueError("Billing smoke must use a training source")
    image = a.training.parent / row["image"]
    if sha256(image) != row["image_sha256"]:
        raise ValueError("Crop changed")
    a.out.mkdir(parents=True)
    config = {"source_id": row["source_id"], "image_sha256": row["image_sha256"], "split": "train",
              "training_sha256": data["training_sha256"], "model": "deepseek/deepseek-v4.1-flash",
              "max_tokens": 512, "max_attempts": 1, "time_budget_s": 90,
              "price_sha256": sha256(a.prices), "purpose": "billing/transport verification; excluded from accuracy comparisons"}
    save_new(a.out / "config.json", config)
    client = LLMClient(LLMConfig(transport="openrouter", model=config["model"], reasoning_mode="disabled",
                       openrouter_api_key=os.environ["OPENROUTER_API_KEY"], spending_ledger=str(a.ledger.resolve()),
                       verified_prices=str(a.prices.resolve()), spending_category="billing_smoke"))
    configure_connection_accounting(client)
    before = {r["id"] for r in json.loads(a.ledger.read_text())["requests"]}
    started = time.time()
    try:
        response = client.messages_create(
            system="Inspect the supplied P&ID crop and briefly describe the visible symbol shapes. Use the submit tool.",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                    "data": base64.b64encode(image.read_bytes()).decode()}},
                {"type": "text", "text": "Describe only what is visible in this crop."}]}],
            tools=[{"name": "submit", "description": "Return a brief visual description", "input_schema": {
                "type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"],
                "additionalProperties": False}}], tool_choice={"type": "tool", "name": "submit"},
            max_tokens=config["max_tokens"], max_attempts=1, time_budget_s=90)
        save_new(a.out / "response.json", response.model_dump(mode="json"))
        outcome = {"status": "response", "response_id": response.id}
    except Exception as exc:
        outcome = {"status": "error", "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)}
    state = json.loads(a.ledger.read_text())
    ids = [r["id"] for r in state["requests"] if r["id"] not in before and r["category"] == "billing_smoke"]
    result = {**outcome, "elapsed_seconds": time.time() - started, "ledger_request_ids": ids,
              "billing": summarize(state, ids)}
    save_new(a.out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
