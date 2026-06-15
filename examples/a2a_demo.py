"""End-to-end A2A demo client for search4people.

Prerequisites:
  1. A user + token:  uv run s4p-create-user demo demopass
                      uv run s4p-create-token demo --label demo
  2. Server running:  uv run s4p-a2a
  3. Export the token: export A2A_DEMO_TOKEN=<token printed above>

Run:
  uv run python examples/a2a_demo.py

Walks the full input-required cycle:
  send name -> input-required (narrowing) -> pick candidate ->
  input-required (confirm) -> approve -> completed (PersonProfile artifact).
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx

BASE_URL = os.environ.get("A2A_BASE_URL", "http://localhost:8001")
TOKEN = os.environ.get("A2A_DEMO_TOKEN", "")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def _text_message(text: str, task_id: str | None, context_id: str | None) -> dict[str, object]:
    msg: dict[str, object] = {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "messageId": os.urandom(8).hex(),
    }
    if task_id:
        msg["taskId"] = task_id
    if context_id:
        msg["contextId"] = context_id
    return msg


def _data_message(data: dict[str, object], task_id: str, context_id: str) -> dict[str, object]:
    return {
        "role": "user",
        "parts": [{"kind": "data", "data": data}],
        "messageId": os.urandom(8).hex(),
        "taskId": task_id,
        "contextId": context_id,
    }


async def _send(client: httpx.AsyncClient, message: dict[str, object]) -> dict[str, object]:
    payload = {
        "jsonrpc": "2.0",
        "id": os.urandom(4).hex(),
        "method": "message/send",
        "params": {"message": message},
    }
    resp = await client.post("/", json=payload, headers=_headers())
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return body["result"]


def _print_task(label: str, task: dict[str, object]) -> None:
    state = task.get("status", {}).get("state")
    print(f"\n=== {label}: state={state} ===")
    status_msg = task.get("status", {}).get("message") or {}
    for part in status_msg.get("parts", []):
        if part.get("kind") == "text":
            print("  text:", part["text"])
        elif part.get("kind") == "data":
            print("  data:", json.dumps(part["data"], ensure_ascii=False)[:600])
    for art in task.get("artifacts", []) or []:
        for part in art.get("parts", []):
            if part.get("kind") == "data":
                print("  artifact:", json.dumps(part["data"], ensure_ascii=False)[:800])


async def stream_demo(client: httpx.AsyncClient) -> None:
    """Show message/stream: print interim events as the graph runs."""
    print("\n=== message/stream (interim events) ===")
    payload = {
        "jsonrpc": "2.0",
        "id": os.urandom(4).hex(),
        "method": "message/stream",
        "params": {"message": _text_message("Grace Hopper", None, None)},
    }
    async with client.stream("POST", "/", json=payload, headers=_headers()) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            envelope = json.loads(line[len("data:"):].strip())
            result = envelope.get("result") or {}
            # Result is a Task or a status/artifact update event; print its state if present.
            status = result.get("status") or {}
            state = status.get("state") or result.get("kind")
            print("  event:", state)


async def main() -> None:
    if not TOKEN:
        raise SystemExit("Set A2A_DEMO_TOKEN (see the module docstring).")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        # 0. Fetch the Agent Card (public, no auth needed).
        card = (await client.get("/.well-known/agent-card.json")).json()
        print("Agent:", card["name"], "- skills:", [s["id"] for s in card["skills"]])

        # 1. Start: send a name.
        task = await _send(client, _text_message("Jane Smith", None, None))
        _print_task("after name", task)
        task_id = task["id"]
        context_id = task.get("contextId")

        state = task["status"]["state"]

        # 2. If narrowing was requested, pick the first candidate.
        if state == "input-required":
            task = await _send(client, _data_message({"pick_index": 0}, task_id, context_id))
            _print_task("after pick", task)
            state = task["status"]["state"]

        # 3. If confirmation was requested, approve.
        if state == "input-required":
            task = await _send(client, _data_message({"decision": "approve"}, task_id, context_id))
            _print_task("after approve", task)

        print("\nFinal state:", task["status"]["state"])
        await stream_demo(client)


if __name__ == "__main__":
    asyncio.run(main())
