"""Run the real benchmark CLI against a deterministic local synthetic provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MODEL_NAME = "ragcv-deterministic-smoke-v1"


def _assistant_content(user_prompt: str) -> str:
    is_repair = "The previous response was invalid." in user_prompt
    claim = user_prompt.split("Claim:\n", 1)[1].split("\n\nEvidence:", 1)[0]
    if "exactly 17 fastest laps" in claim and not is_repair:
        return "not valid json"

    document_id = ""
    if "1994" in claim:
        label = "REFUTED" if "for Ferrari" in claim else "SUPPORTED"
        document_id = "synthetic_doc_1994"
    elif "Mercedes in 2010" in claim:
        label = "SUPPORTED"
        document_id = "synthetic_doc_2010"
    elif "McLaren in 1991" in claim:
        label = "REFUTED"
        document_id = "synthetic_doc_1991"
    else:
        label = "NOT_ENOUGH_EVIDENCE"

    baseline = "BASELINE_WITHOUT_RETRIEVAL" in user_prompt
    citation_is_available = f'"document_id": "{document_id}"' in user_prompt
    citations = [document_id] if document_id and not baseline and citation_is_available else []
    return json.dumps(
        {
            "label": label,
            "reason": "Deterministic synthetic smoke-fixture decision.",
            "cited_document_ids": citations,
        },
        separators=(",", ":"),
    )


class SmokeHandler(BaseHTTPRequestHandler):
    """Serve the subset of Chat Completions used by the production HTTP client."""

    server_version = "RAGCVSmoke/1.0"

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            messages = payload["messages"]
            system_prompt = str(messages[0]["content"])
            user_prompt = str(messages[1]["content"])
            content = _assistant_content(user_prompt)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self.send_error(400, explain=str(exc))
            return

        request_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]
        prompt_tokens = len((system_prompt + " " + user_prompt).split())
        completion_tokens = len(content.split())
        response: dict[str, Any] = {
            "id": f"smoke-{request_hash}",
            "object": "chat.completion",
            "model": f"{MODEL_NAME}-revision-1",
            "system_fingerprint": "ragcv-smoke-fixture-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic benchmark through the production CLI, HTTP client, verifier, "
            "retrievers, persistence, and evaluator."
        )
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer((HOST, arguments.port), SmokeHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    environment = os.environ.copy()
    environment["RAGCV_SMOKE_BASE_URL"] = f"http://{HOST}:{arguments.port}/v1"
    command = [
        sys.executable,
        "-m",
        "rag_claim_verification",
        "benchmark",
        "--config",
        "configs/smoke_benchmark.yaml",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
