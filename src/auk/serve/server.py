"""Minimal HTTP front-ends for the two AuK nodes.

Stdlib ``http.server`` only — no new dependency. Payloads are a few MB and the call rate is a
handful per generation, so anything heavier (gRPC, FastAPI) would add deps without buying
throughput.

Requests are served by a thread pool but inference is serialised behind a lock:
``Flux2Edit.text_cond`` / ``text_uncond`` are *module-level* state reused via ``cache=True``, so
two overlapping CFG samples would silently reuse each other's text projection.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from auk.serve.codec import pack, unpack

logger = logging.getLogger(__name__)


def _make_handler(node, lock: threading.Lock, kind: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "auk-serve/0.1"

        def log_message(self, fmt, *args):  # route through logging instead of stderr
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes = b"", content_type: str = "application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _fail(self, code: int, message: str):
            logger.warning("%s %s -> %s: %s", self.command, self.path, code, message)
            self._send(code, json.dumps({"error": message}).encode(), "application/json")

        def do_GET(self):
            route = urlparse(self.path).path.rstrip("/") or "/"
            if route == "/health":
                self._send(200, b'{"status":"ok"}', "application/json")
            elif route == "/info":
                info = node.info() if hasattr(node, "info") else {"kind": kind}
                self._send(200, json.dumps(info).encode(), "application/json")
            else:
                self._fail(404, f"unknown route {route}")

        def do_POST(self):
            route = urlparse(self.path).path.rstrip("/") or "/"
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    raise ValueError("missing Content-Length")
                header, tensors = unpack(self.rfile.read(length))
            except Exception as exc:  # noqa: BLE001 - report any framing error to the client
                self._fail(400, f"bad frame: {exc}")
                return

            try:
                with lock:
                    if kind == "text-encoder" and route == "/encode":
                        body = node.encode(tensors)
                    elif kind == "worker" and route == "/generate":
                        body = _worker_generate(node, header, tensors)
                    else:
                        self._fail(404, f"unknown route {route}")
                        return
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s %s failed", self.command, route)
                self._fail(500, f"{type(exc).__name__}: {exc}")
                return

            self._send(200, body)

    return Handler


def _worker_generate(node, header: dict, tensors: dict) -> bytes:
    audio, sample_rate = node.generate(
        tensors.get("ref_audio"),
        tensors["hidden"],
        tensors["attention_mask"],
        gen_latent_len=int(header["gen_latent_len"]),
        nfe=int(header.get("nfe", 32)),
        cfg_strength=float(header.get("cfg_strength", 2.0)),
        sway_sampling_coef=header.get("sway_sampling_coef", -1.0),
        t_grid=header.get("t_grid"),
        seed=header.get("seed"),
        sample_rate=int(header.get("sample_rate", 0)),
    )
    return pack({"sample_rate": sample_rate, "shape": list(audio.shape)}, {"audio": audio})


def serve(node, kind: str, host: str = "0.0.0.0", port: int = 8000):
    """Blocking. ``kind`` is ``"text-encoder"`` or ``"worker"``."""
    lock = threading.Lock()
    httpd = ThreadingHTTPServer((host, port), _make_handler(node, lock, kind))
    logger.info("AuK %s node listening on http://%s:%d", kind, host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
