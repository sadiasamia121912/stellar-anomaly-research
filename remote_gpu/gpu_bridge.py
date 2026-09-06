"""
Jupyter-terminal bridge for driving a vast.ai GPU box over HTTPS when SSH is blocked.

Reads connection info (base_url, token) from gpu_conn.json sitting next to this file,
so credentials never need to be retyped/re-pasted into commands or chat.

Usage:
    python gpu_bridge.py run "nvidia-smi"
    python gpu_bridge.py run "cd /workspace && ls -la" --timeout 60
    python gpu_bridge.py upload <local_path> <remote_path>
    python gpu_bridge.py download <remote_path> <local_path>
"""
import sys
import os
import json
import time
import uuid
import base64
import argparse
import re

import requests
import websocket  # websocket-client

HERE = os.path.dirname(os.path.abspath(__file__))
CONN_PATH = os.path.join(HERE, "gpu_conn.json")

with open(CONN_PATH) as f:
    _conn = json.load(f)

BASE_URL = _conn["base_url"].rstrip("/")
TOKEN = _conn["token"]
WS_BASE = BASE_URL.replace("https://", "wss://").replace("http://", "ws://")


def _api(path):
    return f"{BASE_URL}{path}?token={TOKEN}"


def create_terminal():
    r = requests.post(_api("/api/terminals"), timeout=30)
    r.raise_for_status()
    return r.json()["name"]


def delete_terminal(name):
    try:
        requests.delete(_api(f"/api/terminals/{name}"), timeout=15)
    except Exception:
        pass


def _find_real_marker_end(text, marker):
    """
    The PTY echoes the typed 'cmd; echo MARKER\\r' input back as stdout (sometimes more
    than once, e.g. once raw + once after bracketed-paste-mode processing), so MARKER
    shows up embedded in "echo MARKER" text before the command ever actually runs.
    The one real occurrence -- the command's own printed output -- is the only one NOT
    immediately preceded by "echo ". Returns (echo_end, real_start) or None if not seen yet.
    """
    last_echo_end = None
    for m in re.finditer(re.escape(marker), text):
        preceding = text[max(0, m.start() - 5):m.start()]
        if preceding == "echo ":
            last_echo_end = m.end()
        else:
            if last_echo_end is not None:
                return last_echo_end, m.start()
    return None


def run_command(cmd, timeout=120, term_name=None):
    """Run one shell command in a fresh (or given) PTY, return its stdout output."""
    owns_terminal = term_name is None
    if term_name is None:
        term_name = create_terminal()

    ws_url = f"{WS_BASE}/terminals/websocket/{term_name}?token={TOKEN}"
    ws = websocket.create_connection(ws_url, timeout=timeout)

    marker = f"__DONE_{uuid.uuid4().hex}__"
    full_cmd = f"{cmd}; echo {marker}\r"

    output_chunks = []
    start = time.time()

    ws.settimeout(2)
    ws.send(json.dumps(["stdin", full_cmd]))

    body = None
    try:
        while time.time() - start < timeout:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not msg:
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if len(data) >= 2 and data[0] == "stdout":
                output_chunks.append(data[1])
                full_output = "".join(output_chunks)
                hit = _find_real_marker_end(full_output, marker)
                if hit is not None:
                    echo_end, real_start = hit
                    body = full_output[echo_end:real_start]
                    break
    finally:
        ws.close()
        if owns_terminal:
            delete_terminal(term_name)

    if body is None:
        # Timed out without a clean marker match -- return what we have for debugging.
        body = "".join(output_chunks)
    return body.strip("\r\n")


def upload_file(local_path, remote_path, chunk_size=15 * 1024 * 1024, progress=True):
    """Upload via Jupyter's Contents API, chunked (raw bytes per chunk before
    base64) so large files don't blow past any single-request body limit on the
    tunnel. Chunk numbering per Jupyter's protocol: 1, 2, ... , final chunk = -1."""
    size = os.path.getsize(local_path)
    n_chunks = max(1, (size + chunk_size - 1) // chunk_size)
    with open(local_path, "rb") as f:
        idx = 0
        while True:
            raw = f.read(chunk_size)
            if not raw:
                break
            idx += 1
            is_last = f.tell() >= size
            body = {
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(raw).decode("ascii"),
                "chunk": -1 if is_last else idx,
            }
            r = requests.put(_api(f"/api/contents/{remote_path.lstrip('/')}"), json=body, timeout=300)
            r.raise_for_status()
            if progress:
                print(f"  uploaded chunk {idx}/{n_chunks} ({f.tell()/1e6:.1f}/{size/1e6:.1f} MB)", file=sys.stderr)
    return {"path": remote_path, "size": size}


def download_file(remote_path, local_path):
    r = requests.get(_api(f"/api/contents/{remote_path.lstrip('/')}"), timeout=300)
    r.raise_for_status()
    data = r.json()
    if data.get("format") == "base64":
        content = base64.b64decode(data["content"])
        mode = "wb"
    else:
        content = data["content"].encode("utf-8")
        mode = "wb"
    with open(local_path, mode) as f:
        f.write(content)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("cmd")
    p_run.add_argument("--timeout", type=int, default=120)

    p_up = sub.add_parser("upload")
    p_up.add_argument("local_path")
    p_up.add_argument("remote_path")

    p_down = sub.add_parser("download")
    p_down.add_argument("remote_path")
    p_down.add_argument("local_path")

    args = ap.parse_args()

    if args.action == "run":
        print(run_command(args.cmd, timeout=args.timeout))
    elif args.action == "upload":
        print(upload_file(args.local_path, args.remote_path))
    elif args.action == "download":
        download_file(args.remote_path, args.local_path)
        print(f"saved to {args.local_path}")
