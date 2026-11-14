"""Serve the photo survey and record responses (Python standard library only).

    python docs/user_study/site/run_survey_server.py --port 8000

Participants open http://<this computer's address>:8000/ . Each answered photo is appended to
responses.jsonl next to this file as soon as it is submitted, so a participant who stops halfway still
leaves the photos already answered. The manifest that maps photos to conditions stays outside the
served directory and is never sent to participants.
"""
import argparse
import json
import secrets
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
RESPONSES = HERE / "responses.jsonl"
PHOTOS = sorted(p.name for p in (STATIC / "photos").glob("photo_*.jpg"))
LOCK = threading.Lock()
FIELDS = {"share", "normal"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/start":
            return self._json(200, {"participant": secrets.token_hex(6), "photos": PHOTOS})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/answer":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 10_000:
            return self._json(400, {"error": "bad length"})
        data = json.loads(self.rfile.read(length))
        kind = data.get("kind")
        if kind == "answer":
            if data.get("photo") not in PHOTOS:
                return self._json(400, {"error": "unknown photo"})
            scores = data.get("scores", {})
            if set(scores) != FIELDS or not all(isinstance(v, int) and 1 <= v <= 5 for v in scores.values()):
                return self._json(400, {"error": "scores must be integers 1-5 for every field"})
        elif kind not in {"profile", "finish"}:
            return self._json(400, {"error": "unknown kind"})
        data["received_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LOCK, RESPONSES.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        return self._json(200, {"ok": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if len(PHOTOS) != 20:
        raise SystemExit(f"expected 20 photos in {STATIC / 'photos'}, found {len(PHOTOS)}")
    print(f"Serving {len(PHOTOS)} photos on http://{args.host}:{args.port}/ ; responses -> {RESPONSES}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
