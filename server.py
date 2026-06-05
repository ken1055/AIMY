# -*- coding: utf-8 -*-
"""
AIMY ローカル開発用サーバー
- 静的ファイル配信（Xserver の代替）
- Python 画像処理 API（processing.py を使用）

本番環境では app.py (Flask + gunicorn) を使うこと。
"""

import base64
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

try:
    from processing import process_akami, process_face_mask, process_keana, process_shimi, process_shiwa, process_texture
    _PROCESSING_OK = True
except ImportError as e:
    _PROCESSING_OK = False
    _IMPORT_ERROR = str(e)

PORT = 8080


class AimyHandler(SimpleHTTPRequestHandler):
    """静的ファイル配信 + /api/* エンドポイント"""

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if not _PROCESSING_OK:
            self._json_error(500, f"processing.py のインポートエラー: {_IMPORT_ERROR}")
            return

        path = urlparse(self.path).path
        routes = {
            "/api/face-mask": process_face_mask,
            "/api/shimi":     process_shimi,
            "/api/shiwa":     process_shiwa,
            "/api/texture":   process_texture,
            "/api/keana":     process_keana,
            "/api/akami":     process_akami,
        }
        processor = routes.get(path)
        if processor is None:
            self.send_error(404, "Not Found")
            return
        self._handle(processor)

    def _handle(self, processor):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            self._json_error(400, "JSON パースエラー")
            return

        image_b64 = data.get("image", "")
        if not image_b64:
            self._json_error(400, "image フィールドが必要です")
            return
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(image_b64)
        except Exception:
            self._json_error(400, "base64 デコードエラー")
            return

        severity = float(data.get("severity", 1.0))
        try:
            result_bytes = processor(img_bytes, severity=severity)
        except Exception as e:
            self._json_error(500, f"画像処理エラー: {e}")
            return

        b64 = "data:image/png;base64," + base64.b64encode(result_bytes).decode()
        self._json_ok({"result": b64})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_ok(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, msg):
        body = json.dumps({"error": msg}, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/" in str(args[0] if args else ""):
            print(f"[AIMY] {self.address_string()} {fmt % args}")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not _PROCESSING_OK:
        print(f"❌  processing.py のインポートに失敗: {_IMPORT_ERROR}")
        print("    pip install -r requirements.txt を実行してください")
        sys.exit(1)

    try:
        import mediapipe
        mp_status = "✅ MediaPipe あり（フル機能）"
    except ImportError:
        mp_status = "⚠️  MediaPipe なし（簡易版）"

    server = ThreadedHTTPServer(("", PORT), AimyHandler)
    print(f"✅  AIMY ローカルサーバー起動中 → http://localhost:{PORT}/aimy-app.html")
    print(f"    {mp_status}")
    print("    Ctrl+C で停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")
