# -*- coding: utf-8 -*-
"""
AIMY 本番用 Flask アプリ（API + 静的ファイル配信）
Render / Railway / Fly.io 等にデプロイして使う。

環境変数:
  GEMINI_API_KEY  : Google Gemini の API キー（必須）
  API_BASE_URL    : フロントエンドが呼ぶ API の URL（通常は空でよい）
  ALLOWED_ORIGIN  : CORS 許可オリジン（未設定時は * ）
"""

import base64
import os

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from processing import process_akami, process_face_mask, process_keana, process_shimi, process_shiwa, process_texture

app = Flask(__name__, static_folder="assets", static_url_path="/assets")

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

# ── 静的 HTML ──────────────────────────────────────────────────

_HTML_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return send_from_directory(_HTML_ROOT, "aimy-app.html")

@app.route("/report")
@app.route("/report-v3.html")
def report():
    return send_from_directory(_HTML_ROOT, "report-v3.html")

@app.route("/top-page-v5-app.html")
def top_page():
    return send_from_directory(_HTML_ROOT, "top-page-v5-app.html")

@app.route("/investor-pitch.html")
def investor_pitch():
    return send_from_directory(_HTML_ROOT, "investor-pitch.html")

# ── 動的 config.js（環境変数からキーを注入） ──────────────────

@app.route("/config.js")
def config_js():
    key      = os.environ.get("GEMINI_API_KEY", "")
    api_base = os.environ.get("API_BASE_URL", "")
    js = (
        "window.AIMY_CONFIG = {{\n"
        '  GEMINI_API_KEY: "{key}",\n'
        '  API_BASE_URL: "{api_base}"\n'
        "}};"
    ).format(key=key, api_base=api_base)
    return Response(js, mimetype="application/javascript")


# ── 共通: リクエストから画像バイト列と severity を取り出す ──
def _parse_request():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return None, None, (jsonify({"error": "image フィールドが必要です"}), 400)
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        return None, None, (jsonify({"error": "base64 デコードエラー"}), 400)
    severity = float(data.get("severity", 1.0))
    return img_bytes, severity, None


def _run(processor, img_bytes, severity):
    try:
        result_bytes = processor(img_bytes, severity=severity)
    except Exception as e:
        return jsonify({"error": f"画像処理エラー: {e}"}), 500
    b64 = "data:image/png;base64," + base64.b64encode(result_bytes).decode()
    return jsonify({"result": b64})


# ── エンドポイント ──

@app.route("/api/face-mask", methods=["POST"])
def api_face_mask():
    img_bytes, severity, err = _parse_request()
    if err:
        return err
    return _run(process_face_mask, img_bytes, severity)


@app.route("/api/shimi", methods=["POST"])
def api_shimi():
    img_bytes, severity, err = _parse_request()
    if err:
        return err
    return _run(process_shimi, img_bytes, severity)


@app.route("/api/shiwa", methods=["POST"])
def api_shiwa():
    img_bytes, severity, err = _parse_request()
    if err:
        return err
    return _run(process_shiwa, img_bytes, severity)


@app.route("/api/texture", methods=["POST"])
def api_texture():
    img_bytes, severity, err = _parse_request()
    if err:
        return err
    return _run(process_texture, img_bytes, severity)


@app.route("/api/keana", methods=["POST"])
def api_keana():
    img_bytes, severity, err = _parse_request()
    if err:
        return err
    return _run(process_keana, img_bytes, severity)


@app.route("/api/akami", methods=["POST"])
def api_akami():
    img_bytes, severity, err = _parse_request()
    if err:
        return err
    return _run(process_akami, img_bytes, severity)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
