"""
InstantMech AI - Backend Server

Sits between the browser (index.html) and:
  - parser.py            (turns an uploaded PDF into parsed manual JSON)
  - retrieval_service.py (finds relevant chunks for a question)
  - the Anthropic API     (generates the actual answer)

Why this needs to exist: index.html is currently a static GitHub Pages
site. Client-side JS can't read/write files on a server, can't run
Python, and can't hold an API key without exposing it to anyone who
views page source (Ctrl+U on any live site reveals all of it). This
server is what makes all three possible without leaking your key.
"""

import os
import uuid

from flask import Flask, request, jsonify
import requests

from parser import DocumentProcessingService
from retrieval_service import ManualRetrievalService

MANUALS_DIR = os.path.join(os.path.dirname(__file__), "manuals")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # "*" is fine for local dev (Live Server on one port, Flask on
    # another). Before deploying anywhere public, replace it with your
    # actual frontend origin, e.g. "https://vllnvjoseph8.github.io".
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/manuals", methods=["POST"])
def upload_manual():
    """
    Receives a PDF upload from the "Choose Manual PDF" button, parses
    it, and returns a manual_id the frontend uses for follow-up
    questions. The id (not the original filename) is what keeps two
    users' uploads from colliding.
    """
    if "manual" not in request.files:
        return jsonify({"error": "No file uploaded under field 'manual'."}), 400

    uploaded = request.files["manual"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    os.makedirs(MANUALS_DIR, exist_ok=True)
    manual_id = str(uuid.uuid4())
    saved_path = os.path.join(MANUALS_DIR, f"{manual_id}.pdf")
    uploaded.save(saved_path)

    service = DocumentProcessingService(target_directory=MANUALS_DIR)
    try:
        service.run_pipeline(f"{manual_id}.pdf")
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 422

    return jsonify({
        "manual_id": manual_id,
        "filename": uploaded.filename,
    })


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Receives {manual_id, question} from the AI Mechanic Assistant chat
    box, retrieves relevant excerpts, and calls the Anthropic API with
    a citation-forcing system prompt built by retrieval_service.py.
    """
    body = request.get_json(silent=True) or {}
    manual_id = body.get("manual_id")
    question = body.get("question", "").strip()

    if not manual_id or not question:
        return jsonify({"error": "Both 'manual_id' and 'question' are required."}), 400

    parsed_path = os.path.join(MANUALS_DIR, f"{manual_id}.parsed.json")
    if not os.path.exists(parsed_path):
        return jsonify({"error": f"No parsed manual found for id '{manual_id}'."}), 404

    retrieval = ManualRetrievalService(parsed_path)
    context = retrieval.build_prompt_context(question)

    if not context["has_context"]:
        return jsonify({
            "answer": "This manual doesn't cover that — try rephrasing, "
                      "or check the manual directly.",
            "citations": [],
        })

    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 500

    api_response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1000,
            "system": context["system_prompt"],
            "messages": [{"role": "user", "content": question}],
        },
        timeout=30,
    )

    if api_response.status_code != 200:
        return jsonify({
            "error": f"Anthropic API error ({api_response.status_code}): "
                     f"{api_response.text}"
        }), 502

    data = api_response.json()
    answer_text = "".join(
        block.get("text", "") for block in data.get("content", [])
        if block.get("type") == "text"
    )

    return jsonify({
        "answer": answer_text,
        "citations": [
            {"page": m["page"], "excerpt": m["text"][:200]}
            for m in context["matches"]
        ],
    })


if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("[WARNING] ANTHROPIC_API_KEY is not set. /api/ask will fail "
              "until you set it, e.g.:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...")
    app.run(host="0.0.0.0", port=5000, debug=True)
