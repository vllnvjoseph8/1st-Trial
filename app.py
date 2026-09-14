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
from dotenv import load_dotenv
import requests

from parser import DocumentProcessingService
from retrieval_service import ManualRetrievalService

# Reads a local .env file (GEMINI_API_KEY=AIza...) into the environment
# automatically, so you don't have to re-export the key in every new
# terminal session. .env must be in .gitignore -- it should never be
# committed.
load_dotenv()

MANUALS_DIR = os.path.join(os.path.dirname(__file__), "manuals")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# "gemini-flash-latest" is Google's alias that auto-points to the
# current stable Flash release, rather than a specific version string.
# Google has been retiring specific Flash model names every few months
# through 2026 (2.5-flash is already gone) -- pinning to a literal
# version here would just mean hitting this same 404 again later.
# Tradeoff: behavior can shift slightly whenever Google hot-swaps what
# "latest" points to. Fine for a demo; pin to a specific dated version
# instead if you need output to stay perfectly consistent over time.
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

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

    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is not set on the server."}), 500

    api_response = requests.post(
        GEMINI_URL,
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "system_instruction": {
                "parts": [{"text": context["system_prompt"]}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": question}]}
            ],
        },
        timeout=30,
    )

    if api_response.status_code != 200:
        return jsonify({
            "error": f"Gemini API error ({api_response.status_code}): "
                     f"{api_response.text}"
        }), 502

    data = api_response.json()
    try:
        answer_text = "".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"]["parts"]
        )
    except (KeyError, IndexError):
        # Can happen if the response was blocked by a safety filter
        # instead of returning normal candidates -- surface the raw
        # response rather than crashing, so it's debuggable.
        return jsonify({
            "error": "Gemini returned an unexpected response shape.",
            "raw_response": data,
        }), 502

    return jsonify({
        "answer": answer_text,
        "citations": [
            {"page": m["page"], "excerpt": m["text"][:200]}
            for m in context["matches"]
        ],
    })


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY is not set. /api/ask will fail "
              "until you set it, e.g. in a .env file:\n"
              "  GEMINI_API_KEY=AIza...")
    # host="127.0.0.1" (not "0.0.0.0") means only THIS machine can reach
    # the server -- nobody else on your WiFi network can. debug=False
    # avoids exposing Werkzeug's interactive debugger. For a local demo
    # you don't need either of the more permissive settings.
    app.run(host="127.0.0.1", port=5000, debug=False)
