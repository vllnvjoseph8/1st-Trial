"""
InstantMech AI - Manual Retrieval Service

Takes the parsed_manual.json output from parser.py and answers the
question: "which chunks of this manual are actually relevant to what
the user just asked?"

This is the missing link between "we parsed the manual" and "the AI
assistant can answer with a real citation" -- without it, the only
options are (a) dump the entire manual into every prompt, which is
expensive and lets the model wander off-topic, or (b) guess, which is
exactly what we don't want in a repair-safety context.

Uses TF-IDF + cosine similarity: no external API calls, no embeddings
service dependency, works fully offline. It's not as semantically
sharp as an embeddings model, but it's a solid, honest baseline --
and it's a drop-in upgrade path later (swap _build_index/retrieve
for an embeddings-backed version without touching the callers).
"""

import json
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("ManualRetrieval")

# Below this similarity score, we don't trust the match enough to call
# it "relevant" -- better to tell the user honestly than to force a
# citation onto a chunk that doesn't really answer their question.
MIN_RELEVANCE_SCORE = 0.08


class ManualRetrievalService:
    def __init__(self, parsed_json_path):
        with open(parsed_json_path, "r") as f:
            self.manual_data = json.load(f)

        self.chunks = self.manual_data.get("citable_chunks", [])
        if not self.chunks:
            raise ValueError(
                f"'{parsed_json_path}' has no citable_chunks -- was it "
                f"produced by the current parser.py?"
            )

        self.source_file = self.manual_data.get("application_metadata", {}).get(
            "source_file", "manual"
        )

        self._build_index()
        logger.info(
            f"Loaded {len(self.chunks)} citable chunk(s) from {self.source_file}."
        )

    def _build_index(self):
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_matrix = self.vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------
    # Core retrieval: given a user's question, return the top_k most
    # relevant chunks, each still carrying its page number.
    # ------------------------------------------------------------------
    def retrieve(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_matrix)[0]

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )

        results = [
            {**chunk, "relevance_score": round(float(score), 4)}
            for chunk, score in ranked[:top_k]
            if score >= MIN_RELEVANCE_SCORE
        ]

        if not results:
            logger.info(f"No chunks scored above threshold for query: '{query}'")

        return results

    # ------------------------------------------------------------------
    # Builds the actual prompt context to send to the AI API: the
    # retrieved excerpts, each labeled with its page, plus an explicit
    # instruction to answer only from them and to cite the page used.
    # This is what should replace "dump the whole manual in" in your
    # sendAIMessage() flow.
    # ------------------------------------------------------------------
    def build_prompt_context(self, query, top_k=3):
        matches = self.retrieve(query, top_k=top_k)

        if not matches:
            return {
                "has_context": False,
                "system_prompt": (
                    "The user asked a question that isn't covered by the "
                    "loaded manual. Tell them clearly that this manual "
                    "doesn't address it, rather than guessing."
                ),
                "matches": [],
            }

        excerpt_block = "\n\n".join(
            f"[Page {m['page']}]\n{m['text']}" for m in matches
        )

        system_prompt = (
            "Answer the user's question using ONLY the manual excerpts below. "
            "Every claim you make must be traceable to one of these excerpts. "
            "When you answer, state which page number(s) support your answer. "
            "If the excerpts don't actually answer the question, say the "
            "manual doesn't cover it -- do not fill gaps from general "
            "knowledge.\n\n"
            f"--- MANUAL EXCERPTS ({self.source_file}) ---\n{excerpt_block}"
        )

        return {
            "has_context": True,
            "system_prompt": system_prompt,
            "matches": matches,
        }

    # ------------------------------------------------------------------
    # TF-IDF matches on shared vocabulary, not meaning -- e.g. a question
    # about "tire pressure" can match a step that says "release pressure"
    # on a belt bracket, purely because both contain the word "pressure".
    #
    # A tried-and-discarded idea: flagging matches with low raw token
    # overlap. That approach was tested here and rejected -- it flagged
    # a genuinely correct match ("Tools Needed..." for a "what tools do
    # I need" question) as a false positive, purely because "need" and
    # "needed" aren't the same token without stemming. A heuristic that
    # rejects correct answers is worse than no heuristic.
    #
    # The reliable version of this check requires actual language
    # understanding, i.e. it belongs on the AI model, not in this
    # module. The pattern: pass build_prompt_context()'s retrieved
    # excerpts to the model with an explicit instruction to confirm
    # relevance before citing (see the system_prompt already built
    # above -- it already tells the model to say "not covered" if the
    # excerpts don't answer the question). That live check is more
    # trustworthy than anything this module can verify offline.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    service = ManualRetrievalService("alternator_manual.parsed.json")

    test_questions = [
        "What tools do I need for this repair?",
        "Is it safe to work on this without disconnecting anything?",
        "What's the recommended tire pressure?",  # expected: not covered
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        context = service.build_prompt_context(question)
        if context["has_context"]:
            for m in context["matches"]:
                print(f"  -> page {m['page']} (score {m['relevance_score']}): "
                      f"{m['text'][:80]}...")
            print("  (Final relevance call belongs to the AI model -- see "
                  "system_prompt, which instructs it to say 'not covered' "
                  "if these excerpts don't actually answer the question.)")
        else:
            print("  -> No relevant chunks found (manual doesn't cover this).")
