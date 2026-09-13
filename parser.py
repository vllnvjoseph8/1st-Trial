"""
InstantMech AI - Manual Parsing Service

Reads an uploaded PDF service manual, extracts its text page-by-page,
and produces a structured JSON payload where every extracted fact
(safety warning, tool, step) is tagged with the page it came from.

That page tag is what lets the AI Mechanic Assistant cite its source
("Source: Manual, p.12") instead of just asserting an answer.
"""

import os
import re
import json
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("ManualParser")

# Section headers we look for. Add more variants here as you encounter
# manuals that phrase these differently (e.g. "SAFETY NOTICE" vs
# "SAFETY WARNINGS") -- this list is the main thing you'll extend over
# time as real-world manuals don't all follow one template.
SECTION_HEADERS = {
    "safety": [r"SAFETY WARNINGS?", r"SAFETY NOTICE"],
    "equipment": [r"REQUIRED EQUIPMENT", r"TOOLS(?: AND PARTS)?"],
    "procedure": [r"OPERATIONAL PROCEDURE", r"PROCEDURE", r"STEPS?"],
}


class DocumentProcessingService:
    def __init__(self, target_directory):
        self.target_dir = target_directory
        os.makedirs(self.target_dir, exist_ok=True)
        logger.info("Document processing pipeline initialized.")
        logger.info(f"Monitoring workspace area: {os.path.abspath(target_directory)}")

    # ------------------------------------------------------------------
    # Stage 1: Read the PDF, page by page. This is the piece the old
    # version never actually did -- everything downstream depends on
    # this producing real per-page text.
    # ------------------------------------------------------------------
    def extract_pages(self, filename):
        file_path = os.path.join(self.target_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Manual not found: {file_path}")

        logger.info(f"Fetching payload asset: {filename}")

        try:
            reader = PdfReader(file_path)
        except PdfReadError as e:
            raise ValueError(f"Could not read '{filename}' as a PDF: {e}")

        if reader.is_encrypted:
            # Try an empty password first (common for "restricted but not
            # really locked" PDFs); otherwise this manual can't be parsed.
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError(
                    f"'{filename}' is password-protected and cannot be parsed."
                )

        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                logger.info(f"Page {i} had no extractable text (likely a scanned "
                            f"image) -- skipping. Consider OCR for scanned manuals.")
                continue
            pages.append({"page": i, "text": text})

        if not pages:
            raise ValueError(
                f"No extractable text found in '{filename}'. It may be a "
                f"scanned/image-only PDF that needs OCR."
            )

        logger.info(f"Extracted text from {len(pages)} page(s).")
        return pages

    # ------------------------------------------------------------------
    # Stage 2: Turn each page's raw text into citable chunks. This is
    # what the AI assistant should actually search/retrieve over --
    # small pieces, each stamped with its page number.
    # ------------------------------------------------------------------
    def build_citable_chunks(self, pages):
        chunks = []
        for page in pages:
            # Split on blank lines / paragraph breaks so each chunk stays
            # small enough to be a useful, specific citation.
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page["text"]) if p.strip()]
            for idx, para in enumerate(paragraphs):
                chunks.append({
                    "chunk_id": f"page_{page['page']}_chunk_{idx}",
                    "page": page["page"],
                    "text": para,
                })
        return chunks

    # ------------------------------------------------------------------
    # Stage 3: Best-effort structured extraction (safety/tools/steps),
    # now tracking which page each match came from instead of losing
    # that information the moment everything gets concatenated.
    # ------------------------------------------------------------------
    def extract_structured_sections(self, pages):
        full_text = "\n".join(p["text"] for p in pages)

        # Map every character offset in full_text back to a page number,
        # so any regex match we find can be attributed to a real page.
        offset_to_page = []
        for page in pages:
            offset_to_page.extend([page["page"]] * (len(page["text"]) + 1))  # +1 for the joining \n

        def page_for_offset(offset):
            if 0 <= offset < len(offset_to_page):
                return offset_to_page[offset]
            return None

        def find_section(section_key):
            for pattern in SECTION_HEADERS[section_key]:
                match = re.search(
                    rf"{pattern}\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z \-]{{3,}}\n|\Z)",
                    full_text, re.DOTALL,
                )
                if match:
                    return match, page_for_offset(match.start(1))
            return None, None

        target_match = re.search(r"SYSTEM TARGET:\s*(.*)", full_text)
        system_target = target_match.group(1).strip() if target_match else None
        if not system_target:
            logger.info("Could not identify a 'SYSTEM TARGET' header -- "
                        "field left as None rather than a fake default.")

        safety_match, safety_page = find_section("safety")
        safety_lines = []
        if safety_match:
            safety_lines = [
                line.strip("-• ").strip()
                for line in safety_match.group(1).strip().split("\n") if line.strip()
            ]
        else:
            logger.info("No safety section found in this manual.")

        equipment_match, equipment_page = find_section("equipment")
        tools_array = []
        if equipment_match:
            tools_line = re.search(r"Tools Needed:\s*(.*)", equipment_match.group(1))
            if tools_line:
                tools_array = [t.strip() for t in tools_line.group(1).split(",") if t.strip()]
        else:
            logger.info("No equipment/tools section found in this manual.")

        procedure_match, procedure_page = find_section("procedure")
        steps = []
        if procedure_match:
            steps = re.findall(r"(Step \d+:\s*.*)", procedure_match.group(1))
        else:
            logger.info("No procedure/steps section found in this manual.")

        return {
            "parsed_system": system_target,
            "safety_protocols": {
                "items": safety_lines,
                "source_page": safety_page,
            },
            "extracted_tools": {
                "items": tools_array,
                "source_page": equipment_page,
            },
            "procedural_steps": {
                "items": [s.strip() for s in steps],
                "source_page": procedure_page,
            },
        }

    # ------------------------------------------------------------------
    # Orchestrates the full pipeline for one manual.
    # ------------------------------------------------------------------
    def run_pipeline(self, filename):
        pages = self.extract_pages(filename)
        chunks = self.build_citable_chunks(pages)
        structured = self.extract_structured_sections(pages)

        payload = {
            "application_metadata": {
                "source_file": filename,
                "page_count": len(pages),
            },
            **structured,
            "citable_chunks": chunks,
        }

        return self.save_processed_output(payload, filename)

    def save_processed_output(self, data, source_filename):
        # Key the output filename off the source manual, so parsing a
        # second manual doesn't silently overwrite the first one's data.
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", base_name)
        output_filename = f"{safe_name}.parsed.json"
        output_path = os.path.join(self.target_dir, output_filename)

        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        logger.info("Extraction sequence complete.")
        logger.info(f"Structured output written to disk: {output_path}")
        return output_path


if __name__ == "__main__":
    service = DocumentProcessingService(target_directory=".")

    try:
        output_log = service.run_pipeline("alternator_manual.pdf")
        with open(output_log, "r") as f:
            print("\n--- Verified Backend Service JSON Yield ---")
            print(f.read())
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Parsing failed: {e}")