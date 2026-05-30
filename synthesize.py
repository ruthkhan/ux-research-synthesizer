import sys
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from docx import Document
import os
import pathlib

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TRANSCRIPTS_DIR   = pathlib.Path("transcripts")
CODED_DATA_DIR    = pathlib.Path("coded_data")
CODEBOOK_PATH     = pathlib.Path("codebook.json")
OPEN_CODING_BATCH = 5

CODED_DATA_DIR.mkdir(exist_ok=True)

MODEL = "gemini-3.5-flash"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_docx(filepath):
    doc = Document(filepath)
    return [para.text for para in doc.paragraphs if para.text.strip()]


def extract_timestamp(filepath, paragraphs):
    """Extract interview date from first line of transcript; fall back to file mtime."""
    if paragraphs:
        first_line = paragraphs[0]
        for pattern, fmts in [
            (r'\d{4}-\d{2}-\d{2}',  ['%Y-%m-%d']),
            (r'\d{1,2}/\d{1,2}/\d{4}', ['%d/%m/%Y', '%m/%d/%Y']),
            (r'\d{1,2} \w+ \d{4}',  ['%d %B %Y', '%d %b %Y']),
        ]:
            m = re.search(pattern, first_line)
            if m:
                for fmt in fmts:
                    try:
                        return datetime.strptime(m.group(), fmt).isoformat()
                    except ValueError:
                        continue
    return datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()


def load_codebook():
    if CODEBOOK_PATH.exists() and CODEBOOK_PATH.stat().st_size > 0:
        return json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
    return None


def write_codebook(codebook):
    CODEBOOK_PATH.write_text(
        json.dumps(codebook, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_codebook(codebook):
    barriers  = {k: v for k, v in codebook.items() if v["type"] == "barrier"}
    practices = {k: v for k, v in codebook.items() if v["type"] == "practice"}
    contexts  = {k: v for k, v in codebook.items() if v["type"] == "context"}
    for section, items in [("BARRIERS", barriers), ("PRACTICES", practices), ("CONTEXTS", contexts)]:
        if items:
            print(f"\n  [{section}]")
            for name, entry in items.items():
                links_str = f"  → {', '.join(entry['links'])}" if entry.get('links') else ""
                print(f"  {name}: {entry['definition']}{links_str}")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

_CODE_ITEM_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "name":       types.Schema(type=types.Type.STRING),
        "type":       types.Schema(type=types.Type.STRING),
        "definition": types.Schema(type=types.Type.STRING),
        "links":      types.Schema(
                          type=types.Type.ARRAY,
                          items=types.Schema(type=types.Type.STRING)
                      ),
    },
    required=["name", "type", "definition", "links"]
)

CODEBOOK_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "codes": types.Schema(type=types.Type.ARRAY, items=_CODE_ITEM_SCHEMA)
    },
    required=["codes"]
)

_QUOTE_ITEM_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "quote":     types.Schema(type=types.Type.STRING),
        "codes":     types.Schema(
                         type=types.Type.ARRAY,
                         items=types.Schema(type=types.Type.STRING)
                     ),
        "inference": types.Schema(type=types.Type.STRING),
    },
    required=["quote", "codes"]
)

_PROPOSED_CODE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "name":                  types.Schema(type=types.Type.STRING),
        "type":                  types.Schema(type=types.Type.STRING),
        "definition":            types.Schema(type=types.Type.STRING),
        "links":                 types.Schema(
                                     type=types.Type.ARRAY,
                                     items=types.Schema(type=types.Type.STRING)
                                 ),
        "closest_existing_code": types.Schema(type=types.Type.STRING),
        "why_existing_code_fails": types.Schema(type=types.Type.STRING),
    },
    required=["name", "type", "definition", "links",
              "closest_existing_code", "why_existing_code_fails"]
)

CODING_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "quotes": types.Schema(type=types.Type.ARRAY, items=_QUOTE_ITEM_SCHEMA),
        "new_codes_proposed": types.Schema(
            type=types.Type.ARRAY, items=_PROPOSED_CODE_SCHEMA
        ),
    },
    required=["quotes"]
)

RECODE_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "quotes": types.Schema(type=types.Type.ARRAY, items=_QUOTE_ITEM_SCHEMA)
    },
    required=["quotes"]
)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def generate_initial_codebook(batch):
    """Single LLM call across the first OPEN_CODING_BATCH transcripts → codebook dict."""
    combined = ""
    for name, paragraphs in batch:
        combined += f"\n\n=== TRANSCRIPT: {name} ===\n" + "\n".join(paragraphs)

    prompt = f"""The following are {len(batch)} semi-structured interview transcripts from a study on researchers' perceptions and practices around open qualitative research.

Generate a master codebook by analysing all transcripts together.

Rules:
1. Identify recurring themes across ALL transcripts — not one participant's idiosyncratic concerns.
2. Classify each code as exactly one of:
   - barrier  — a problem or obstacle working against open qualitative research
   - practice — something researchers currently do or propose to enable it
   - context  — a background condition that shapes the issue without being a problem or solution
3. Produce NO MORE THAN 10 barrier codes. Merge related barriers aggressively into umbrella codes.
4. practice and context codes are unconstrained in number, but each MUST link to at least one barrier via the links field.
5. barrier codes must have an empty links array.
6. Use SNAKE_CASE for code names. Write a precise one-line definition per code.

TRANSCRIPTS:
{combined}
"""

    response = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CODEBOOK_RESPONSE_SCHEMA,
            temperature=0.0,
            system_instruction=(
                "You are a qualitative research methodologist. Produce a lean, "
                "non-redundant codebook. Merge barriers aggressively — max 10 total. "
                "Every practice and context code must link to at least one barrier."
            )
        ),
        contents=prompt
    )

    codes = json.loads(response.text)["codes"]
    codebook = {}
    barrier_count = 0
    for code in codes:
        if code["type"] == "barrier":
            if barrier_count >= 10:
                continue
            barrier_count += 1
            codebook[code["name"]] = {
                "type": "barrier",
                "definition": code["definition"]
            }
        else:
            entry = {"type": code["type"], "definition": code["definition"]}
            if code.get("links"):
                entry["links"] = code["links"]
            codebook[code["name"]] = entry
    return codebook


def code_transcript(name, paragraphs, codebook, propose_new_codes=True):
    """
    Extract and code relevant quotes from a transcript.
    propose_new_codes=False for batch transcripts (codebook was built from them)
    and for the second call after any proposals were rejected.
    """
    codebook_text   = json.dumps(codebook, indent=2, ensure_ascii=False)
    transcript_text = "\n".join(paragraphs)

    if not propose_new_codes:
        new_codes_instruction = (
            "Do NOT propose any new codes. Assign only codes from the master codebook."
        )
        schema = RECODE_RESPONSE_SCHEMA
    else:
        new_codes_instruction = (
            "Default assumption: every quote can be coded with the existing codebook. "
            "Only propose a new code as a last resort, when you have first identified the "
            "closest existing code and can explain precisely why it fails to capture this "
            "concept. Partial overlap is not sufficient — the concept must be genuinely "
            "absent from the codebook. New practice and context codes must link to at "
            "least one existing barrier. For each proposal, populate closest_existing_code "
            "and why_existing_code_fails."
        )
        schema = CODING_RESPONSE_SCHEMA

    prompt = f"""Semi-structured interview transcript — participant: {name}.
Study topic: researchers' perceptions and practices around open qualitative research.

Task: extract all quotes relevant to barriers, practices, or contextual factors related to open qualitative research. Ignore off-topic content.

For each relevant quote:
- Copy verbatim — do not paraphrase
- Assign one or more codes from the master codebook
- If interpretation beyond the literal text is needed, add a brief inference note in the inference field; otherwise omit it

{new_codes_instruction}

MASTER CODEBOOK:
{codebook_text}

TRANSCRIPT:
{transcript_text}
"""

    response = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            system_instruction=(
                "You are an experienced qualitative research analyst. "
                "Extract only relevant verbatim quotes — do not paraphrase. "
                "Assign a code only when it genuinely fits; do not over-code."
            )
        ),
        contents=prompt
    )
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def review_proposed_codes(proposed):
    if not proposed:
        return [], False

    print(f"\n{'='*60}")
    print(f"  {len(proposed)} new code(s) proposed:")
    print(f"{'='*60}\n")

    accepted    = []
    rejected_any = False
    for code in proposed:
        links_str = f"  → {', '.join(code['links'])}" if code.get('links') else ""
        print(f"  {code['name']}: [{code['type']}] {code['definition']}{links_str}")
        print(f"  Closest existing: {code.get('closest_existing_code', '—')}")
        print(f"  Why it fails:     {code.get('why_existing_code_fails', '—')}")
        if input("  Add to codebook? (y/n): ").strip().lower() == "y":
            accepted.append(code)
            print("  ✓ Added.\n")
        else:
            rejected_any = True
            print("  Skipped.\n")

    return accepted, rejected_any


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

codebook    = load_codebook()
transcripts = sorted(TRANSCRIPTS_DIR.glob("*.docx"))

if not transcripts:
    print("No .docx files found in transcripts/")
    sys.exit()

batch_stems = {t.stem for t in transcripts[:OPEN_CODING_BATCH]}

if codebook is None:
    batch = transcripts[:OPEN_CODING_BATCH]
    print(f"\nNo codebook found — generating from first {len(batch)} transcript(s)...")

    batch_data = [(p.stem, read_docx(p)) for p in batch]
    codebook   = generate_initial_codebook(batch_data)

    print(f"\n  Codebook generated ({len(codebook)} codes):")
    print_codebook(codebook)
    write_codebook(codebook)
    print(f"\n  Saved to {CODEBOOK_PATH}")

    print(f"\n{'='*60}")
    print(f"  Review codebook.json now. Edit the file directly, then press Enter.")
    print(f"  (Ctrl+C to stop here and re-run later.)")
    print(f"{'='*60}")
    input("\n  Press Enter when you're happy with the codebook: ")

    codebook = load_codebook()

# Closed coding
unprocessed = [t for t in transcripts
               if not (CODED_DATA_DIR / f"{t.stem}.json").exists()]

if not unprocessed:
    print("\nAll transcripts already coded.")
    sys.exit()

print(f"\nCodebook loaded ({len(codebook)} codes) — coding {len(unprocessed)} transcript(s).")

for docx_path in unprocessed:
    name = docx_path.stem
    print(f"\n  Processing: {docx_path.name}")

    paragraphs        = read_docx(docx_path)
    timestamp         = extract_timestamp(docx_path, paragraphs)
    is_batch          = name in batch_stems

    result   = code_transcript(name, paragraphs, codebook, propose_new_codes=not is_batch)
    proposed = result.get("new_codes_proposed", [])

    accepted, rejected_any = review_proposed_codes(proposed)

    if accepted:
        for code in accepted:
            entry = {"type": code["type"], "definition": code["definition"]}
            if code.get("links"):
                entry["links"] = code["links"]
            codebook[code["name"]] = entry
        write_codebook(codebook)
        codebook = load_codebook()
        print(f"  Codebook updated ({len(accepted)} code(s) added).")

    if rejected_any:
        print(f"  Re-coding {docx_path.name} without rejected codes...")
        result = code_transcript(name, paragraphs, codebook, propose_new_codes=False)

    output = {
        "participant":         name,
        "interview_timestamp": timestamp,
        "coding_mode":         "closed",
        "quotes":              result["quotes"]
    }

    out_path = CODED_DATA_DIR / f"{name}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved to {out_path}")

print("\nAll transcripts coded.")
