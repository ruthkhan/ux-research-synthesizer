# UX Research Synthesizer

A Python pipeline that turns interview transcripts into a structured, queryable JSON dataset using schema-enforced LLM outputs.

Processes real academic interview data (15 participants, University of Sheffield / ORDA).

---

## What it does

Given a folder of `.docx` interview transcripts, the script:

1. **Generates a codebook** from the first 5 transcripts in a single LLM call — producing barrier, practice, and context codes with explicit links between them
2. **Codes each transcript** by extracting verbatim quotes and assigning codes from the codebook — no paraphrasing, no intermediate summaries
3. **Handles new codes incrementally** — subsequent transcripts can propose new codes; the researcher reviews each proposal with a justification before it is accepted
4. **Outputs structured JSON** per participant, queryable by code, type, or participant

---

## Architecture decisions

**Coded excerpts, not briefs.** Marrative summaries ("briefs") per participant compounds LLM translation errors: the model paraphrases the transcript, then the synthesis re-interprets the paraphrase. The current design extracts verbatim quotes only — the LLM selects which quotes are relevant and assigns codes, but does not reword them.

**Batch codebook generation.** The codebook is generated from all 5 seed transcripts together in one call, not per-transcript then consolidated. This produces a leaner codebook because the model can see cross-participant patterns rather than coding each person's idiosyncratic phrasing independently.

**Incremental production design.** The codebook is generated once and reused. New transcripts added later extend the codebook only if they introduce genuinely new concepts — the researcher reviews each proposal with a required justification field (`closest_existing_code` + `why_existing_code_fails`) before any new code is accepted.

**Schema-enforced JSON.** All LLM outputs use Gemini's `response_schema` parameter for structural enforcement — no regex parsing, no brittle text extraction.

**Codebook links.** Every practice and context code carries a `links` array pointing to the barriers it addresses, making the barrier→solution relationships queryable without re-running any LLM call.

---

## Output format

**`codebook.json`**
```json
{
  "CONSENT_LIMITATIONS": {
    "type": "barrier",
    "definition": "Restrictions preventing data sharing due to consent agreements."
  },
  "PROACTIVE_DATA_MANAGEMENT": {
    "type": "practice",
    "definition": "Designing workflows early to enable open sharing.",
    "links": ["CONSENT_LIMITATIONS", "TIME_AND_RESOURCE_CONSTRAINTS"]
  }
}
```

**`coded_data/<participant>.json`**
```json
{
  "participant": "Ana",
  "interview_timestamp": "2019-06-12T00:00:00",
  "coding_mode": "closed",
  "quotes": [
    {
      "quote": "verbatim text from transcript",
      "codes": ["CONSENT_LIMITATIONS"],
      "inference": "optional: reasoning if the quote requires interpretation"
    }
  ]
}
```

---

## Sample outputs

- [`codebook.json`](codebook.json) — master codebook with barrier/practice/context classification and links
- [`coded_data/`](coded_data/) — participant JSON files

---

## Running it yourself

**Requirements**
```
pip install google-genai python-dotenv python-docx
```

**Setup**
```
GEMINI_API_KEY=your_key_here   # in .env
```

**First run** (no codebook yet — generates codebook from first 5 transcripts):
```
python synthesize.py
```
The script pauses after generating the codebook so you can review and edit `codebook.json` before coding begins.

**Subsequent runs** (codebook exists — codes remaining transcripts):
```
python synthesize.py
```
Proposes new codes for review after each new transcript. Accepted codes are added to `codebook.json` immediately and used for all subsequent transcripts in the same run.

---

## Data source

Interview transcripts from: *Fostering cultures of open qualitative research: Dataset 2*, University of Sheffield, deposited at ORDA. Licensed CC-BY-NC. Transcripts not included in this repository — outputs use pseudonyms only.

---

## Stack

- Python 3.11
- [google-genai](https://pypi.org/project/google-genai/) (Gemini API)
- `response_schema` for structured JSON enforcement
- python-docx for transcript parsing
