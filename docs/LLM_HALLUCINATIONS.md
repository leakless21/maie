# LLM Hallucination Filtering

MAIE's LLM processor supports a lightweight, deterministic mechanism to remove known problematic LLM-generated lines (promo/boilerplate phrases ASR or LLMs often hallucinate).

Location

- The file lives at `src/config/llm_hallucinations.json` in the repository.

Format

```json
{
  "exact": ["Exact phrase to match and remove"]
}
```

Behavior

- "exact" entries are normalized (Unicode NFC, trimmed/collapsed whitespace, case-insensitive) and used for exact whole-text matches. If the entire LLM output equals a normalized "exact" phrase, the result will be removed.
- Parsed structured outputs (JSON summaries) are post-processed: string fields that exactly match an "exact" phrase are nulled; list entries equal to an exact phrase are removed.

Note: At present the LLM config supports only exact whole-text matches under the "exact" key. If you need regex or substring matching for the LLM processor, we can extend the configuration and processor logic to support those match modes in a follow-up change.

Usage and tuning

- Keep the file small and conservative to avoid accidental removal of valid text.
- For ASR hallucination filtering, use `data/asr_hallucinations.json` and the ASR pipeline (different file and tooling).

If you'd like the LLM behavior to be configurable (turn on/off, substring-mode matching), I can add application config values and tests for that.
