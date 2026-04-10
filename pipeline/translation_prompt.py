"""
Prompt templates for the two-stage summarisation chain.

Stage 1 — Extraction prompt
    Instructs the LLM to return a strict JSON object containing
    decisions, action items (with owners), open questions, and
    key topics.  The output language matches the meeting language.

Stage 2 — Narrative summary prompt
    Takes the structured JSON from Stage 1 and writes a concise
    executive summary paragraph in the same language.
"""

# ── Stage 1: structured extraction ───────────────────────────────────────────

EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert meeting analyst. You will be given a transcript from a meeting.
Your task is to extract structured information from it.

The meeting was conducted in {language_name}. Your entire response MUST be written in {language_name}.

Return ONLY a valid JSON object — no markdown fences, no explanation, no preamble.
The JSON must follow this exact schema:

{{
  "key_topics": ["<topic>", ...],
  "decisions": ["<decision made during the meeting>", ...],
  "action_items": [
    {{
      "task": "<what needs to be done>",
      "owner": "<person responsible, or 'Unassigned' if not mentioned>",
      "due_date": "<due date if mentioned, otherwise null>"
    }},
    ...
  ],
  "open_questions": ["<unresolved question from the meeting>", ...]
}}

If a category has no entries, use an empty list [].

Meeting transcript:
\"\"\"
{transcript}
\"\"\"
"""

# ── Stage 2: narrative summary ────────────────────────────────────────────────

SUMMARY_PROMPT_TEMPLATE = """\
You are an expert meeting analyst. Using the structured meeting data below,
write a concise executive summary (3–5 sentences) of the meeting.

The summary MUST be written in {language_name}.

Focus on: what was discussed, what was decided, and what the next steps are.
Do not repeat raw lists — synthesise the information into clear, flowing prose.

Structured meeting data:
{structured_data}
"""

# ── Context-injected variant (used when extra context is provided by user) ───

EXTRACTION_PROMPT_WITH_CONTEXT_TEMPLATE = """\
You are an expert meeting analyst. You will be given a transcript from a meeting,
along with some context about the meeting provided by the user.

The meeting was conducted in {language_name}. Your entire response MUST be written in {language_name}.

User-provided context:
{context}

Return ONLY a valid JSON object — no markdown fences, no explanation, no preamble.
The JSON must follow this exact schema:

{{
  "key_topics": ["<topic>", ...],
  "decisions": ["<decision made during the meeting>", ...],
  "action_items": [
    {{
      "task": "<what needs to be done>",
      "owner": "<person responsible, or 'Unassigned' if not mentioned>",
      "due_date": "<due date if mentioned, otherwise null>"
    }},
    ...
  ],
  "open_questions": ["<unresolved question from the meeting>", ...]
}}

If a category has no entries, use an empty list [].

Meeting transcript:
\"\"\"
{transcript}
\"\"\"
"""


def build_extraction_prompt(
    transcript: str,
    language_name: str,
    context: str | None = None,
) -> str:
    """
    Build the Stage 1 extraction prompt.

    Args:
        transcript:     The diarized (or plain) transcript text.
        language_name:  Human-readable language name, e.g. "French".
        context:        Optional user-provided context string.

    Returns:
        Formatted prompt string ready to send to Ollama.
    """
    if context and context.strip():
        return EXTRACTION_PROMPT_WITH_CONTEXT_TEMPLATE.format(
            language_name=language_name,
            context=context.strip(),
            transcript=transcript,
        )
    return EXTRACTION_PROMPT_TEMPLATE.format(
        language_name=language_name,
        transcript=transcript,
    )


def build_summary_prompt(structured_data: str, language_name: str) -> str:
    """
    Build the Stage 2 narrative summary prompt.

    Args:
        structured_data: JSON string from Stage 1 (or a formatted string).
        language_name:   Human-readable language name, e.g. "French".

    Returns:
        Formatted prompt string ready to send to Ollama.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(
        language_name=language_name,
        structured_data=structured_data,
    )