import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.exceptions import OutputParsingError
from core.logger import get_logger

logger = get_logger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    """A single action item extracted from the meeting."""
    task: str
    owner: str = "Unassigned"
    due_date: str | None = None


class MeetingExtraction(BaseModel):
    """
    Structured output from Stage 1 of the summarisation chain.

    All fields default to empty lists so partial LLM responses
    don't cause validation errors.
    """
    key_topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("key_topics", "decisions", "open_questions", mode="before")
    @classmethod
    def ensure_string_list(cls, v: Any) -> list[str]:
        """Coerce items to strings and drop empty entries."""
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if item]

    @field_validator("action_items", mode="before")
    @classmethod
    def ensure_action_items(cls, v: Any) -> list[dict]:
        """Accept dicts or ActionItem-like objects; drop malformed entries."""
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, dict) and "task" in item:
                result.append(item)
            elif isinstance(item, str):
                # LLM sometimes returns plain strings instead of objects
                result.append({"task": item, "owner": "Unassigned", "due_date": None})
        return result


class MeetingOutput(BaseModel):
    """
    Complete output produced by the two-stage summarisation pipeline.
    This is what gets stored in SQLite and displayed in the UI.
    """
    summary: str
    extraction: MeetingExtraction
    language_code: str
    language_name: str


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences that LLMs often wrap JSON in.
    Handles ```json ... ```, ``` ... ```, and bare JSON.
    """
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def parse_extraction(raw_response: str) -> MeetingExtraction:
    """
    Parse the Stage 1 LLM response into a MeetingExtraction model.

    The LLM is instructed to return pure JSON but may wrap it in
    markdown fences or include minor formatting noise.  We strip
    those before parsing.

    Args:
        raw_response: Raw string response from Ollama.

    Returns:
        Validated MeetingExtraction instance.

    Raises:
        OutputParsingError: If the response cannot be parsed as JSON
                            or fails Pydantic validation.
    """
    cleaned = _strip_json_fences(raw_response)
    logger.debug("Parsing extraction response (%d chars)", len(cleaned))

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Attempt to extract the first JSON object from a noisy response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                raise OutputParsingError(
                    f"Could not parse JSON from LLM response.\n"
                    f"Raw (first 300 chars): {raw_response[:300]}"
                ) from exc
        else:
            raise OutputParsingError(
                f"LLM response contained no JSON object.\n"
                f"Raw (first 300 chars): {raw_response[:300]}"
            ) from exc

    try:
        result = MeetingExtraction.model_validate(data)
    except Exception as exc:
        raise OutputParsingError(
            f"JSON parsed but failed schema validation: {exc}\n"
            f"Data: {data}"
        ) from exc

    logger.info(
        "Extraction parsed: %d topics, %d decisions, %d action items, %d questions",
        len(result.key_topics),
        len(result.decisions),
        len(result.action_items),
        len(result.open_questions),
    )
    return result