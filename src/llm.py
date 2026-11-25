"""Gemini models for MedLabel.

Two ways to get a yes / no / maybe answer from Gemini:
- Gemini Decision 1 (Decisions API): scores the three options and returns calibrated
  probabilities. It generates no text, so it has no temperature.
- Generative Gemini models (generate_content): return JSON with a label, a self-reported
  confidence and a short rationale. These accept temperature and top_p.
"""
import json
import os
import time

LABELS = ("yes", "no", "maybe")
DECISION_MODEL = "gemini-decision-1"
GENERATIVE_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash")
# gemini-3.8-flash allows only 20 requests a day on the free tier, too few for a 50-question sweep
EVAL_GENERATIVE_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
DEFAULT_GENERATIVE_MODEL = "gemini-3.5-flash-lite"

LABEL_CRITERIA = {
    "yes": "The study's findings support the statement in the question.",
    "no": "The study's findings contradict the statement or find no effect.",
    "maybe": "The findings are mixed, inconclusive, or only partly support the statement.",
}

# Each prompt variant is used by both model types so they can be compared on equal terms.
PROMPTS = {
    "plain": {
        "system": None,
        "instructions": "Answer this research question with yes, no or maybe: {question}",
        "criteria": False,
    },
    "criteria": {
        "system": None,
        "instructions": (
            "Based on the study abstract, what is the answer to this research question: "
            "{question}"
        ),
        "criteria": True,
    },
    "expert": {
        "system": (
            "You are an expert biomedical reviewer. Judge only what the study's own results "
            "and conclusion show. Do not use outside knowledge. Use 'maybe' only when the "
            "evidence is genuinely mixed or inconclusive."
        ),
        "instructions": (
            "As an expert reviewer reading only this abstract, what does the evidence say about "
            "this research question: {question}"
        ),
        "criteria": True,
    },
}

JUDGE_LEVELS = [
    "Unfaithful: the rationale contradicts the abstract or invents findings.",
    "Weak: the rationale is vague or only loosely tied to the abstract.",
    "Mostly faithful: the rationale is supported by the abstract with minor gaps.",
    "Faithful: every claim in the rationale is directly supported by the abstract.",
]

_RETRYABLE = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")


class GeminiNotConfigured(RuntimeError):
    pass


def get_client():
    """Create a Gemini client from GEMINI_API_KEY. The Decisions API is only on v1beta."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise GeminiNotConfigured("Set GEMINI_API_KEY to use Gemini models.")
    from google import genai

    return genai.Client(api_key=key, http_options={"api_version": "v1beta"})


def _with_retry(fn, attempts: int = 6):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # the SDK raises different classes per API surface
            if i == attempts - 1 or not any(code in str(exc) for code in _RETRYABLE):
                raise
            time.sleep(min(2**i * 5, 65))


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def build_prompt(prompt_id: str, question: str) -> tuple[str | None, str]:
    p = PROMPTS[prompt_id]
    return p["system"], p["instructions"].format(question=question)


def decide(client, question: str, context: str, prompt_id: str = "criteria") -> dict:
    """Ask Gemini Decision 1 to choose yes, no or maybe for one question."""
    system, instructions = build_prompt(prompt_id, question)
    if system:
        instructions = f"{system}\n\n{instructions}"
    use_criteria = PROMPTS[prompt_id]["criteria"]
    criteria = {lbl: (LABEL_CRITERIA[lbl] if use_criteria else "") for lbl in LABELS}
    start = time.perf_counter()
    result = _with_retry(
        lambda: client.decisions.create(
            model=DECISION_MODEL,
            input=f"Study abstract:\n{context}" if context else "No abstract was provided.",
            questions={
                "answer": {"type": "choice", "instructions": instructions, "criteria": criteria}
            },
        )
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = _get(result.answers, "answer")
    probs = {str(k): float(v) for k, v in dict(_get(answer, "probabilities")).items()}
    usage = _get(result, "usage")
    return {
        "label": str(_get(answer, "choice")),
        "confidence": float(_get(answer, "confidence")),
        "probabilities": probs,
        "rationale": None,
        "latency_ms": latency_ms,
        "input_tokens": int(_get(usage, "total_input_tokens") or 0),
        "output_tokens": 0,
    }


def generate(
    client,
    question: str,
    context: str,
    model: str = DEFAULT_GENERATIVE_MODEL,
    prompt_id: str = "criteria",
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_output_tokens: int = 512,
) -> dict:
    """Ask a generative Gemini model for a label, a confidence and a short rationale."""
    from google.genai import types

    system, instructions = build_prompt(prompt_id, question)
    parts = [instructions]
    if PROMPTS[prompt_id]["criteria"]:
        parts.append("\n".join(f"- {lbl}: {desc}" for lbl, desc in LABEL_CRITERIA.items()))
    parts.append(f"Study abstract:\n{context}" if context else "No abstract was provided.")
    parts.append("Reply with the label, your confidence from 0 to 1, and a one-sentence rationale.")

    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["label", "confidence", "rationale"],
    }
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    start = time.perf_counter()
    response = _with_retry(
        lambda: client.models.generate_content(
            model=model, contents="\n\n".join(parts), config=config
        )
    )
    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.usage_metadata
    out = {
        "label": None,
        "confidence": None,
        "probabilities": None,
        "rationale": None,
        "latency_ms": latency_ms,
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
    }
    try:
        data = json.loads(response.text)
        label = str(data["label"]).strip().lower()
        if label in LABELS:
            out["label"] = label
            out["confidence"] = min(max(float(data.get("confidence", 0.0)), 0.0), 1.0)
            out["rationale"] = str(data.get("rationale", "")).strip()
    except (TypeError, ValueError, KeyError):
        pass  # counted as an invalid output by the caller
    return out


def judge_rationale(client, context: str, rationale: str) -> dict:
    """LLM-as-a-judge: Gemini Decision 1 rates how faithful a rationale is to the abstract."""
    result = _with_retry(
        lambda: client.decisions.create(
            model=DECISION_MODEL,
            input=f"Study abstract:\n{context}\n\nRationale to grade:\n{rationale}",
            questions={
                "faithfulness": {
                    "type": "score",
                    "instructions": "How faithful is the rationale to the study abstract?",
                    "criteria": JUDGE_LEVELS,
                }
            },
        )
    )
    answer = _get(result.answers, "faithfulness")
    max_level = len(JUDGE_LEVELS) - 1
    return {
        "score": float(_get(answer, "score")) / max_level,
        "confidence": float(_get(answer, "confidence")),
    }
