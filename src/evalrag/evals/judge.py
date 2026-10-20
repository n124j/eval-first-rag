"""Optional LLM-as-judge scoring. Noisy by nature; never read by the gate.

Takes a plain `complete(messages) -> str` callable so it depends on no LLM client.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of answers produced by a retrieval-augmented system.
You are given a question, the retrieved context, and the answer. Score two criteria
from 1 to 5, using only the context and the answer.

faithfulness: is every claim in the answer supported by the context?
  5 = every claim is supported by the context
  4 = all key claims are supported; a minor detail is unsupported
  3 = some claims are supported, others are not
  2 = most claims are unsupported
  1 = the answer contradicts or ignores the context

relevance: does the answer address the question that was asked?
  5 = fully and directly answers the question
  4 = answers the question with minor gaps or extra material
  3 = partly answers the question
  2 = mostly off-topic
  1 = does not answer the question

Reply with ONLY a JSON object, no other text:
{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "rationale": "<one or two sentences>"}
"""

_EXCERPT_CHARS = 200


class JudgeError(ValueError):
    """The judge reply could not be parsed into a valid score."""


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: int
    relevance: int
    rationale: str


def build_judge_messages(question: str, answer: str, context: list[str]) -> list[dict[str, str]]:
    """System rubric, then a user message with the question, numbered context and answer."""
    numbered = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(context, start=1))
    user = f"Question:\n{question}\n\nContext:\n{numbered}\n\nAnswer:\n{answer}\n"
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _first_object(reply: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    pos = reply.find("{")
    while pos != -1:
        try:
            value, _ = decoder.raw_decode(reply, pos)
        except json.JSONDecodeError:
            pos = reply.find("{", pos + 1)
            continue
        return value if isinstance(value, dict) else None
    return None


def _score(obj: dict[str, object], key: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError(f"{key} must be an integer 1-5, got {value!r}")
    return value


def judge_answer(
    complete: Callable[[list[dict[str, str]]], str],
    question: str,
    answer: str,
    context: list[str],
) -> JudgeScore:
    """Ask the judge once and parse its reply; raise `JudgeError` on anything malformed."""
    reply = complete(build_judge_messages(question, answer, context))
    excerpt = reply[:_EXCERPT_CHARS]
    obj = _first_object(reply)
    if obj is None:
        raise JudgeError(f"no JSON object in judge reply: {excerpt!r}")
    try:
        faithfulness = _score(obj, "faithfulness")
        relevance = _score(obj, "relevance")
        rationale = obj.get("rationale")
        if not isinstance(rationale, str):
            raise TypeError(f"rationale must be a string, got {rationale!r}")
    except (ValueError, TypeError) as exc:
        raise JudgeError(f"invalid judge reply ({exc}): {excerpt!r}") from exc
    return JudgeScore(faithfulness, relevance, rationale)
