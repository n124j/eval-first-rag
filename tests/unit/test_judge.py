import ast
from pathlib import Path

import pytest

import evalrag.evals.judge as judge_mod
from evalrag.evals.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeError,
    JudgeScore,
    build_judge_messages,
    judge_answer,
)

Q, A, CTX = "What is X?", "X is Y [a#001].", ["chunk one", "chunk two"]


def ask(reply: str) -> JudgeScore:
    return judge_answer(lambda _m: reply, Q, A, CTX)


def test_judge_parses_valid_json() -> None:
    s = ask('{"faithfulness": 5, "relevance": 4, "rationale": "ok"}')
    assert s == JudgeScore(5, 4, "ok")


def test_judge_accepts_fenced_json() -> None:
    s = ask('Sure:\n```json\n{"faithfulness": 3, "relevance": 2, "rationale": "r {x}"}\n```\nDone')
    assert s == JudgeScore(3, 2, "r {x}")


def test_judge_takes_first_object() -> None:
    s = ask('{"faithfulness": 1, "relevance": 1, "rationale": "a"} {"faithfulness": 5}')
    assert s.faithfulness == 1


@pytest.mark.parametrize(
    "reply",
    [
        "no json here",
        "",
        "{broken",
        '{"faithfulness": 5, "relevance": 4}',
        '{"faithfulness": 5, "rationale": "x"}',
        '{"faithfulness": 0, "relevance": 4, "rationale": "x"}',
        '{"faithfulness": 5, "relevance": 6, "rationale": "x"}',
        '{"faithfulness": 4.5, "relevance": 4, "rationale": "x"}',
        '{"faithfulness": 5.0, "relevance": 4, "rationale": "x"}',
        '{"faithfulness": true, "relevance": 4, "rationale": "x"}',
        '{"faithfulness": 5, "relevance": "4", "rationale": "x"}',
        '{"faithfulness": 5, "relevance": 4, "rationale": 7}',
        '["faithfulness"]',
    ],
)
def test_judge_rejects_garbage_missing_keys_out_of_range_float_and_bool(reply: str) -> None:
    with pytest.raises(JudgeError):
        ask(reply)


def test_judge_error_includes_excerpt() -> None:
    with pytest.raises(JudgeError, match="no json here"):
        ask("no json here")


def test_judge_prompt_is_deterministic() -> None:
    m1 = build_judge_messages(Q, A, CTX)
    assert m1 == build_judge_messages(Q, A, CTX)
    assert [m["role"] for m in m1] == ["system", "user"]
    assert m1[0]["content"] == JUDGE_SYSTEM_PROMPT
    for text in (Q, A, "chunk one", "chunk two"):
        assert text in m1[1]["content"]
    for word in ("faithfulness", "relevance", "JSON"):
        assert word in JUDGE_SYSTEM_PROMPT


def test_judge_calls_complete_once() -> None:
    calls: list[list[dict[str, str]]] = []

    def complete(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"faithfulness": 5, "relevance": 5, "rationale": "x"}'

    judge_answer(complete, Q, A, CTX)
    assert calls == [build_judge_messages(Q, A, CTX)]


def test_judge_module_has_no_generation_import() -> None:
    tree = ast.parse(Path(judge_mod.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("generation" in a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "generation" not in (node.module or "")
