import json
from pathlib import Path

import pytest

from evalrag.evals.dataset import DatasetError, GoldenItem, load_golden


def item(**over: object) -> dict:
    base = {
        "id": "q1",
        "question": "How do I parse args?",
        "expected_sources": ["argparse"],
        "answerable": True,
        "kind": "single",
    }
    base.update(over)
    return base


def write(tmp_path: Path, *rows: object) -> Path:
    path = tmp_path / "golden.jsonl"
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def check_error(tmp_path: Path, rows: list, line: int, ident: str, **kw: object) -> str:
    with pytest.raises(DatasetError) as exc:
        load_golden(write(tmp_path, *rows), **kw)  # type: ignore[arg-type]
    msg = str(exc.value)
    assert f"line {line}" in msg
    assert ident in msg
    return msg


def test_loads_valid_items_in_file_order(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        item(id="b", question="Second?"),
        "",
        item(id="a", question="First?", kind="multi", expected_sources=["argparse", "os.path"]),
        item(
            id="c",
            question="Unknown?",
            kind="unanswerable",
            answerable=False,
            expected_sources=[],
        ),
    )
    items = load_golden(path, known_sources={"argparse", "os.path"})
    assert [i.id for i in items] == ["b", "a", "c"]
    assert items[0] == GoldenItem("b", "Second?", ("argparse",), True, "single")
    assert items[1].expected_sources == ("argparse", "os.path")


def test_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as exc:
        load_golden(write(tmp_path, item(), "{not json"))
    assert "line 2" in str(exc.value)


def test_non_object_line(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as exc:
        load_golden(write(tmp_path, "[1, 2]"))
    assert "line 1" in str(exc.value)


def test_missing_and_unknown_fields(tmp_path: Path) -> None:
    bad = item()
    del bad["kind"]
    check_error(tmp_path, [bad], 1, "q1")
    check_error(tmp_path, [item(), item(id="q2", question="Other?", extra=1)], 2, "q2")


def test_duplicate_id(tmp_path: Path) -> None:
    check_error(tmp_path, [item(), item(question="Other?")], 2, "q1")


def test_duplicate_question_case_insensitive(tmp_path: Path) -> None:
    rows = [item(), item(id="q2", question="  how DO I   parse args? ")]
    check_error(tmp_path, rows, 2, "q2")


def test_empty_id_and_question(tmp_path: Path) -> None:
    check_error(tmp_path, [item(), item(id="", question="Other?")], 2, "line 2")
    check_error(tmp_path, [item(id="q9", question="   ")], 1, "q9")


def test_bad_kind(tmp_path: Path) -> None:
    check_error(tmp_path, [item(kind="weird")], 1, "q1")


def test_unanswerable_must_not_be_answerable_or_have_sources(tmp_path: Path) -> None:
    check_error(
        tmp_path, [item(kind="unanswerable", answerable=True, expected_sources=[])], 1, "q1"
    )
    check_error(tmp_path, [item(kind="unanswerable", answerable=False)], 1, "q1")


def test_single_needs_answerable_and_exactly_one_source(tmp_path: Path) -> None:
    check_error(tmp_path, [item(answerable=False)], 1, "q1")
    check_error(tmp_path, [item(expected_sources=["argparse", "os.path"])], 1, "q1")


def test_multi_needs_two_distinct_sources(tmp_path: Path) -> None:
    check_error(tmp_path, [item(kind="multi", expected_sources=["argparse"])], 1, "q1")
    check_error(tmp_path, [item(kind="multi", expected_sources=["argparse", "argparse"])], 1, "q1")
    check_error(
        tmp_path,
        [item(kind="multi", answerable=False, expected_sources=["argparse", "os.path"])],
        1,
        "q1",
    )


def test_answerable_without_sources(tmp_path: Path) -> None:
    check_error(tmp_path, [item(expected_sources=[])], 1, "q1")


def test_unknown_source(tmp_path: Path) -> None:
    msg = check_error(tmp_path, [item(expected_sources=["nope"])], 1, "q1", known_sources={"os"})
    assert "nope" in msg
    # without known_sources there is no such check
    assert load_golden(write(tmp_path, item(expected_sources=["nope"])))
