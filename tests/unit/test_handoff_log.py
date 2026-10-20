import pytest

from tools.handoff_log import Handoff, HandoffError, append, read_all

TS = "2026-10-19T10:00:00Z"


def mk(**kw):
    base = {
        "project": "p1",
        "from_role": "developer",
        "to_role": "tester",
        "summary": "done",
        "verdict": "submitted",
        "ts": TS,
    }
    base.update(kw)
    return Handoff(**base)


def test_round_trip(tmp_path):
    log = tmp_path / "sub" / "h.jsonl"
    h = mk()
    append(log, h)
    assert read_all(log) == [h]


def test_missing_and_empty_file(tmp_path):
    log = tmp_path / "h.jsonl"
    assert read_all(log) == []
    log.write_text("")
    assert read_all(log) == []


@pytest.mark.parametrize(
    "kw",
    [
        {"from_role": "ceo"},
        {"to_role": "ceo"},
        {"verdict": "maybe"},
        {"summary": ""},
        {"summary": "   "},
        {"ts": "yesterday"},
        {"ts": "2026-10-19T10:00:00"},
    ],
)
def test_invalid_rejected_and_file_untouched(tmp_path, kw):
    log = tmp_path / "h.jsonl"
    with pytest.raises(HandoffError):
        append(log, mk(**kw))
    assert not log.exists() or log.read_text() == ""


def test_ts_offset_form_accepted(tmp_path):
    log = tmp_path / "h.jsonl"
    append(log, mk(ts="2026-10-19T10:00:00+00:00"))
    assert len(read_all(log)) == 1


def test_deployer_requires_tester_signoff(tmp_path):
    log = tmp_path / "h.jsonl"
    d = mk(from_role="deployer", to_role="manager")
    with pytest.raises(HandoffError, match="tester sign-off"):
        append(log, d)
    assert not log.exists() or log.read_text() == ""
    append(log, mk(from_role="tester", to_role="deployer", verdict="approved", project="other"))
    with pytest.raises(HandoffError, match="tester sign-off"):
        append(log, d)
    append(log, mk(from_role="tester", to_role="deployer", verdict="approved"))
    append(log, d)
    assert len(read_all(log)) == 3


def test_deployer_non_submitted_needs_no_signoff(tmp_path):
    append(tmp_path / "h.jsonl", mk(from_role="deployer", to_role="manager", verdict="blocked"))


def test_append_only_two_lines(tmp_path):
    log = tmp_path / "h.jsonl"
    append(log, mk())
    first = log.read_text()
    append(log, mk(summary="again"))
    text = log.read_text()
    assert text.startswith(first)
    assert len(text.splitlines()) == 2


def test_malformed_line_names_line_number(tmp_path):
    log = tmp_path / "h.jsonl"
    append(log, mk())
    with log.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    with pytest.raises(HandoffError, match="line 2"):
        read_all(log)
