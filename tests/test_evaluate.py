"""P3-04 (MVP): basic scoring and ``murmurs eval run``."""

import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import pytest
import yaml
from typer.testing import CliRunner

from murmurs.backends import M, registry
from murmurs.backends.registry import BackendKind
from murmurs.cli import app
from murmurs.config import PROVISIONAL_ENTITY_TYPES
from murmurs.errors import PipelineError
from murmurs.evalset import load_dir
from murmurs.evaluate import names_recall, overall, score_set
from murmurs.models import DetectorSource, LLMUsage, RedactionSpan

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
TYPES = [e.name for e in PROVISIONAL_ENTITY_TYPES]


def _span(start: int, end: int, entity_type: str) -> RedactionSpan:
    return RedactionSpan(
        id="s",
        start=start,
        end=end,
        entity_type=entity_type,
        tag="[X]",
        confidence=1.0,
        sources=[DetectorSource.LLM],
    )


def test_score_counts_found_covered_missed_and_precision() -> None:
    ceca = load_dir(FIXTURES, TYPES)[0]
    person, location, age, org, date = ceca.labels
    spans = [
        _span(person.start, person.start + 4, "PERSON"),  # partial overlap still counts
        _span(location.start, location.end, "ORGANIZATION"),  # covered, wrong type
        _span(org.start, org.end, "ORGANIZATION"),
        _span(0, 3, "PERSON"),  # false positive
        _span(age.start, age.end, "SOCIOECONOMIC"),  # a type with no labels
    ]
    score = score_set(ceca, spans)
    assert score.by_type["PERSON"].model_dump() == {
        "labels": 1,
        "found": 1,
        "covered": 1,
        "spans": 2,
        "true_spans": 1,
    }
    assert score.by_type["LOCATION"].found == 0 and score.by_type["LOCATION"].covered == 1
    assert score.by_type["ORGANIZATION"].precision == 0.5
    assert (
        score.by_type["SOCIOECONOMIC"].labels == 0
        and score.by_type["SOCIOECONOMIC"].precision is None
    )
    assert score.missed == [location.id, age.id, date.id] and score.uncovered == [date.id]


def test_overall_sums_sets_and_reports_names_recall() -> None:
    sets = load_dir(FIXTURES, TYPES)
    scores = [
        score_set(s, [_span(s.labels[0].start, s.labels[0].end, s.labels[0].entity_type)])
        for s in sets
    ]
    total = overall(scores)
    assert total["PERSON"].labels == 2 and names_recall(total) == (1, 2)


# --- the CLI ---------------------------------------------------------------------------------


class _FindAll:
    """A scripted LLM that reports every label of the fixture set (by surface)."""

    model_version = "scripted"
    usage = LLMUsage()  # counts only; scripted LLMs report no tokens

    def __init__(self, surfaces: dict[str, str]) -> None:
        self.surfaces = surfaces

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        body = prompt.split("never report entities from CONTEXT lines.", 1)[1]
        found = []
        for line in body.splitlines():
            if line.startswith("[") and "] " in line:
                number = int(line[1 : line.index("]")])
                for surface, entity_type in self.surfaces.items():
                    if surface in line:
                        found.append({"line": number, "text": surface, "entity_type": entity_type})
        return response_model.model_validate({"entities": found})

    def close(self) -> None:
        pass


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    surfaces = {lb.text: lb.entity_type for s in load_dir(FIXTURES, TYPES) for lb in s.labels}
    registry.register(BackendKind.LLM, "scripted", lambda _: _FindAll(surfaces))
    yield


def _config(tmp_path: Path, **deid: Any) -> Path:
    raw = {
        "data": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")},
        "deid": {
            "ner_model": "stub",
            "llm_model": "scripted",
            "entity_types": [e.model_dump(mode="json") for e in PROVISIONAL_ENTITY_TYPES],
            **deid,
        },
    }
    (tmp_path / "in").mkdir()
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_eval_run_prints_counts_only_and_writes_scores(tmp_path: Path, scripted: None) -> None:
    result = CliRunner().invoke(app, ["eval", "run", str(FIXTURES), "-c", str(_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "names recall: 2/2 (100.0%)" in result.stdout
    assert "PERSON" in result.stdout and "found 1/1" in result.stdout
    for text in ("SENTINEL_PHI_7f3a", "Rosa", "Millbrook", "555-0100", "Lakeview"):
        assert text not in result.output
    [run_dir] = (tmp_path / "out" / "eval").iterdir()
    scores = json.loads((run_dir / "scores.json").read_text())
    assert scores["names_recall"] == {"found": 2, "labels": 2}
    assert {p.name for p in run_dir.iterdir()} == {
        "scores.json",
        "tiny_ceca.annotated.txt",
        "tiny_leds.annotated.txt",
        "tiny_ceca.detections.json",
        "tiny_leds.detections.json",
    }
    assert "{{PERSON_1|Rosa Mendez}}" in (run_dir / "tiny_ceca.annotated.txt").read_text()
    for name in ("tiny_ceca.detections.json", "tiny_leds.detections.json"):
        saved = (run_dir / name).read_text()
        for text in ("Rosa", "Millbrook", "555-0100", "Lakeview"):
            assert text not in saved  # offsets and types only


# --- resuming and rescoring -------------------------------------------------------------------


class _FailOn(_FindAll):
    """The scripted LLM, but its requests fail for one interview."""

    def __init__(self, surfaces: dict[str, str], fail_on: str) -> None:
        super().__init__(surfaces)
        self.fail_on = fail_on

    def generate(
        self, prompt: str, response_model: type[M], file_id: str, system: str | None = None
    ) -> M:
        if file_id == self.fail_on:
            raise PipelineError("LLM_REQUEST_FAILED", "detail with text")
        return super().generate(prompt, response_model, file_id, system)


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Registers the scripted LLM (finds every label) and records each LLM created.
    Appending ``"fail:<set_id>"`` makes LLMs created afterwards fail on that interview."""
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))
    surfaces = {lb.text: lb.entity_type for s in load_dir(FIXTURES, TYPES) for lb in s.labels}
    calls: list[str] = []

    def factory(_: object) -> _FindAll:
        calls.append("created")
        fail = [c for c in calls if c.startswith("fail:")]
        return _FailOn(surfaces, fail[-1][5:]) if fail else _FindAll(surfaces)

    registry.register(BackendKind.LLM, "scripted", factory)
    yield calls


def _run(evalset_dir: Path, config: Path, out: Path) -> Any:
    args = ["eval", "run", str(evalset_dir), "-c", str(config), "--out", str(out)]
    return CliRunner().invoke(app, args)


def test_a_rerun_skips_detected_interviews_and_starts_no_models(
    tmp_path: Path, llm_calls: list[str]
) -> None:
    config, out = _config(tmp_path), tmp_path / "results"
    first = _run(FIXTURES, config, out)
    assert first.exit_code == 0, first.output
    assert "already detected: 0  to detect: 2" in first.stdout
    assert llm_calls == ["created"]

    second = _run(FIXTURES, config, out)
    assert second.exit_code == 0, second.output
    assert "already detected: 2  to detect: 0" in second.stdout
    assert llm_calls == ["created"]  # no LLM server for a run with nothing to detect
    assert "names recall: 2/2 (100.0%)" in second.stdout
    assert json.loads((out / "scores.json").read_text())["names_recall"]["found"] == 2


def test_a_stopped_run_keeps_finished_interviews_and_resumes(
    tmp_path: Path, llm_calls: list[str]
) -> None:
    config, out = _config(tmp_path), tmp_path / "results"
    llm_calls.append("fail:tiny_leds")
    stopped = _run(FIXTURES, config, out)
    assert stopped.exit_code == 2
    assert "evaluation stopped (LLM_REQUEST_FAILED)" in stopped.stderr
    assert f"1 of 2 interviews detected in this run; rerun with --out {out}" in stopped.stderr
    assert "detail with text" not in stopped.output
    assert (out / "tiny_ceca.detections.json").exists()
    assert not (out / "tiny_leds.detections.json").exists()
    assert not (out / "scores.json").exists()

    llm_calls.clear()
    resumed = _run(FIXTURES, config, out)
    assert resumed.exit_code == 0, resumed.output
    assert "already detected: 1  to detect: 1" in resumed.stdout
    assert "names recall: 2/2 (100.0%)" in resumed.stdout


def test_changed_settings_detect_again(tmp_path: Path, llm_calls: list[str]) -> None:
    out = tmp_path / "results"
    assert _run(FIXTURES, _config(tmp_path), out).exit_code == 0
    changed = tmp_path / "changed"
    changed.mkdir()
    config = _config(changed, llm_server={"reasoning_effort": "low"})
    result = _run(FIXTURES, config, out)
    assert result.exit_code == 0, result.output
    assert "already detected: 0  to detect: 2" in result.stdout


def test_changed_labels_rescore_without_detecting(tmp_path: Path, llm_calls: list[str]) -> None:
    evalset_dir, out, config = tmp_path / "set", tmp_path / "results", _config(tmp_path)
    shutil.copytree(FIXTURES, evalset_dir)
    assert _run(evalset_dir, config, out).exit_code == 0

    ceca = evalset_dir / "tiny_ceca.md"
    ceca.write_text(re.sub(r"\{\{PERSON\|([^}|]+)\}\}", r"\1", ceca.read_text()))  # unlabel
    result = _run(evalset_dir, config, out)
    assert result.exit_code == 0, result.output
    assert "already detected: 2  to detect: 0" in result.stdout and llm_calls == ["created"]
    assert "names recall: 1/1 (100.0%)" in result.stdout


def test_eval_run_exits_2_when_models_are_not_set_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("MURMURS_CACHE", str(tmp_path / "cache"))
    path = _config(
        tmp_path, ner_model="OpenMed/privacy-filter-nemotron-v2", llm_model="openai/gpt-oss-120b"
    )
    result = CliRunner().invoke(app, ["eval", "run", str(FIXTURES), "-c", str(path)])
    assert result.exit_code == 2 and "murmurs models download" in result.stderr
