"""P3-04 (MVP): basic scoring and ``murmurs eval run``."""

import json
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
    }
    assert "{{PERSON_1|Rosa Mendez}}" in (run_dir / "tiny_ceca.annotated.txt").read_text()


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
