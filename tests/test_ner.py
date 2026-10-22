"""P3-01: the OpenMed privacy-filter NER backend."""

import pytest
from conftest import StubJob

from murmurs.backends import registry
from murmurs.backends.openmed_ner import (
    MODEL_ID,
    SPEC,
    RawSpan,
    clean_spans,
    decode_bioes,
    snap_to_words,
    windows,
)
from murmurs.backends.registry import BackendKind
from murmurs.config import load_config


@pytest.mark.parametrize("n", [0, 1, 2048, 2049, 5000, 30_000])
def test_windows_keep_every_token_exactly_once(n: int) -> None:
    kept = [i for w in windows(n) for i in range(w.keep_start, w.keep_end)]
    assert kept == list(range(n))
    for w in windows(n):
        assert w.start <= w.keep_start <= w.keep_end <= w.end <= n
        assert w.end - w.start <= 2048


def test_windows_overlap_and_keep_the_centre() -> None:
    first, second = windows(3000)[:2]
    assert second.start < first.end  # they overlap
    assert first.keep_end == second.keep_start  # the shared tokens are split between them
    assert second.keep_start - second.start == 128  # half the overlap


TEXT = "I met Maria Lopez in Oak Hollow today."
#       0123456789012345678901234567890123456789
OFFSETS = [(0, 1), (1, 5), (5, 11), (11, 17), (17, 20), (20, 24), (24, 31), (31, 37), (37, 38)]


def test_decode_bioes_builds_spans_with_mean_confidence() -> None:
    labels = ["O", "O", "S-first_name", "S-last_name", "O", "B-city", "E-city", "O", "O"]
    conf = [0.9, 0.9, 0.8, 0.6, 0.9, 0.5, 0.7, 0.9, 0.9]
    spans = decode_bioes(OFFSETS, labels, conf)
    assert [(TEXT[s.start : s.end], s.label) for s in spans] == [
        (" Maria", "first_name"),
        (" Lopez", "last_name"),
        (" Oak Hollow", "city"),
    ]
    assert spans[2].confidence == pytest.approx(0.6)


def test_decode_bioes_splits_on_label_change_and_o() -> None:
    labels = ["O", "O", "B-first_name", "I-last_name", "O", "I-city", "O", "O", "O"]
    spans = decode_bioes(OFFSETS, labels, [1.0] * 9)
    assert [(s.label, TEXT[s.start : s.end]) for s in spans] == [
        ("first_name", " Maria"),
        ("last_name", " Lopez"),
        ("city", " Oak"),  # an I- tag with no open span starts one
    ]


def test_snap_to_words_trims_and_widens() -> None:
    text = "Mrs. Halvorsen, next door"
    assert snap_to_words(text, 4, 8) == (5, 14)  # ' Hal' -> 'Halvorsen'
    assert snap_to_words(text, 14, 15) is None  # punctuation only
    assert snap_to_words(text, 0, 2) == (0, 3)  # 'Mr' -> 'Mrs'


def test_clean_spans_merges_overlaps() -> None:
    text = "Mrs. Halvorsen lives in Cedar Bend."
    raw = [
        RawSpan(4, 8, "first_name", 0.66),  # ' Hal'
        RawSpan(8, 14, "last_name", 0.68),  # 'vorsen' -> same word: keeps the more confident
        RawSpan(23, 29, "city", 0.9),
        RawSpan(29, 34, "city", 0.7),  # ' Bend' adjacent, same label: kept separate
    ]
    spans = clean_spans(text, raw)
    assert [(text[s.start : s.end], s.label) for s in spans] == [
        ("Halvorsen", "last_name"),
        ("Cedar", "city"),
        ("Bend", "city"),
    ]


def test_registry_declares_only_inference_files(stub_job: StubJob) -> None:
    config = load_config(stub_job(deid={"ner_model": MODEL_ID}))
    plan = registry.required_weights(config, [BackendKind.NER])
    assert plan.weights == [SPEC] and plan.unavailable == []
    assert "model.safetensors" in (SPEC.allow_patterns or ())


@pytest.mark.gpu
def test_real_ner_finds_planted_fake_phi(
    stub_job: StubJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    from murmurs.backends.openmed_ner import OpenMedNER
    from murmurs.offline import enforce_offline

    monkeypatch.delenv("HF_TOKEN", raising=False)
    enforce_offline()
    ner = OpenMedNER(load_config(stub_job(deid={"ner_model": MODEL_ID})))
    text = (
        "My aunt Rosa Mendez lived on Kestrel Lane in Harrow Falls.\n"
        "You can call her on 555-0147 or email rosa.mendez@example.org."
    )
    spans = ner.detect(text, "f1")
    found = {(text[s.start : s.end], s.entity_type) for s in spans}
    assert ("Rosa", "first_name") in found and ("Mendez", "last_name") in found
    assert any(t == "555-0147" for t, _ in found)
    assert all(text[s.start].isalnum() and text[s.end - 1].isalnum() for s in spans)
    ner.close()
