"""Import human-deidentified transcripts as an evaluation set.

The input is two folders of ``.txt`` transcripts, the raw copies and the copies a human
deidentified, paired by their path relative to each folder; a CSV column of file names
can limit the import to the transcripts a reviewer finished (``select_pairs``). Each turn
starts with a line such as ``SPEAKER_1:`` or ``Speaker 1:``; a line without that prefix
continues the turn before it, and lines before the first turn are skipped. The
deidentified copy replaces PHI with bracketed tags such as ``[NAME]`` or
``[PARTICIPANT NAME]``, worded differently by different reviewers. A tag mapping (YAML,
tag -> entity type or ``IGNORE``) gives each tag its entity type; tags match ignoring
case, spacing and brackets.

Each pair is aligned token by token. Where tags replace raw tokens, the raw tokens become
a ground-truth label of the tag's type. Bracketed notes that both copies share, such as
``[inaudible]``, are ordinary text. The output is one interview per pair in the
evaluation-set format (``murmurs.evalset``) with ``SPEAKER_NN`` turns, ready for
``murmurs eval check`` and ``murmurs eval run``. The interview type comes from the file
name, which must contain ``CECA`` or ``LEDS``.

Nothing here prints, logs or raises transcript text or tag wording (tags are written by
reviewers and could hold PHI): reports hold counts and file names only, and the tag list
is written only to the mapping file.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from murmurs.models import InterviewType
from murmurs.store import atomic_writer

IGNORE = "IGNORE"
"""Mapping value for tags that do not mark PHI (they produce no label)."""

_SPEAKER_LINE = re.compile(r"^\s*speaker[\s_]*(\d+)\s*:\s?(.*)$", re.IGNORECASE)
"""``SPEAKER_1:``, ``Speaker 0:`` or ``speaker2:``, with the turn's text after it on the
same line or on the lines below."""
_TOKEN = re.compile(r"\[[^\[\]\n]*\]|\w+|[^\w\s]")
_TAG = re.compile(r"\[[^\[\]\n]*\]")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_TURN_MARK = "\x00turn"
"""Prefix of the token that starts each turn in the aligned streams, so the alignment
keeps turns in step and a label never crosses a turn boundary."""


class EvalImportError(ValueError):
    """The import cannot go ahead. Messages name files, folders and counts only."""


# --- reading transcripts --------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One speaker turn, its whitespace collapsed to single spaces."""

    speaker: int
    text: str


def parse_turns(source: str) -> tuple[list[Turn], int]:
    """The turns of a ``Speaker N:`` transcript and the number of lines skipped before the
    first turn. Turns with no text are dropped."""
    turns: list[tuple[int, list[str]]] = []
    skipped = 0
    for line in source.splitlines():
        if not line.strip():
            continue
        match = _SPEAKER_LINE.match(line)
        if match is not None:
            turns.append((int(match.group(1)), [match.group(2)]))
        elif turns:
            turns[-1][1].append(line)
        else:
            skipped += 1
    parsed = [Turn(speaker, " ".join(" ".join(parts).split())) for speaker, parts in turns]
    return [turn for turn in parsed if turn.text], skipped


def normalize_tag(tag: str) -> str:
    """A tag's mapping key: brackets removed, spacing collapsed, upper case."""
    inner = tag.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    return " ".join(inner.split()).upper()


def interview_type_of(name: str) -> InterviewType | None:
    """CECA or LEDS from a file name, or None if it names neither or both."""
    upper = name.upper()
    found = [kind for kind in InterviewType if kind.value in upper]
    return found[0] if len(found) == 1 else None


# --- alignment -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    text: str
    turn: int
    start: int
    end: int

    @property
    def is_mark(self) -> bool:
        return self.text.startswith(_TURN_MARK)

    @property
    def is_tag(self) -> bool:
        return _TAG.fullmatch(self.text) is not None


def _tokens(turns: Sequence[Turn]) -> list[_Token]:
    tokens = []
    for index, turn in enumerate(turns):
        tokens.append(_Token(f"{_TURN_MARK}{turn.speaker}", index, 0, 0))
        tokens.extend(
            _Token(m.group(), index, m.start(), m.end()) for m in _TOKEN.finditer(turn.text)
        )
    return tokens


@dataclass(frozen=True)
class TaggedSpan:
    """Raw text a reviewer replaced with a tag: ``turns[turn].text[start:end]``."""

    turn: int
    start: int
    end: int
    tag: str
    """The normalized tag (see ``normalize_tag``)."""


@dataclass
class Alignment:
    """What a raw/deidentified pair's alignment found. Counts are of diff blocks."""

    spans: list[TaggedSpan] = field(default_factory=list)
    tags: Counter[str] = field(default_factory=Counter)
    """Every normalized tag that marks a change, with its occurrences."""
    untagged_edits: int = 0
    """Text the reviewer changed without a tag (typo fixes, rewording)."""
    deleted: int = 0
    """Raw text removed with no tag in its place."""
    tags_without_text: int = 0
    """Tags inserted where the raw copy has no text."""
    unaligned_tags: int = 0
    """Tags whose raw text could not be pinned down (mixed with other edits, a different
    number of tags and words, or across a turn boundary); they produce no label."""


def align(raw: Sequence[Turn], deid: Sequence[Turn]) -> Alignment:
    """Align a raw transcript with its deidentified copy and find the tagged spans."""
    a, b = _tokens(raw), _tokens(deid)
    matcher = SequenceMatcher(None, [t.text for t in a], [t.text for t in b], autojunk=False)
    result = Alignment()
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        old, new = a[i1:i2], b[j1:j2]
        tags = [normalize_tag(t.text) for t in new if t.is_tag]
        result.tags.update(tags)
        if not tags:
            if op == "delete":
                result.deleted += 1
            else:
                result.untagged_edits += 1
        elif op == "insert":
            result.tags_without_text += len(tags)
        else:
            found = _tagged_spans(old, new, tags)
            if found is None:
                result.unaligned_tags += len(tags)
            else:
                spans, edits = found
                result.spans.extend(spans)
                result.untagged_edits += edits
    return result


def _tagged_spans(
    old: Sequence[_Token], new: Sequence[_Token], tags: Sequence[str]
) -> tuple[list[TaggedSpan], int] | None:
    """Spans for a block where tags replace the words ``old``, and the number of other
    words changed alongside them; None if the block cannot be pinned down.

    Pinned down means: one tag for the whole block, one tag per word, or (when the
    reviewer also fixed a neighbouring word) both sides the same length, paired word for
    word."""
    if any(t.is_mark for t in [*old, *new]) or len({t.turn for t in old}) != 1:
        return None
    if len(tags) == len(new) == 1:
        return [TaggedSpan(old[0].turn, old[0].start, old[-1].end, tags[0])], 0
    if len(old) != len(new):  # e.g. two tags for three words
        return None
    spans, edits = [], 0
    for before, after in zip(old, new, strict=True):  # e.g. "[FIRST NAME] [LAST NAME]"
        if after.is_tag:
            spans.append(
                TaggedSpan(before.turn, before.start, before.end, normalize_tag(after.text))
            )
        else:
            edits += 1
    return spans, edits


# --- tag mapping ---------------------------------------------------------------------------


def load_mapping(path: Path) -> dict[str, str | None]:
    """Read a tag mapping file: normalized tag -> entity type, ``IGNORE``, or None when
    the entry is still blank."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise EvalImportError(f"cannot read tag mapping {path} ({type(exc).__name__})") from None
    except yaml.YAMLError:
        raise EvalImportError(f"tag mapping {path} is not valid YAML") from None
    if not isinstance(raw, dict):
        raise EvalImportError(f"tag mapping {path} must be a YAML mapping")
    mapping: dict[str, str | None] = {}
    for key, value in raw.items():
        target = str(value).strip() if value is not None else ""
        mapping[normalize_tag(str(key))] = target or None
    return mapping


def check_mapping(mapping: dict[str, str | None], entity_types: Iterable[str]) -> None:
    """Every filled-in entry names a known entity type or ``IGNORE``. The error gives the
    unknown type names (written by the user), never the tags."""
    allowed = {*entity_types, IGNORE}
    unknown = sorted({v for v in mapping.values() if v is not None and v not in allowed})
    if unknown:
        raise EvalImportError(
            f"tag mapping uses unknown entity types: {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )


def write_mapping(
    path: Path,
    tags: Counter[str],
    existing: dict[str, str | None],
    entity_types: Sequence[str],
) -> int:
    """Write the mapping file: existing entries kept, new tags added blank, each with its
    occurrence count. Returns the number of new tags."""
    keys = sorted(set(existing) | set(tags), key=lambda k: (-tags[k], k))
    lines = [
        "# Tag mapping for `murmurs eval import`. Give each tag the entity type it marks,",
        f"# or {IGNORE} if it is not PHI. Entity types: {', '.join(entity_types)}.",
        "# Tags match ignoring case, spacing and brackets. Counts are occurrences found by",
        "# `murmurs eval tags`. This file lists reviewer tags: keep it with the data.",
    ]
    for key in keys:
        entry = yaml.safe_dump({key: existing.get(key) or ""}, allow_unicode=True, width=1000)
        lines.append(f"{entry.strip()}  # {tags[key]}")
    with atomic_writer(path) as f:
        f.write("\n".join(lines) + "\n")
    return len(set(tags) - set(existing))


# --- pairing and import --------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """A raw transcript and its deidentified copy."""

    relative: Path
    raw: Path
    deid: Path

    @property
    def set_id(self) -> str:
        """The interview's ID in the evaluation set: its relative path, made safe."""
        parts = [*self.relative.parent.parts, self.relative.stem]
        return "__".join(_UNSAFE_NAME.sub("_", part) for part in parts)


@dataclass(frozen=True)
class Pairing:
    """Files found in the two folders."""

    pairs: list[Pair]
    raw_only: list[Path]
    deid_only: list[Path]


def pair_files(raw_dir: Path, deid_dir: Path) -> Pairing:
    """Pair ``*.txt`` files by their path relative to each folder."""
    for directory in (raw_dir, deid_dir):
        if not directory.is_dir():
            raise EvalImportError(f"{directory} is not a directory")
    raw = {p.relative_to(raw_dir): p for p in raw_dir.rglob("*.txt") if p.is_file()}
    deid = {p.relative_to(deid_dir): p for p in deid_dir.rglob("*.txt") if p.is_file()}
    pairs = [Pair(rel, raw[rel], deid[rel]) for rel in sorted(raw.keys() & deid.keys())]
    ids = Counter(pair.set_id for pair in pairs)
    clashes = sorted(set_id for set_id, n in ids.items() if n > 1)
    if clashes:
        raise EvalImportError(f"file names clash after making them safe: {', '.join(clashes)}")
    return Pairing(pairs, sorted(raw.keys() - deid.keys()), sorted(deid.keys() - raw.keys()))


def file_key(name: str) -> str:
    """How a file name is matched against a list: spaces trimmed, a ``.txt`` extension
    dropped, case ignored."""
    name = name.strip()
    return (name[:-4] if name.lower().endswith(".txt") else name).casefold()


def load_file_list(path: Path, column: str) -> set[str]:
    """File-name keys (see ``file_key``) from one column of a CSV; empty cells skipped."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            if column not in columns:
                raise EvalImportError(
                    f"{path} has no column named {column!r} (columns: {', '.join(columns)})"
                )
            names = {file_key(row[column]) for row in reader if (row[column] or "").strip()}
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise EvalImportError(f"cannot read file list {path} ({type(exc).__name__})") from None
    if not names:
        raise EvalImportError(f"{path} lists no file names in column {column!r}")
    return names


def select_pairs(pairing: Pairing, names: set[str]) -> tuple[Pairing, list[str]]:
    """Only the pairs (and unpaired files) whose file name is in ``names``, and the listed
    names that match no pair, sorted."""

    def listed(path: Path) -> bool:
        return file_key(path.name) in names

    selected = Pairing(
        pairs=[pair for pair in pairing.pairs if listed(pair.relative)],
        raw_only=[path for path in pairing.raw_only if listed(path)],
        deid_only=[path for path in pairing.deid_only if listed(path)],
    )
    paired = {file_key(pair.relative.name) for pair in selected.pairs}
    return selected, sorted(names - paired)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalImportError(f"cannot read {path} ({type(exc).__name__})") from None


def align_pair(pair: Pair) -> tuple[list[Turn], int, Alignment]:
    """Read and align one pair: the raw turns, lines skipped before the raw transcript's
    first turn, and the alignment."""
    raw, skipped = parse_turns(_read(pair.raw))
    deid, _ = parse_turns(_read(pair.deid))
    return raw, skipped, align(raw, deid)


class FileReport(BaseModel):
    """Counts for one imported (or skipped) pair. No text."""

    file: str
    set_id: str
    imported: bool
    reason: str = ""
    """Why the pair was skipped."""
    interview_type: str = ""
    turns: int = 0
    skipped_lines: int = 0
    labels: dict[str, int] = Field(default_factory=dict)
    ignored_tags: int = 0
    untagged_edits: int = 0
    deleted: int = 0
    tags_without_text: int = 0
    unaligned_tags: int = 0


class ImportReport(BaseModel):
    """The whole import: per-file counts and unpaired files."""

    files: list[FileReport]
    raw_only: list[str]
    deid_only: list[str]
    not_found: list[str] = Field(default_factory=list)
    """Names in the file list (``--only``) that match no pair."""


@dataclass
class TagScan:
    """Tags found across the pairs, and pairs where a copy has no turns (a sign the line
    format is not ``SPEAKER_N:``)."""

    tags: Counter[str] = field(default_factory=Counter)
    turns: int = 0
    """Turns in the raw copies."""
    no_turns: list[Path] = field(default_factory=list)


def collect_tags(pairing: Pairing) -> TagScan:
    """Every normalized tag across the pairs, with its occurrences."""
    scan = TagScan()
    for pair in pairing.pairs:
        raw, _ = parse_turns(_read(pair.raw))
        deid, _ = parse_turns(_read(pair.deid))
        scan.turns += len(raw)
        if not raw or not deid:
            scan.no_turns.append(pair.relative)
        scan.tags.update(align(raw, deid).tags)
    return scan


def _render(
    set_id: str, kind: InterviewType, turns: Sequence[Turn], labels: list[tuple[TaggedSpan, str]]
) -> str:
    by_turn: dict[int, list[tuple[TaggedSpan, str]]] = {}
    for span, entity_type in labels:
        by_turn.setdefault(span.turn, []).append((span, entity_type))
    lines = [
        "---",
        f"interview_type: {kind.value}",
        "---",
        "# Imported by `murmurs eval import` from a raw and a human-deidentified transcript.",
    ]
    for index, turn in enumerate(turns):
        pieces, pos = [], 0
        for span, entity_type in sorted(by_turn.get(index, []), key=lambda x: x[0].start):
            pieces += [
                turn.text[pos : span.start],
                f"{{{{{entity_type}|{turn.text[span.start : span.end]}}}}}",
            ]
            pos = span.end
        pieces.append(turn.text[pos:])
        lines.append(f"SPEAKER_{turn.speaker:02d}: {''.join(pieces)}")
    return "\n".join(lines) + "\n"


def _representable(turns: Sequence[Turn], spans: Sequence[TaggedSpan]) -> bool:
    """Whether the evaluation-set format can hold this text: no ``{{``/``}}`` in turns and
    no ``{``, ``}`` or ``|`` inside a label."""
    if any("{{" in t.text or "}}" in t.text for t in turns):
        return False
    return not any(set("{}|") & set(turns[s.turn].text[s.start : s.end]) for s in spans)


def import_pairs(
    pairing: Pairing,
    mapping: dict[str, str | None],
    out_dir: Path,
    inputs: Sequence[Path],
    not_found: Sequence[str] = (),
) -> ImportReport:
    """Convert every pair to an evaluation interview in ``out_dir`` and write
    ``import_report.json`` there.

    Raises ``EvalImportError`` before writing anything if ``out_dir`` is not empty or is
    inside an input folder, or if any tag is not mapped yet. A pair whose file name names
    neither CECA nor LEDS, or whose text the format cannot hold, is skipped and reported.
    """
    target = out_dir.expanduser().absolute()
    for folder in inputs:
        folder = folder.expanduser().absolute()
        if target == folder or target.is_relative_to(folder) or folder.is_relative_to(target):
            raise EvalImportError("the output folder must be outside the input folders")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise EvalImportError(f"{target} must be a new or empty folder")

    aligned = [(pair, *align_pair(pair)) for pair in pairing.pairs]
    unmapped = {
        tag for _, _, _, result in aligned for tag in result.tags if mapping.get(tag) is None
    }
    if unmapped:
        raise EvalImportError(
            f"{len(unmapped)} tags have no entity type in the mapping; "
            "run `murmurs eval tags` to add them, then fill them in"
        )

    reports = []
    for pair, turns, skipped, result in aligned:
        report = FileReport(
            file=str(pair.relative),
            set_id=pair.set_id,
            imported=False,
            turns=len(turns),
            skipped_lines=skipped,
            untagged_edits=result.untagged_edits,
            deleted=result.deleted,
            tags_without_text=result.tags_without_text,
            unaligned_tags=result.unaligned_tags,
        )
        kind = interview_type_of(pair.relative.name)
        labels = [(s, mapping[s.tag] or "") for s in result.spans if mapping[s.tag] != IGNORE]
        report.ignored_tags = len(result.spans) - len(labels)
        if kind is None:
            report.reason = "file name names neither CECA nor LEDS (or both)"
        elif not turns:
            report.reason = "no Speaker N: turns"
        elif not _representable(turns, [s for s, _ in labels]):
            report.reason = "text holds {{, }} or a label holds {, } or |"
        else:
            with atomic_writer(target / f"{pair.set_id}.md") as f:
                f.write(_render(pair.set_id, kind, turns, labels))
            report.imported = True
            report.interview_type = kind.value
            report.labels = dict(sorted(Counter(t for _, t in labels).items()))
        reports.append(report)

    full = ImportReport(
        files=reports,
        raw_only=[str(p) for p in pairing.raw_only],
        deid_only=[str(p) for p in pairing.deid_only],
        not_found=list(not_found),
    )
    with atomic_writer(target / "import_report.json") as f:
        f.write(full.model_dump_json(indent=2))
    return full
