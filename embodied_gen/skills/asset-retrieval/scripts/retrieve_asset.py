#!/usr/bin/env python3
"""Retrieve EmbodiedGen asset URDF paths from a CSV index."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

STOP_WORDS = {
    "a",
    "an",
    "and",
    "asset",
    "for",
    "in",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class AssetRecord:
    """Single asset entry from the CSV index."""

    uuid: str
    primary_category: str
    secondary_category: str
    category: str
    description: str
    generate_time: str
    relative_urdf_path: str
    absolute_urdf_path: str
    search_text: str
    primary_tokens: frozenset[str]
    secondary_tokens: frozenset[str]
    category_tokens: frozenset[str]
    description_tokens: frozenset[str]


@dataclass(frozen=True)
class SearchResult:
    """Ranked retrieval result."""

    score: float
    coverage: float
    record: AssetRecord

    def to_dict(self, use_relative_paths: bool) -> dict[str, object]:
        """Convert the result to JSON-friendly output."""
        urdf_path = (
            self.record.relative_urdf_path
            if use_relative_paths
            else self.record.absolute_urdf_path
        )
        return {
            "urdf_path": urdf_path,
            "score": round(self.score, 3),
            "coverage": round(self.coverage, 3),
            "uuid": self.record.uuid,
            "primary_category": self.record.primary_category,
            "secondary_category": self.record.secondary_category,
            "category": self.record.category,
            "description": self.record.description,
            "generate_time": self.record.generate_time,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_dataset_root() -> Path:
    configured_root = os.getenv("EMBODIEDGEN_DATASET_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return _repo_root() / "outputs" / "EmbodiedGenData" / "dataset"


def _default_index_file(dataset_root: Path) -> Path:
    configured_index = os.getenv("EMBODIEDGEN_DATASET_INDEX")
    if configured_index:
        return Path(configured_index).expanduser().resolve()
    return dataset_root / "dataset_index.csv"


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("_", " ").replace("-", " ").replace("&", " and ")
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if (
        token.endswith("s")
        and len(token) > 3
        and not token.endswith(("ss", "us"))
    ):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(_normalize_text(text)):
        token = _normalize_token(raw_token)
        if len(token) < 2 or token in STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)
    return deduped


def load_records(index_file: Path, dataset_root: Path) -> list[AssetRecord]:
    """Load asset records from dataset_index.csv."""
    records: list[AssetRecord] = []
    with index_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            relative_urdf_path = (row.get("urdf_path") or "").strip()
            absolute_urdf_path = str(
                (dataset_root / relative_urdf_path).resolve()
            )

            primary_category = row.get("primary_category", "")
            secondary_category = row.get("secondary_category", "")
            category = row.get("category", "")
            description = row.get("description", "")

            records.append(
                AssetRecord(
                    uuid=row.get("uuid", ""),
                    primary_category=primary_category,
                    secondary_category=secondary_category,
                    category=category,
                    description=description,
                    generate_time=row.get("generate_time", ""),
                    relative_urdf_path=relative_urdf_path,
                    absolute_urdf_path=absolute_urdf_path,
                    search_text=" ".join(
                        part
                        for part in (
                            _normalize_text(primary_category),
                            _normalize_text(secondary_category),
                            _normalize_text(category),
                            _normalize_text(description),
                        )
                        if part
                    ),
                    primary_tokens=frozenset(_tokenize(primary_category)),
                    secondary_tokens=frozenset(_tokenize(secondary_category)),
                    category_tokens=frozenset(_tokenize(category)),
                    description_tokens=frozenset(_tokenize(description)),
                )
            )
    return records


def _score_record(
    record: AssetRecord,
    query_text: str,
    query_tokens: list[str],
) -> SearchResult | None:
    matched_tokens = 0
    score = 0.0

    for token in query_tokens:
        token_score = 0.0
        if token in record.category_tokens:
            token_score = max(token_score, 8.0)
        if token in record.secondary_tokens:
            token_score = max(token_score, 5.0)
        if token in record.primary_tokens:
            token_score = max(token_score, 3.0)
        if token in record.description_tokens:
            token_score = max(token_score, 2.0)

        if token_score > 0:
            matched_tokens += 1
            score += token_score

    if query_text and query_text in record.search_text:
        score += 8.0

    if matched_tokens == 0:
        return None

    coverage = matched_tokens / len(query_tokens)
    score += 4.0 * coverage
    return SearchResult(score=score, coverage=coverage, record=record)


def search_assets(
    records: list[AssetRecord],
    query: str,
    top_k: int,
) -> list[SearchResult]:
    """Return top-k lexical matches for a query."""
    query_text = _normalize_text(query)
    query_tokens = _dedupe_tokens(_tokenize(query))
    if not query_text or not query_tokens:
        raise ValueError(
            "Query must contain searchable keywords after normalization."
        )

    ranked: list[SearchResult] = []
    for record in records:
        result = _score_record(record, query_text, query_tokens)
        if result is not None:
            ranked.append(result)

    ranked.sort(
        key=lambda result: (
            -result.score,
            -result.coverage,
            -int(result.record.generate_time or 0),
            result.record.absolute_urdf_path,
        ),
    )
    return ranked[:top_k]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve EmbodiedGen asset URDF paths from dataset_index.csv."
    )
    parser.add_argument("query", help="Natural-language asset query.")
    parser.add_argument(
        "--dataset-root",
        default=str(_default_dataset_root()),
        help=(
            "Dataset root. "
            "Default: $EMBODIEDGEN_DATASET_ROOT or repo-relative dataset path."
        ),
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help=(
            "CSV index path. "
            "Default: $EMBODIEDGEN_DATASET_INDEX or <dataset-root>/dataset_index.csv."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of matches to return.",
    )
    parser.add_argument(
        "--format",
        choices=("paths", "json"),
        default="paths",
        help="Output format.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Return dataset-relative URDF paths instead of absolute paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    index_file = (
        Path(args.index_file).expanduser().resolve()
        if args.index_file
        else _default_index_file(dataset_root)
    )
    if not index_file.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_file}")

    records = load_records(index_file=index_file, dataset_root=dataset_root)
    results = search_assets(
        records=records, query=args.query, top_k=args.top_k
    )
    if not results:
        raise SystemExit("No matching assets found.")

    if args.format == "json":
        payload = [
            result.to_dict(use_relative_paths=args.relative_paths)
            for result in results
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for result in results:
        urdf_path = (
            result.record.relative_urdf_path
            if args.relative_paths
            else result.record.absolute_urdf_path
        )
        print(urdf_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
