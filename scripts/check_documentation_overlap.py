"""Detect simple documentation content overlap via cosine similarity of shingles."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple


def shingles(text: str, k: int = 8) -> set[str]:
    tokens = text.split()
    return {" ".join(tokens[i : i + k]) for i in range(max(0, len(tokens) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main() -> int:
    docs = list(Path("docs").rglob("*.md"))
    content: Dict[Path, set[str]] = {}
    for p in docs:
        try:
            content[p] = shingles(p.read_text(encoding="utf-8"))
        except Exception:
            continue

    pairs: List[Tuple[Path, Path, float]] = []
    for a, b in itertools.combinations(content.keys(), 2):
        sim = jaccard(content[a], content[b])
        if sim >= 0.3:  # flag 30%+ overlap
            pairs.append((a, b, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, sim in pairs:
        print(f"{a} <> {b}: {sim:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



