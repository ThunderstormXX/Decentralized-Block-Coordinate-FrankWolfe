from __future__ import annotations

from collections import Counter


def render_counts(winners: list[str]) -> str:
    counts = Counter(winners)
    total = len(winners)
    return (
        f"Summary: DBCFW better: {counts['DBCFW']}/{total}; "
        f"DFW better: {counts['DFW']}/{total}; ties: {counts['tie']}/{total}."
    )
