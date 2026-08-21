"""Shared class-grouping contract for flare datasets and classifiers."""

from __future__ import annotations

from collections.abc import Sequence


BASE_CLASS_SYMBOLS = ("0", "A", "B", "C", "M", "X")
DEFAULT_CLASS_GROUPS = ("0AB", "C", "M", "X")


def normalize_class_groups(
    class_groups: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate and canonicalize an ordered partition of ``0/A/B/C/M/X``.

    Group order defines the downstream label IDs. Character order within each
    group is canonicalized to ``0, A, B, C, M, X`` and has no semantic effect.
    Every base symbol must occur exactly once across all groups.
    """

    if class_groups is None:
        groups = list(DEFAULT_CLASS_GROUPS)
    else:
        if isinstance(class_groups, (str, bytes)):
            raise TypeError(
                "class_groups must be a sequence such as "
                "['0AB', 'C', 'M', 'X'], not one string"
            )
        if not isinstance(class_groups, Sequence):
            raise TypeError("class_groups must be an ordered sequence of strings")
        groups = list(class_groups)

    if len(groups) < 2:
        raise ValueError("class_groups must define at least two output classes")

    allowed = set(BASE_CLASS_SYMBOLS)
    seen: set[str] = set()
    normalized: list[str] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, str) or not group:
            raise ValueError(f"class_groups[{group_index}] must be a non-empty string")
        invalid = sorted(set(group).difference(allowed))
        if invalid:
            raise ValueError(
                f"class_groups[{group_index}] contains invalid symbols {invalid}; "
                f"allowed symbols are {list(BASE_CLASS_SYMBOLS)}"
            )
        repeated_within = sorted(
            symbol for symbol in allowed if group.count(symbol) > 1
        )
        repeated_across = sorted(set(group).intersection(seen))
        repeated = sorted(set(repeated_within + repeated_across))
        if repeated:
            raise ValueError(
                "Every base class must occur exactly once; duplicated symbols: "
                f"{repeated}"
            )

        symbols = set(group)
        normalized.append(
            "".join(symbol for symbol in BASE_CLASS_SYMBOLS if symbol in symbols)
        )
        seen.update(symbols)

    missing = [symbol for symbol in BASE_CLASS_SYMBOLS if symbol not in seen]
    if missing:
        raise ValueError(
            "class_groups must contain every base class 0/A/B/C/M/X exactly once; "
            f"missing symbols: {missing}"
        )
    return tuple(normalized)


def build_raw_label_to_group(
    class_groups: Sequence[str] | None = None,
) -> dict[int, int]:
    """Return the raw NOAA label ID to grouped output label ID mapping."""

    normalized = normalize_class_groups(class_groups)
    symbol_to_group = {
        symbol: group_index
        for group_index, group in enumerate(normalized)
        for symbol in group
    }
    return {
        raw_label: symbol_to_group[symbol]
        for raw_label, symbol in enumerate(BASE_CLASS_SYMBOLS)
    }
