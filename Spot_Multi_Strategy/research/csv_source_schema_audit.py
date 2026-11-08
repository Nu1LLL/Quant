"""Value-blind CSV format audit for future preregistered studies.

This inspects lexical *shapes*, not numeric values, signals or returns. It is
separate from every sealed experiment parser and must never be used to rerun a
failed historical study.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from hashlib import sha256
import csv
import io
import re


_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_NONFINITE = frozenset({"nan", "+nan", "-nan", "inf", "+inf", "-inf",
                        "infinity", "+infinity", "-infinity"})


class SchemaAuditError(ValueError):
    """The CSV is not structurally safe for a numeric source parser."""


def _shape(token: str | None) -> str:
    if token is None:
        return "null_field"
    if token == "":
        return "empty"
    if token == ".":
        return "dot"
    if _NUMBER.fullmatch(token):
        return "numeric_lexeme"
    if token.lower() in _NONFINITE:
        return "nonfinite_lexeme"
    return "other_lexeme"


def audit_csv_schema(
    body: bytes,
    *,
    date_column: str,
    numeric_columns: tuple[str, ...],
    parser_missing_tokens: frozenset[str] = frozenset(),
) -> dict:
    """Count every field shape; reveal no numeric cell contents.

    An incompatible result flags a source/parser mismatch before preregistration.
    The caller still has to freeze source hashes, time semantics and cost rules.
    """
    if not numeric_columns or len(set(numeric_columns)) != len(numeric_columns):
        raise SchemaAuditError("numeric columns must be unique and nonempty")
    if date_column in numeric_columns:
        raise SchemaAuditError("date column overlaps numeric columns")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")),
                                restkey="__extra_fields__")
        expected = (date_column, *numeric_columns)
        if tuple(reader.fieldnames or ()) != expected:
            raise SchemaAuditError("header does not match expected columns")
        counts = {column: Counter() for column in numeric_columns}
        incompatible = {column: Counter() for column in numeric_columns}
        first = last = previous = None
        row_count = 0
        for row in reader:
            if set(row) != set(expected) or "__extra_fields__" in row:
                raise SchemaAuditError("row has missing or extra fields")
            day = date.fromisoformat(row[date_column])
            if previous is not None and day <= previous:
                raise SchemaAuditError("dates are duplicated or unsorted")
            first = first or day
            last = previous = day
            row_count += 1
            for column in numeric_columns:
                token = row[column]
                shape = _shape(token)
                counts[column][shape] += 1
                if shape != "numeric_lexeme" and token not in parser_missing_tokens:
                    incompatible[column][shape] += 1
        if not row_count:
            raise SchemaAuditError("no observations")
    except (UnicodeError, csv.Error, ValueError, TypeError) as exc:
        if isinstance(exc, SchemaAuditError):
            raise
        raise SchemaAuditError("invalid UTF-8, CSV or date structure") from exc
    return {
        "sha256": sha256(body).hexdigest(),
        "bytes": len(body),
        "rows": row_count,
        "first_date": first.isoformat(),
        "last_date": last.isoformat(),
        "column_shapes": {key: dict(value) for key, value in counts.items()},
        "incompatible_shapes": {key: dict(value) for key, value in incompatible.items()},
        "parser_compatible": all(not value for value in incompatible.values()),
    }
