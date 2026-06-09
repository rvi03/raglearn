"""Validation gate for LLM-generated SQL on the facts store.

A language model writing SQL over financial data is the riskiest path in the
system: a plausible-but-wrong query returns a confidently wrong figure, and a
malicious or careless one could mutate or exfiltrate data. This gate makes the
generated SQL safe to execute by reducing what the model can do to a single,
narrow shape:

  * exactly one statement, and it must be a ``SELECT`` (no DDL/DML, no multiple
    statements, no ``PRAGMA``/``ATTACH``/``COPY``/``SET``, no CTEs/subqueries/
    set-operations);
  * it may only read the three allow-listed facts tables;
  * the projection is **overwritten** with the canonical fact columns, so the
    model only controls the filtering (joins/where) — never which columns or
    values come back; rows then map positionally to :class:`FinancialFact`;
  * a row ``LIMIT`` is enforced (added when absent, clamped when too high).

The parse/validate/rewrite is done on a sqlglot AST, not by string matching, so
comment- or literal-hidden tricks cannot slip through. Anything that does not fit
the shape raises :class:`SqlGuardError`; the caller abstains rather than guess.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from finrag.core.errors import FinragError

# Tables the generated query may read, and the only columns it may reference.
_ALLOWED_TABLES = {"financial_facts", "filings", "collections"}
_ALLOWED_COLUMNS = {
    # financial_facts
    "fact_id",
    "filing_id",
    "concept",
    "value",
    "unit",
    "period",
    "dimension",
    "origin",
    # filings
    "fiscal_year",
    "fiscal_period",
    "filing_type",
    "logical_key",
    "version",
    # collections
    "collection_id",
    "company",
    "ticker",
    "cik",
    "market",
    "status",
}
# The projection forced onto every query, in the order FinancialFact expects.
_FACT_COLUMNS = (
    "fact_id",
    "filing_id",
    "concept",
    "value",
    "unit",
    "period",
    "dimension",
    "origin",
)
# Nodes that must never appear: writes, schema changes, raw commands, and compound
# shapes we do not support in v1 (kept tight on purpose).
_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Set,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.With,
    exp.Subquery,
)
# Hard ceiling on rows a generated query may return.
_MAX_ROWS = 50


class SqlGuardError(FinragError):
    """Raised when generated SQL is not a safe, allow-listed ``SELECT``."""


def validate_select(sql: str, *, max_rows: int = _MAX_ROWS) -> str:
    """Validate and normalise LLM-generated SQL into a safe fact-reading ``SELECT``.

    Args:
      sql: The raw SQL the model produced.
      max_rows: Row ceiling enforced on the result.

    Returns:
      Sanitised SQL: a single ``SELECT`` of the canonical fact columns over the
      allow-listed tables, with an enforced ``LIMIT``.

    Raises:
      SqlGuardError: If the SQL is unparseable, not a lone ``SELECT``, references a
        table/column outside the allow-list, or contains a forbidden construct.
    """
    try:
        statements = sqlglot.parse(sql)
    except Exception as exc:  # sqlglot.errors.ParseError and friends
        raise SqlGuardError(f"unparseable SQL: {exc}") from exc

    real = [s for s in statements if s is not None]
    if len(real) != 1:
        raise SqlGuardError(f"expected exactly one statement, got {len(real)}")
    select = real[0]
    if not isinstance(select, exp.Select):
        raise SqlGuardError(f"only SELECT is allowed, got {type(select).__name__}")

    for node in select.walk():
        if isinstance(node, _FORBIDDEN):
            raise SqlGuardError(f"forbidden construct: {type(node).__name__}")

    tables = {t.name for t in select.find_all(exp.Table)}
    if not tables or not tables.issubset(_ALLOWED_TABLES):
        raise SqlGuardError(f"tables not allowed: {sorted(tables - _ALLOWED_TABLES)}")
    if "financial_facts" not in tables:
        raise SqlGuardError("query must read from financial_facts")

    for col in select.find_all(exp.Column):
        if col.name and col.name not in _ALLOWED_COLUMNS:
            raise SqlGuardError(f"column not allowed: {col.name}")

    # Overwrite the projection with the canonical fact columns, qualified by the
    # facts table's alias — the model never controls what is returned.
    alias = _facts_alias(select)
    select.set("expressions", [exp.column(c, table=alias) for c in _FACT_COLUMNS])

    # Enforce a row ceiling: add a LIMIT when absent, clamp it when too high.
    limit = select.args.get("limit")
    current = _limit_value(limit)
    if current is None or current > max_rows:
        select.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return select.sql()


def _facts_alias(select: exp.Select) -> str:
    """Return the alias (or name) the ``financial_facts`` table is referenced by."""
    for table in select.find_all(exp.Table):
        if table.name == "financial_facts":
            return table.alias_or_name
    return "financial_facts"  # unreachable: caller already checked the table is present


def _limit_value(limit: exp.Expression | None) -> int | None:
    """Return the integer row count of a ``LIMIT`` node, or ``None`` if absent/non-literal."""
    if limit is None:
        return None
    expr = limit.expression if isinstance(limit, exp.Limit) else limit
    if isinstance(expr, exp.Literal) and expr.is_int:
        return int(expr.name)
    return None
