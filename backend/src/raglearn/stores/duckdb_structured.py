"""DuckDB structured store: the persistent home for a filing's exact figures.

US exact figures are extracted from XBRL once and persisted here, where the
numeric retrieval path (metric-registry / text-to-SQL) reads them. A filing's
``collection`` and ``filing`` rows (from DEI) and its ``financial_facts`` are
written together in one transaction, so a fact always has its parent ``filings``
row and the foreign key holds. Everything is keyed so re-ingesting a filing is
idempotent (``ON CONFLICT DO NOTHING``); a restatement arrives under a new
accession, hence new keys, and coexists.

DuckDB is single-writer (one process holds the write lock). The ingestion
consumer is that single writer and its router runs off the event loop, so the
connection here is a single long-lived handle; query-time access is read-only.
"""

from __future__ import annotations

import duckdb

from raglearn.core.registry import registry
from raglearn.core.types import XbrlExtraction

# Run as separate statements: DuckDB's execute() runs one statement per call.
# Order matters — a table is created after the table its foreign key references.
_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS collections (
        collection_id VARCHAR PRIMARY KEY,
        company       VARCHAR,
        ticker        VARCHAR,
        cik           VARCHAR,
        market        VARCHAR NOT NULL,
        status        VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS filings (
        filing_id     VARCHAR PRIMARY KEY,
        collection_id VARCHAR NOT NULL REFERENCES collections (collection_id),
        filing_type   VARCHAR,
        fiscal_year   INTEGER,
        fiscal_period VARCHAR,
        filed_date    DATE,
        source_url    VARCHAR,
        version       INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS financial_facts (
        fact_id    VARCHAR PRIMARY KEY,
        filing_id  VARCHAR NOT NULL REFERENCES filings (filing_id),
        concept    VARCHAR NOT NULL,
        value      DOUBLE  NOT NULL,
        unit       VARCHAR NOT NULL,
        period     VARCHAR NOT NULL,
        dimension  VARCHAR,
        origin     VARCHAR NOT NULL CHECK (origin IN ('xbrl', 'extracted'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_filing     ON financial_facts (filing_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_concept    ON financial_facts (concept)",
    "CREATE INDEX IF NOT EXISTS idx_filings_collection ON filings (collection_id)",
)

_INSERT_COLLECTION = """
    INSERT INTO collections (collection_id, company, ticker, cik, market, status)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT (collection_id) DO NOTHING
"""

_INSERT_FILING = """
    INSERT INTO filings
        (filing_id, collection_id, filing_type, fiscal_year, fiscal_period,
         filed_date, source_url, version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (filing_id) DO NOTHING
"""

_INSERT_FACT = """
    INSERT INTO financial_facts
        (fact_id, filing_id, concept, value, unit, period, dimension, origin)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (fact_id) DO NOTHING
"""


@registry.register("structured_store", "duckdb")
class DuckdbStructuredStore:
    """A :class:`~raglearn.core.interfaces.StructuredStore` backed by DuckDB."""

    def __init__(self, path: str) -> None:
        """Open (or create) the database and ensure the schema exists.

        Args:
          path: Filesystem path to the DuckDB file, or ``":memory:"`` for an
            ephemeral in-process database (tests).
        """
        self._conn = duckdb.connect(path)
        for statement in _SCHEMA:
            self._conn.execute(statement)

    def write(self, extraction: XbrlExtraction) -> int:
        """Persist a filing's collection, filing row, and facts in one transaction.

        Written in foreign-key order (collection → filing → facts) so the facts'
        parent rows exist first. Each row is idempotent on its key, so re-ingesting
        a filing changes nothing. On any failure the whole write rolls back.

        Args:
          extraction: The filing's collection, filing, and facts.

        Returns:
          The number of facts submitted (facts already present leave their row
          untouched but are counted).
        """
        collection, filing = extraction.collection, extraction.filing
        fact_rows = [
            (
                fact.fact_id,
                fact.filing_id,
                fact.concept,
                fact.value,
                fact.unit,
                fact.period,
                fact.dimension,
                fact.origin.value,
            )
            for fact in extraction.facts
        ]
        self._conn.execute("BEGIN TRANSACTION")
        try:
            self._conn.execute(
                _INSERT_COLLECTION,
                [
                    collection.collection_id,
                    collection.company,
                    collection.ticker,
                    collection.cik,
                    collection.market.value,
                    collection.status,
                ],
            )
            self._conn.execute(
                _INSERT_FILING,
                [
                    filing.filing_id,
                    filing.collection_id,
                    filing.filing_type,
                    filing.fiscal_year,
                    filing.fiscal_period,
                    filing.filed_date,
                    filing.source_url,
                    filing.version,
                ],
            )
            if fact_rows:
                self._conn.executemany(_INSERT_FACT, fact_rows)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        return len(fact_rows)

    def close(self) -> None:
        """Close the connection, releasing the write lock."""
        self._conn.close()
