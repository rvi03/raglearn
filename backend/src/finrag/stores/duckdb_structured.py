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

from finrag.core.registry import registry
from finrag.core.types import FactOrigin, FinancialFact, XbrlExtraction

# Company-name suffixes stripped when matching a query to a collection.
_NAME_SUFFIXES = (
    " incorporated",
    " inc.",
    " inc",
    " co.",
    " co",
    " ltd.",
    " ltd",
    " corporation",
    " corp.",
    " corp",
    " plc",
)

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
        version       INTEGER NOT NULL DEFAULT 1,
        logical_key   VARCHAR
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
    """
    CREATE TABLE IF NOT EXISTS ingested_documents (
        content_hash VARCHAR PRIMARY KEY,
        doc_id       VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quarantine (
        doc_id          VARCHAR NOT NULL,
        detected_format VARCHAR NOT NULL,
        reason          VARCHAR NOT NULL,
        quarantined_at  TIMESTAMP DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS concept_index (
        collection_id VARCHAR NOT NULL,
        concept       VARCHAR NOT NULL,
        PRIMARY KEY (collection_id, concept)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_filing     ON financial_facts (filing_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_concept    ON financial_facts (concept)",
    "CREATE INDEX IF NOT EXISTS idx_filings_collection ON filings (collection_id)",
    "CREATE INDEX IF NOT EXISTS idx_filings_logical  ON filings (logical_key)",
)

_INSERT_COLLECTION = """
    INSERT INTO collections (collection_id, company, ticker, cik, market, status)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT (collection_id) DO NOTHING
"""

_INSERT_FILING = """
    INSERT INTO filings
        (filing_id, collection_id, filing_type, fiscal_year, fiscal_period,
         filed_date, source_url, version, logical_key)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (filing_id) DO NOTHING
"""

_INSERT_FACT = """
    INSERT INTO financial_facts
        (fact_id, filing_id, concept, value, unit, period, dimension, origin)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (fact_id) DO NOTHING
"""

_INSERT_CONCEPT = """
    INSERT INTO concept_index (collection_id, concept) VALUES (?, ?)
    ON CONFLICT (collection_id, concept) DO NOTHING
"""


@registry.register("structured_store", "duckdb")
class DuckdbStructuredStore:
    """A :class:`~finrag.core.interfaces.StructuredStore` backed by DuckDB."""

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
        concept_rows = sorted(
            {(collection.collection_id, fact.concept) for fact in extraction.facts}
        )
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
                    filing.logical_key,
                ],
            )
            if fact_rows:
                self._conn.executemany(_INSERT_FACT, fact_rows)
            if concept_rows:
                self._conn.executemany(_INSERT_CONCEPT, concept_rows)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        return len(fact_rows)

    def is_ingested(self, content_hash: str) -> bool:
        """Return whether a document with this content hash was already ingested."""
        row = self._conn.execute(
            "SELECT 1 FROM ingested_documents WHERE content_hash = ?", [content_hash]
        ).fetchone()
        return row is not None

    def mark_ingested(self, content_hash: str, doc_id: str) -> None:
        """Record that a document with this content hash has been ingested."""
        self._conn.execute(
            "INSERT INTO ingested_documents (content_hash, doc_id) VALUES (?, ?) "
            "ON CONFLICT (content_hash) DO NOTHING",
            [content_hash, doc_id],
        )

    def concepts(self, collection_id: str) -> list[str]:
        """Return the XBRL concepts a collection has structured facts for (sorted)."""
        rows = self._conn.execute(
            "SELECT concept FROM concept_index WHERE collection_id = ? ORDER BY concept",
            [collection_id],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def has_concept(self, collection_id: str, concept: str) -> bool:
        """Return whether a collection has a structured fact for a concept.

        A routing aid: lets the query router tell that an exact, structured answer
        exists for a concept before choosing the narrative path.
        """
        row = self._conn.execute(
            "SELECT 1 FROM concept_index WHERE collection_id = ? AND concept = ?",
            [collection_id, concept],
        ).fetchone()
        return row is not None

    def find_collection(self, text: str) -> tuple[str, str] | None:
        """Resolve a company mentioned in free text to a collection.

        Matches a ticker appearing as a whole word first (most specific), then a
        company name (minus its legal suffix) appearing as a substring. This is
        the entity-resolution the exact path needs to know *which* company's
        figures to read.

        Args:
          text: The query text, e.g. "What was Apple's FY2023 net sales?".

        Returns:
          ``(collection_id, company)`` for the first match, or ``None``.
        """
        tokens = set(text.lower().replace("'s", " ").replace(",", " ").split())
        rows = self._conn.execute(
            "SELECT collection_id, company, ticker FROM collections"
        ).fetchall()
        for collection_id, company, ticker in rows:
            if ticker and ticker.lower() in tokens:
                return str(collection_id), str(company or ticker)
        for collection_id, company, _ticker in rows:
            if not company:
                continue
            core = company.lower().replace(",", "")
            for suffix in _NAME_SUFFIXES:
                core = core.removesuffix(suffix)
            # Match on the distinctive first word of the name ("Northwind",
            # "Acme", "Apple") so possessives/partial mentions still resolve.
            words = core.split()
            if words and len(words[0]) > 2 and words[0] in tokens:
                return str(collection_id), str(company)
        return None

    def query_facts(self, collection_id: str, concepts: list[str]) -> list[FinancialFact]:
        """Return a collection's undimensioned facts for any of the given concepts.

        Only undimensioned facts (``dimension IS NULL``) are returned — these are
        the consolidated top-line figures, not segment/member breakdowns. Period
        selection (which fiscal year) is left to the caller, which has the query's
        intent.

        Args:
          collection_id: The company collection to read.
          concepts: Candidate XBRL concepts (e.g. revenue's several taggings).

        Returns:
          The matching facts (possibly across several filings/periods).
        """
        if not concepts:
            return []
        placeholders = ", ".join("?" * len(concepts))
        rows = self._conn.execute(
            "SELECT f.fact_id, f.filing_id, f.concept, f.value, f.unit, f.period, "
            "f.dimension, f.origin "
            "FROM financial_facts f JOIN filings fi ON f.filing_id = fi.filing_id "
            f"WHERE fi.collection_id = ? AND f.concept IN ({placeholders}) "
            "AND f.dimension IS NULL",
            [collection_id, *concepts],
        ).fetchall()
        return [
            FinancialFact(
                fact_id=str(row[0]),
                filing_id=str(row[1]),
                concept=str(row[2]),
                value=float(row[3]),
                unit=str(row[4]),
                period=str(row[5]),
                dimension=row[6],
                origin=FactOrigin(row[7]),
            )
            for row in rows
        ]

    def select_facts(self, sql: str) -> list[FinancialFact]:
        """Run a pre-validated read-only SELECT and map its rows to facts.

        The SQL must already have passed the SQL guard, which guarantees a single
        SELECT projecting the canonical fact columns over the allow-listed tables;
        rows map to :class:`FinancialFact` positionally.
        """
        rows = self._conn.execute(sql).fetchall()
        return [
            FinancialFact(
                fact_id=str(row[0]),
                filing_id=str(row[1]),
                concept=str(row[2]),
                value=float(row[3]),
                unit=str(row[4]),
                period=str(row[5]),
                dimension=row[6],
                origin=FactOrigin(row[7]),
            )
            for row in rows
        ]

    def quarantine(self, doc_id: str, detected_format: str, reason: str) -> None:
        """Record a document set aside for review instead of ingested."""
        self._conn.execute(
            "INSERT INTO quarantine (doc_id, detected_format, reason) VALUES (?, ?, ?)",
            [doc_id, detected_format, reason],
        )

    def latest_filing_id(self, logical_key: str) -> str | None:
        """Return the current filing for a logical document (highest version).

        The precedence signal is ``version``; once a filer's ``filed_date`` is
        populated (connector/header), order by that first. ``None`` if unknown.
        """
        row = self._conn.execute(
            "SELECT filing_id FROM filings WHERE logical_key = ? "
            "ORDER BY version DESC, filed_date DESC NULLS LAST LIMIT 1",
            [logical_key],
        ).fetchone()
        return str(row[0]) if row else None

    def gc_superseded(self, logical_key: str) -> int:
        """Delete every filing of a logical document except its current one.

        Removes superseded filings and their facts (facts first, for the FK).
        Returns the number of filings removed. Correctness does not depend on
        this — read-time precedence already prefers the latest; this is cleanup.
        """
        keep = self.latest_filing_id(logical_key)
        if keep is None:
            return 0
        superseded = [
            str(row[0])
            for row in self._conn.execute(
                "SELECT filing_id FROM filings WHERE logical_key = ? AND filing_id != ?",
                [logical_key, keep],
            ).fetchall()
        ]
        if not superseded:
            return 0
        # Facts first (the FK), then their filings. Done as separate statements,
        # not one transaction: DuckDB's FK check does not see same-transaction
        # deletes, and GC is idempotent cleanup that need not be atomic.
        placeholders = ", ".join("?" * len(superseded))
        self._conn.execute(
            f"DELETE FROM financial_facts WHERE filing_id IN ({placeholders})", superseded
        )
        self._conn.execute(f"DELETE FROM filings WHERE filing_id IN ({placeholders})", superseded)
        return len(superseded)

    def close(self) -> None:
        """Close the connection, releasing the write lock."""
        self._conn.close()
