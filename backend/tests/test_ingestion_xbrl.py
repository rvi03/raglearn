"""Tests for the XBRL facts arm: classification, bundle assembly, extraction.

Fully hermetic — no real filings, no network. A tiny self-contained mock XBRL
filing (a custom one-concept taxonomy that resolves against the core XBRL schema
Arelle ships with) drives the extraction and completeness-gate tests, paired with
an in-memory object-store stand-in. This exercises the real Arelle parse and the
fact -> FinancialFact mapping without committing megabytes of SEC data.
"""

from __future__ import annotations

import pytest

from finrag.core.errors import IngestionError
from finrag.core.types import FactOrigin, Market, RawDocument
from finrag.ingestion.bundle import BundleAssembler
from finrag.ingestion.xbrl_extract import (
    ArelleXbrlExtractor,
    _build_metadata,
    _dimensions,
    _to_int,
    extract_facts,
    is_inline_xbrl,
    is_xbrl_instance,
    is_xbrl_schema,
)

# --- a minimal, self-contained mock XBRL filing -------------------------------
# One concept (mock:Revenue, a monetary item) defined in a custom namespace; its
# type comes from the core XBRL instance schema, which Arelle resolves locally —
# so this needs no us-gaap taxonomy and no network.

_MOCK_XSD = b"""<?xml version="1.0" encoding="UTF-8"?>
<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
            xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:mock="http://example.com/mock"
            targetNamespace="http://example.com/mock" elementFormDefault="qualified">
  <xsd:import namespace="http://www.xbrl.org/2003/instance"
              schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>
  <xsd:element name="Revenue" id="mock_Revenue" type="xbrli:monetaryItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
</xsd:schema>
"""

_FACT = b'<mock:Revenue contextRef="c1" unitRef="usd" decimals="0">100</mock:Revenue>'

_INSTANCE_HEAD = b"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:link="http://www.xbrl.org/2003/linkbase"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns:mock="http://example.com/mock"
            xmlns:iso4217="http://www.xbrl.org/2003/iso4217">
  <link:schemaRef xlink:type="simple" xlink:href="mock.xsd"/>
  <xbrli:context id="c1">
    <xbrli:entity><xbrli:identifier scheme="http://example.com">MOCK</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
"""
_INSTANCE_TAIL = b"</xbrli:xbrl>"
_MOCK_INSTANCE = _INSTANCE_HEAD + _FACT + _INSTANCE_TAIL
# Same fact tagged twice — exercises dedup (same concept/period/unit/dimension).
_MOCK_INSTANCE_DUP = _INSTANCE_HEAD + _FACT + _FACT + _INSTANCE_TAIL

# --- a mock DEI taxonomy + instance (closes the hermetic DEI gap) --------------
# DEI concepts in a custom namespace whose URI carries the "/dei/" marker the
# extractor keys on; string items resolve against the core XBRL schema, so real
# Arelle reads them with no SEC dei taxonomy and no network. The instance's entry
# schema IS this dei schema (no us-gaap needed); its stem matches so the bundle
# completeness gate (`{stem}.xsd`) is satisfied.
_DEI_XSD = b"""<?xml version="1.0" encoding="UTF-8"?>
<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
            xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:dei="http://example.com/dei/2024"
            targetNamespace="http://example.com/dei/2024" elementFormDefault="qualified">
  <xsd:import namespace="http://www.xbrl.org/2003/instance"
              schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>
  <xsd:element name="EntityRegistrantName" type="xbrli:stringItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
  <xsd:element name="EntityCentralIndexKey" type="xbrli:stringItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
  <xsd:element name="TradingSymbol" type="xbrli:stringItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
  <xsd:element name="DocumentType" type="xbrli:stringItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
  <xsd:element name="DocumentFiscalYearFocus" type="xbrli:stringItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
  <xsd:element name="DocumentFiscalPeriodFocus" type="xbrli:stringItemType"
               substitutionGroup="xbrli:item" xbrli:periodType="duration" nillable="true"/>
</xsd:schema>
"""

_DEI_INSTANCE = b"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:link="http://www.xbrl.org/2003/linkbase"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns:dei="http://example.com/dei/2024">
  <link:schemaRef xlink:type="simple" xlink:href="dei.xsd"/>
  <xbrli:context id="c1">
    <xbrli:entity><xbrli:identifier scheme="http://example.com">MOCK</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <dei:EntityRegistrantName contextRef="c1">Mock Corp</dei:EntityRegistrantName>
  <dei:EntityCentralIndexKey contextRef="c1">1234567</dei:EntityCentralIndexKey>
  <dei:TradingSymbol contextRef="c1">MOCK</dei:TradingSymbol>
  <dei:DocumentType contextRef="c1">10-K</dei:DocumentType>
  <dei:DocumentFiscalYearFocus contextRef="c1">2024</dei:DocumentFiscalYearFocus>
  <dei:DocumentFiscalPeriodFocus contextRef="c1">FY</dei:DocumentFiscalPeriodFocus>
</xbrli:xbrl>
"""

_PREFIX = "us/mock/0000-00-000-xbrl/"


# --- classification (hermetic byte markers) -----------------------------------

_IXBRL = b'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
_INSTANCE_XML = b'<xbrl xmlns="http://www.xbrl.org/2003/instance">'
_LINKBASE_XML = b'<linkbase xmlns="http://www.xbrl.org/2003/linkbase">'
_SCHEMA = (
    b'<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
    b'xmlns:xbrli="http://www.xbrl.org/2003/instance">'
)


def test_is_inline_xbrl_detects_the_ix_namespace() -> None:
    assert is_inline_xbrl(_IXBRL) is True
    assert is_inline_xbrl(b"<html><body>plain</body></html>") is False


def test_is_xbrl_instance_distinguishes_instance_from_bundle_members() -> None:
    assert is_xbrl_instance(_INSTANCE_XML) is True
    assert is_xbrl_instance(_LINKBASE_XML) is False
    assert is_xbrl_instance(_SCHEMA) is False


def test_is_xbrl_schema_detects_the_schema_not_the_linkbase() -> None:
    assert is_xbrl_schema(_SCHEMA) is True
    assert is_xbrl_schema(_LINKBASE_XML) is False
    assert is_xbrl_schema(_INSTANCE_XML) is False


# --- in-memory object store + bundle assembly ---------------------------------


class _DictStore:
    """ObjectStore stand-in over an in-memory key->bytes map."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def list_prefix(self, bucket: str, prefix: str) -> list[str]:
        return [k for k in self._objects if k.startswith(prefix)]

    def fetch_sync(self, bucket: str, key: str) -> bytes:
        return self._objects[key]


def _doc(name: str) -> RawDocument:
    return RawDocument(
        doc_id=_PREFIX + name, filename=name, content_type="application/xml", source_bucket="b"
    )


def _mock_store(*, with_schema: bool = True) -> _DictStore:
    objects = {_PREFIX + "mock.xml": _MOCK_INSTANCE}
    if with_schema:
        objects[_PREFIX + "mock.xsd"] = _MOCK_XSD
    return _DictStore(objects)


def test_bundle_assembler_fetches_only_bundle_files_and_cleans_up() -> None:
    store = _DictStore(
        {
            _PREFIX + "mock.xml": b"instance",
            _PREFIX + "mock.xsd": b"schema",
            _PREFIX + "mock_lab.xml": b"labels",
            _PREFIX + "exhibit99.htm": b"exhibit",  # different stem -> excluded
            _PREFIX + "mock_g1.jpg": b"image",  # not a bundle extension -> excluded
        }
    )

    with BundleAssembler(store).materialize(_doc("mock.xml")) as instance_path:  # type: ignore[arg-type]
        names = sorted(p.name for p in instance_path.parent.iterdir())
        tmp = instance_path.parent
        assert instance_path.name == "mock.xml"
        assert names == ["mock.xml", "mock.xsd", "mock_lab.xml"]

    assert not tmp.exists()  # temp dir removed on exit


def test_bundle_assembler_requires_a_source_bucket() -> None:
    doc = RawDocument(doc_id="us/x/f.htm", filename="f.htm", content_type="text/html")
    with pytest.raises(IngestionError, match="source bucket"):
        with BundleAssembler(_DictStore({})).materialize(doc):  # type: ignore[arg-type]
            pass


def test_bundle_incomplete_without_schema_defers() -> None:
    store = _DictStore({_PREFIX + "mock.xml": b"i", _PREFIX + "mock_lab.xml": b"l"})
    with BundleAssembler(store).materialize(_doc("mock.xml")) as path:  # type: ignore[arg-type]
        assert path is None


def test_bundle_incomplete_without_instance_defers() -> None:
    store = _DictStore({_PREFIX + "mock.xsd": b"s", _PREFIX + "mock_lab.xml": b"l"})
    with BundleAssembler(store).materialize(_doc("mock.xsd")) as path:  # type: ignore[arg-type]
        assert path is None


# --- extraction on the mock filing (real Arelle, no real data) ----------------


def test_extract_facts_maps_a_fact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "mock.xsd").write_bytes(_MOCK_XSD)
    (tmp_path / "mock.xml").write_bytes(_MOCK_INSTANCE)

    facts = extract_facts(tmp_path / "mock.xml", "mock-filing")

    assert len(facts) == 1
    fact = facts[0]
    assert fact.concept == "mock:Revenue"
    assert fact.value == 100.0
    assert fact.unit == "USD"
    assert fact.period == "2024-01-01/2024-12-31"  # end is the as-filed date, not Arelle's +1
    assert fact.dimension is None
    assert fact.origin is FactOrigin.XBRL


def test_extract_facts_deduplicates_repeated_facts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "mock.xsd").write_bytes(_MOCK_XSD)
    (tmp_path / "mock.xml").write_bytes(_MOCK_INSTANCE_DUP)  # fact tagged twice

    facts = extract_facts(tmp_path / "mock.xml", "f")

    assert len(facts) == 1  # collapsed to one


def test_extractor_extracts_from_a_complete_bundle() -> None:
    extraction = ArelleXbrlExtractor(BundleAssembler(_mock_store())).extract(_doc("mock.xml"))  # type: ignore[arg-type]

    assert extraction is not None
    assert [(f.concept, f.value) for f in extraction.facts] == [("mock:Revenue", 100.0)]
    assert all(f.filing_id == "0000-00-000" for f in extraction.facts)
    # No DEI in the mock → the per-filing collection fallback.
    assert extraction.filing.filing_id == "0000-00-000"
    assert extraction.collection.collection_id == "us-0000-00-000"


def test_extractor_defers_until_schema_is_present() -> None:
    store = _mock_store(with_schema=False)
    extraction = ArelleXbrlExtractor(BundleAssembler(store)).extract(_doc("mock.xml"))  # type: ignore[arg-type]
    assert extraction is None


def test_schema_event_completes_the_bundle_via_located_instance() -> None:
    # The .xsd event (not the instance) must locate the instance and extract.
    extraction = ArelleXbrlExtractor(BundleAssembler(_mock_store())).extract(_doc("mock.xsd"))  # type: ignore[arg-type]

    assert extraction is not None
    assert any(f.concept == "mock:Revenue" and f.value == 100.0 for f in extraction.facts)


# --- DEI -> metadata mapping --------------------------------------------------
# The Arelle DEI *iteration* is exercised by a live smoke on a real filing (no DEI
# taxonomy ships in the hermetic mock); the pure DEI -> rows mapping is unit-tested
# here with fabricated, non-real DEI values.


def test_build_metadata_maps_dei_to_collection_and_filing() -> None:
    dei = {
        "EntityRegistrantName": "Mock Corp",
        "EntityCentralIndexKey": "1234567",
        "TradingSymbol": "MOCK",
        "DocumentType": "10-K",
        "DocumentFiscalYearFocus": "2024",
        "DocumentFiscalPeriodFocus": "FY",
    }

    collection, filing = _build_metadata(dei, "acc-1")

    assert collection.collection_id == "us-cik-1234567"
    assert (collection.company, collection.ticker, collection.cik) == (
        "Mock Corp",
        "MOCK",
        "1234567",
    )
    assert collection.market is Market.US
    assert filing.collection_id == "us-cik-1234567"
    assert (filing.filing_type, filing.fiscal_year, filing.fiscal_period) == ("10-K", 2024, "FY")


def test_build_metadata_without_cik_falls_back_to_a_per_filing_collection() -> None:
    collection, filing = _build_metadata({}, "acc-1")

    assert collection.collection_id == "us-acc-1"
    assert filing.collection_id == "us-acc-1"
    assert collection.company is None
    assert filing.fiscal_year is None


def test_to_int_parses_a_year_or_returns_none() -> None:
    assert _to_int("2024") == 2024
    assert _to_int(None) is None
    assert _to_int("not-a-year") is None


# --- DEI iteration through REAL Arelle (hermetic, via the mock dei taxonomy) ---


def _dei_store() -> _DictStore:
    return _DictStore({_PREFIX + "dei.xml": _DEI_INSTANCE, _PREFIX + "dei.xsd": _DEI_XSD})


def test_extractor_reads_real_dei_into_metadata() -> None:
    extraction = ArelleXbrlExtractor(BundleAssembler(_dei_store())).extract(_doc("dei.xml"))  # type: ignore[arg-type]

    assert extraction is not None
    assert extraction.collection.collection_id == "us-cik-1234567"
    assert extraction.collection.company == "Mock Corp"
    assert extraction.collection.cik == "1234567"
    assert extraction.collection.ticker == "MOCK"
    assert extraction.filing.filing_type == "10-K"
    assert extraction.filing.fiscal_year == 2024
    assert extraction.filing.fiscal_period == "FY"


def test_us_identity_from_real_dei_extraction() -> None:
    from finrag.core.types import NumericAuthority
    from finrag.ingestion.identity.us import identity_from_extraction

    doc = _doc("dei.xml")
    extraction = ArelleXbrlExtractor(BundleAssembler(_dei_store())).extract(doc)  # type: ignore[arg-type]
    assert extraction is not None

    ident = identity_from_extraction(extraction, doc)
    assert ident.market is Market.US
    assert ident.doc_type == "10-K"
    assert ident.cik == "1234567"
    assert ident.numeric_authority is NumericAuthority.AUTHORITATIVE
    assert "us-cik-1234567" in ident.logical_key


# --- dimension rendering (the fact_id key must be total over both kinds) -------
# Arelle's dimension-value objects are duck-typed here: explicit members expose a
# QName, typed members expose an element carrying a stringValue. These stand in
# for them so the rendering is tested without building a dimensional filing.


class _Axis:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name


class _ExplicitMember:
    isExplicit = True  # noqa: N815 (mirrors Arelle's attribute name)

    def __init__(self, member: str) -> None:
        self.memberQname = _Axis(member)


class _TypedMember:
    isExplicit = False  # noqa: N815 (mirrors Arelle's attribute name)

    class _Element:
        def __init__(self, value: str) -> None:
            self.stringValue = value

    def __init__(self, value: str) -> None:
        self.typedMember = _TypedMember._Element(value)


class _Context:
    def __init__(self, dims: dict[object, object]) -> None:
        self.qnameDims = dims


def test_dimensions_is_none_when_undimensioned() -> None:
    assert _dimensions(_Context({})) is None


def test_dimensions_renders_an_explicit_member() -> None:
    ctx = _Context({_Axis("seg:Axis"): _ExplicitMember("seg:iPhone")})
    assert _dimensions(ctx) == "seg:Axis=seg:iPhone"


def test_dimensions_includes_a_typed_member() -> None:
    # The gap this closes: a typed member must not be dropped, or two facts that
    # differ only by it would collapse onto one fact_id.
    ctx = _Context({_Axis("seg:PropertyAxis"): _TypedMember("123 Main St")})
    assert _dimensions(ctx) == "seg:PropertyAxis=123 Main St"


def test_dimensions_are_sorted_by_axis_for_a_stable_key() -> None:
    dims = {_Axis("b:Axis"): _ExplicitMember("b:M"), _Axis("a:Axis"): _TypedMember("v")}
    assert _dimensions(_Context(dims)) == "a:Axis=v;b:Axis=b:M"
