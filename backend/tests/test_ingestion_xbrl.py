"""Tests for the XBRL facts arm: classification, bundle assembly, extraction.

Fully hermetic — no real filings, no network. A tiny self-contained mock XBRL
filing (a custom one-concept taxonomy that resolves against the core XBRL schema
Arelle ships with) drives the extraction and completeness-gate tests, paired with
an in-memory object-store stand-in. This exercises the real Arelle parse and the
fact -> FinancialFact mapping without committing megabytes of SEC data.
"""

from __future__ import annotations

import pytest

from raglearn.core.errors import IngestionError
from raglearn.core.types import FactOrigin, RawDocument
from raglearn.ingestion.bundle import BundleAssembler
from raglearn.ingestion.xbrl_extract import (
    ArelleXbrlExtractor,
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
    facts = list(ArelleXbrlExtractor(BundleAssembler(_mock_store())).extract(_doc("mock.xml")))  # type: ignore[arg-type]

    assert [(f.concept, f.value) for f in facts] == [("mock:Revenue", 100.0)]
    assert all(f.filing_id == "0000-00-000" for f in facts)


def test_extractor_defers_until_schema_is_present() -> None:
    store = _mock_store(with_schema=False)
    deferred = list(ArelleXbrlExtractor(BundleAssembler(store)).extract(_doc("mock.xml")))  # type: ignore[arg-type]
    assert deferred == []


def test_schema_event_completes_the_bundle_via_located_instance() -> None:
    # The .xsd event (not the instance) must locate the instance and extract.
    facts = list(ArelleXbrlExtractor(BundleAssembler(_mock_store())).extract(_doc("mock.xsd")))  # type: ignore[arg-type]

    assert any(f.concept == "mock:Revenue" and f.value == 100.0 for f in facts)
