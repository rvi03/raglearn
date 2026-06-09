"""Tests for the synthetic eval corpus and its golden set.

These guard the properties the corpus must have to make retrieval metrics
meaningful: it is deterministic, the golden cases reference real chunks, the
expected figures are actually present in those chunks, and there are distractors
(more than one company with a revenue/risk chunk) so retrieval has to choose.
"""

from __future__ import annotations

from finrag.eval.synthetic import synthetic_corpus, synthetic_facts


def test_corpus_is_deterministic() -> None:
    chunks_a, cases_a = synthetic_corpus()
    chunks_b, cases_b = synthetic_corpus()

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.id for c in cases_a] == [c.id for c in cases_b]


def test_golden_references_real_targets() -> None:
    # Narrative cases point at chunk ids; exact cases point at fact ids.
    chunks, cases = synthetic_corpus()
    chunk_ids = {c.chunk_id for c in chunks}
    fact_ids = {f.fact_id for ex in synthetic_facts() for f in ex.facts}

    for case in cases:
        targets = fact_ids if case.query_type == "exact" else chunk_ids
        for expected in case.expected_chunk_ids:
            assert expected in targets, f"{case.id} references missing target {expected}"


def test_narrative_expected_values_appear_in_their_chunk() -> None:
    chunks, cases = synthetic_corpus()
    by_id = {c.chunk_id: c for c in chunks}

    for case in cases:
        if case.query_type == "exact":
            continue  # exact figures live in facts, checked via the live run
        for value in case.expected_values:
            texts = " ".join(by_id[cid].text for cid in case.expected_chunk_ids)
            assert value in texts, f"{case.id}: expected value {value!r} not in its chunk"


def test_has_exact_cases_pointing_at_facts() -> None:
    _chunks, cases = synthetic_corpus()
    fact_ids = {f.fact_id for ex in synthetic_facts() for f in ex.facts}
    exact = [c for c in cases if c.query_type == "exact"]

    assert len(exact) >= 2
    for case in exact:
        assert case.expected_chunk_ids and case.expected_chunk_ids[0] in fact_ids


def test_synthetic_facts_carry_expected_values() -> None:
    # Each exact case's figure should be the formatted value of its fact.
    _chunks, cases = synthetic_corpus()
    facts = {f.fact_id: f for ex in synthetic_facts() for f in ex.facts}

    for case in cases:
        if case.query_type != "exact":
            continue
        fact = facts[case.expected_chunk_ids[0]]
        formatted = f"{int(fact.value):,}"
        for value in case.expected_values:
            assert value in formatted, f"{case.id}: {value!r} not in {formatted!r}"


def test_corpus_has_distractors() -> None:
    chunks, _cases = synthetic_corpus()
    # Both companies have a revenue chunk and a risk chunk -> topic alone is not
    # enough to retrieve the right one.
    revenue = [c for c in chunks if "revenue" in c.text.lower()]
    companies = {c.metadata.company_name for c in chunks}

    assert len(companies) >= 2
    assert len({c.metadata.company_name for c in revenue}) >= 2


def test_has_a_negative_unanswerable_case() -> None:
    _chunks, cases = synthetic_corpus()
    negatives = [c for c in cases if not c.answerable]

    assert len(negatives) >= 1  # the suite can score an abstention
    assert all(c.expected_chunk_ids == [] for c in negatives)


def test_has_near_duplicate_distractor() -> None:
    chunks, _cases = synthetic_corpus()
    ids = {c.chunk_id for c in chunks}

    # same company + metric, different year -> retrieval must disambiguate the year
    assert {"nw-revenue", "nw-revenue-fy23"} <= ids


def test_no_plumbing_in_synthetic_text() -> None:
    chunks, _cases = synthetic_corpus()
    from finrag.eval import metrics

    assert metrics.plumbing_leakage([c.text for c in chunks]) == 0.0
