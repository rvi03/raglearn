"""Chunking adapters and the primitives they share.

The adaptive chunker routes a parsed document between a structure-aware path and
a semantic path, and falls back to a token-aware recursive split when either
produces a chunk that overruns the embedder's token budget. The recursive
splitter is that shared fallback and lives in :mod:`recursive`; the strategy
adapters and the router that selects between them are added alongside it.
"""
