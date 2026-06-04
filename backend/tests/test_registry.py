"""Tests for the adapter registry."""

from __future__ import annotations

import pytest

from raglearn.core.bootstrap import load_adapters
from raglearn.core.errors import AdapterNotImplementedError, RegistryError
from raglearn.core.registry import Registry
from raglearn.core.registry import registry as global_registry


def test_register_and_create() -> None:
    reg = Registry()

    @reg.register("widget", "basic")
    class BasicWidget:
        def __init__(self, size: int = 1) -> None:
            self.size = size

    assert reg.has("widget", "basic")
    instance = reg.create("widget", "basic", size=3)
    assert isinstance(instance, BasicWidget)
    assert instance.size == 3


def test_duplicate_registration_raises() -> None:
    reg = Registry()

    @reg.register("widget", "basic")
    class _First:
        pass

    with pytest.raises(RegistryError):

        @reg.register("widget", "basic")
        class _Second:
            pass


def test_unimplemented_adapter_raises() -> None:
    reg = Registry()
    with pytest.raises(AdapterNotImplementedError) as exc_info:
        reg.create("widget", "missing")
    assert exc_info.value.stage == "widget"
    assert exc_info.value.name == "missing"


def test_names_for_stage() -> None:
    reg = Registry()
    reg.register("widget", "b")(object)
    reg.register("widget", "a")(object)
    reg.register("other", "x")(object)
    assert reg.names("widget") == ["a", "b"]


def test_load_adapters_registers_local_cost_model() -> None:
    load_adapters()
    assert global_registry.has("cost_model", "local")
    model = global_registry.create("cost_model", "local")
    assert hasattr(model, "price")
