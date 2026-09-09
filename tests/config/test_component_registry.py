import pytest

from src.config.component_registry import (
    CHANNEL_REGISTRY,
    MESH_REGISTRY,
    SCALAR_FIELD_REGISTRY,
    SPREAD_ENGINE_REGISTRY,
    VECTOR_FIELD_REGISTRY,
    resolve_registered_name,
)
from src.spark.acoustic.channel_protocol import ChannelProtocol
from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.terrain.mesh_protocol import MeshProtocol

ALL_REGISTRIES = (
    MESH_REGISTRY,
    SPREAD_ENGINE_REGISTRY,
    CHANNEL_REGISTRY,
    SCALAR_FIELD_REGISTRY,
    VECTOR_FIELD_REGISTRY,
)

REGISTRY_PROTOCOLS = (
    (MESH_REGISTRY, MeshProtocol),
    (SPREAD_ENGINE_REGISTRY, SpreadEngineProtocol),
    (CHANNEL_REGISTRY, ChannelProtocol),
    (SCALAR_FIELD_REGISTRY, ScalarFieldProtocol),
    (VECTOR_FIELD_REGISTRY, VectorFieldProtocol),
)


@pytest.mark.parametrize("registry", ALL_REGISTRIES)
def test_no_registry_is_empty(registry: dict[str, type]) -> None:
    assert len(registry) > 0


@pytest.mark.parametrize("registry", ALL_REGISTRIES)
def test_every_registered_name_maps_to_a_class(registry: dict[str, type]) -> None:
    assert all(isinstance(entry, type) for entry in registry.values())


@pytest.mark.parametrize(("registry", "protocol"), REGISTRY_PROTOCOLS)
def test_every_registered_class_declares_the_protocol_methods(
    registry: dict[str, type], protocol: type
) -> None:
    required = [
        name
        for name in vars(protocol)
        if not name.startswith("_") and callable(getattr(protocol, name))
    ]
    for registered_class in registry.values():
        for method_name in required:
            assert hasattr(registered_class, method_name), (
                f"{registered_class.__name__} is missing {method_name}"
            )


def test_resolving_an_unregistered_name_lists_what_is_registered() -> None:
    with pytest.raises(ValueError, match="unknown spread engine 'quantum'; registered"):
        resolve_registered_name(SPREAD_ENGINE_REGISTRY, "quantum", "spread engine")


def test_resolving_a_registered_name_returns_the_class() -> None:
    for name, registered_class in SPREAD_ENGINE_REGISTRY.items():
        assert (
            resolve_registered_name(SPREAD_ENGINE_REGISTRY, name, "spread engine")
            is registered_class
        )
