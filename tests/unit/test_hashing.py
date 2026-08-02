"""Stable hashing tests."""

from rag_claim_verification.utils.hashing import hash_mapping, sha256_text


def test_config_mapping_hash_is_key_order_independent() -> None:
    first = {"temperature": 0.0, "model": "test", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "model": "test", "temperature": 0.0}

    assert hash_mapping(first) == hash_mapping(second)


def test_hash_changes_with_configuration_value() -> None:
    assert hash_mapping({"top_k": 3}) != hash_mapping({"top_k": 5})
    assert sha256_text("a") != sha256_text("b")
