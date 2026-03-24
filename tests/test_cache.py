"""Tests for JSONCache.

Uses tmp_path — no real data/cache directory touched.
"""

from __future__ import annotations

import json

import pytest

from src.cache import JSONCache


class TestJSONCache:
    def test_roundtrip_dict(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        key = cache.make_key("gpt-4o-mini", "system", "user")
        cache.set(key, {"content": "hello"})
        assert cache.get(key) == {"content": "hello"}

    def test_roundtrip_string(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        key = cache.make_key("model", "sys", "usr")
        cache.set(key, "plain string")
        assert cache.get(key) == "plain string"

    def test_miss_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        assert cache.get("nonexistent_key") is None

    def test_deterministic_key(self) -> None:
        cache = JSONCache("data/cache")
        k1 = cache.make_key("gpt-4o-mini", "You are helpful.", "What is RAG?")
        k2 = cache.make_key("gpt-4o-mini", "You are helpful.", "What is RAG?")
        assert k1 == k2

    def test_different_inputs_different_keys(self) -> None:
        cache = JSONCache("data/cache")
        k1 = cache.make_key("gpt-4o-mini", "sys", "query A")
        k2 = cache.make_key("gpt-4o-mini", "sys", "query B")
        assert k1 != k2

    def test_model_difference_changes_key(self) -> None:
        cache = JSONCache("data/cache")
        k1 = cache.make_key("gpt-4o-mini", "sys", "q")
        k2 = cache.make_key("gpt-4o", "sys", "q")
        assert k1 != k2

    def test_system_prompt_difference_changes_key(self) -> None:
        cache = JSONCache("data/cache")
        k1 = cache.make_key("model", "prompt A", "q")
        k2 = cache.make_key("model", "prompt B", "q")
        assert k1 != k2

    def test_creates_missing_directory(self, tmp_path: pytest.TempPathFactory) -> None:
        nested = str(tmp_path / "a" / "b" / "c")
        cache = JSONCache(nested)
        key = cache.make_key("m", "s", "u")
        cache.set(key, {"x": 1})
        assert cache.get(key) == {"x": 1}

    def test_overwrite_existing_key(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        key = cache.make_key("m", "s", "u")
        cache.set(key, {"v": 1})
        cache.set(key, {"v": 2})
        assert cache.get(key) == {"v": 2}

    def test_key_is_hex_string(self) -> None:
        cache = JSONCache("data/cache")
        key = cache.make_key("model", "system", "user")
        assert all(c in "0123456789abcdef" for c in key)
        assert len(key) == 32  # MD5 hex digest length

    def test_file_written_to_cache_dir(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        key = cache.make_key("m", "s", "u")
        cache.set(key, {"data": "value"})
        expected_file = tmp_path / f"{key}.json"
        assert expected_file.exists()
        assert json.loads(expected_file.read_text()) == {"data": "value"}
