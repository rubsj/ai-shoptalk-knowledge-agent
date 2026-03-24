"""JSON file cache for LLM responses and embeddings.

Cache key: MD5 hash of (model + system_prompt + user_prompt).
Cache location: data/cache/ as JSON files.

35+ configs x 15 queries = 500+ LLM calls per full run.
Cache avoids re-spend. Same pattern from P1/P2/P4.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JSONCache:
    """File-backed JSON cache keyed by MD5 hash of (model, system_prompt, user_prompt)."""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Return MD5 hex digest of the canonical cache key string."""
        raw = f"{model}\n{system_prompt}\n---\n{user_prompt}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Return cached value or None if not found."""
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        """Write value to cache as JSON."""
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")