from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping


@dataclass(slots=True)
class GenerationResult:
    """Represents a single provider output for a batch task."""

    provider: str
    task_name: str
    image_bytes: bytes
    metadata: Mapping[str, Any]

    def mutable_metadata(self) -> MutableMapping[str, Any]:
        """Return a mutable copy of the metadata payload."""
        return dict(self.metadata)