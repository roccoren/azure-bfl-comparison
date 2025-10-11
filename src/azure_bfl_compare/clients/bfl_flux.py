from __future__ import annotations

import base64
from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import BFLFluxConfig
from ..types import GenerationResult


class BFLFluxClient:
    """Client for accessing the official BFL Flux image modification endpoint."""

    def __init__(self, config: BFLFluxConfig) -> None:
        self._config = config
        url = httpx.URL(config.api_url)

        base_url = url.copy_with(path="/", query=None, fragment=None)
        self._session = httpx.Client(
            base_url=str(base_url),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0),
        )

        path = (url.path or "").rstrip("/")
        if not path or path == "":
            self._modification_path = "/v1/images/edits"
        elif "/images/generations" in path:
            self._modification_path = path.replace("images/generations", "images/edits")
        else:
            self._modification_path = path

    def close(self) -> None:
        self._session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
        reraise=True,
    )
    def generate(self, task_name: str, payload: Dict[str, Any]) -> GenerationResult:
        """
        Issue an image modification request to the BFL API.

        Expected payload structure:
        {
            "input": {
                "prompt": "...",
                "image": "<base64>",
                "mask": "<base64 optional>",
                ...
            },
            ... provider overrides ...
        }
        """
        response = self._session.post(self._modification_path, json=payload)
        response.raise_for_status()

        data = response.json()
        image_entries = (
            data.get("data")
            or data.get("result", {}).get("data")
            or data.get("images")
            or []
        )
        if not image_entries:
            raise RuntimeError(f"BFL response missing image data: {list(data.keys())}")

        first_entry = image_entries[0]
        image_base64 = first_entry.get("b64_json") or first_entry.get("base64")
        if not image_base64:
            raise RuntimeError("BFL response did not include base64 image data")

        image_bytes = base64.b64decode(image_base64)

        metadata = {
            "provider": "bfl",
            "id": data.get("id"),
            "created": data.get("created"),
            "usage": data.get("usage"),
        }
        metadata.update(payload)

        return GenerationResult(
            provider="bfl",
            task_name=task_name,
            image_bytes=image_bytes,
            metadata=metadata,
        )

    def __enter__(self) -> "BFLFluxClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()