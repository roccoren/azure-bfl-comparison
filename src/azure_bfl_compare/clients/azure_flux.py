from __future__ import annotations

import base64
import time
from typing import Any, Dict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import AzureFluxConfig
from ..types import GenerationResult


class AzureFluxClient:
    """Client for invoking Azure-hosted Flux image modification (edits) endpoint."""

    def __init__(self, config: AzureFluxConfig) -> None:
        self._config = config
        url = httpx.URL(config.endpoint)

        base_url = url.copy_with(path="/", query=None, fragment=None)
        self._session = httpx.Client(
            base_url=str(base_url),
            headers={
                "api-key": config.api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0),
        )

        params_dict = dict(url.params)
        self._api_version = params_dict.get("api-version", self._config.api_version)

        self._deployment_path = f"/openai/deployments/{self._config.deployment}"
        self._modification_path: str | None = None

        path = (url.path or "").rstrip("/")
        if path and not path.startswith("/"):
            path = f"/{path}"

        if path:
            if "/openai/deployments/" in path:
                deployment_segment = path.split("/openai/deployments/")[1].split("/")[0]
                if deployment_segment:
                    self._deployment_path = f"/openai/deployments/{deployment_segment}"
            if "/images" in path:
                if "/images/edits" in path or "/images/modifications" in path:
                    self._modification_path = path
                elif "/images/generations" in path:
                    self._modification_path = path.replace("/images/generations", "/images/edits")

        if self._modification_path is None:
            self._modification_path = f"{self._deployment_path}/images/edits"

    def close(self) -> None:
        self._session.close()

    def _poll_operation(self, operation_url: str) -> Dict[str, Any]:
        """Poll Azure operation endpoint until completion."""
        url = httpx.URL(operation_url, base=self._session.base_url)
        params: Dict[str, Any] = {}
        if "api-version" not in url.params:
            params["api-version"] = self._api_version

        for _ in range(self._config.max_poll_attempts):
            response = self._session.get(url, params=params or None)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            status = (data.get("status") or "").lower()

            if status in {"succeeded", "success"}:
                return data
            if status in {"failed", "cancelled", "canceled"}:
                raise RuntimeError(f"Azure modification failed with status '{status}': {data}")

            time.sleep(self._config.poll_interval_seconds)

        raise TimeoutError(
            f"Azure operation did not complete after {self._config.max_poll_attempts} polls."
        )

    def _extract_image(self, task_name: str, data: Dict[str, Any], payload: Dict[str, Any]) -> GenerationResult:
        image_data = None
        image_base64 = None

        if "data" in data and data["data"]:
            image_data = data["data"][0]
            image_base64 = image_data.get("b64_json")
        elif "result" in data and data["result"].get("data"):
            result_data = data["result"]["data"]
            if result_data:
                image_data = result_data[0]
                image_base64 = image_data.get("b64_json")
        elif "images" in data and data["images"]:
            image_data = data["images"][0]
            image_base64 = image_data.get("b64_json") or image_data.get("base64")
        elif "b64_json" in data:
            image_base64 = data["b64_json"]
            image_data = data

        if not image_base64:
            raise RuntimeError(f"Azure response missing image data. Response format: {list(data.keys())}")

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode base64 image data: {exc}") from exc

        metadata = {
            "provider": "azure",
            "status": data.get("status"),
            "usage": data.get("usage"),
            "response_format": list(data.keys()),
        }
        metadata.update(payload)

        return GenerationResult(
            provider="azure",
            task_name=task_name,
            image_bytes=image_bytes,
            metadata=metadata,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
        reraise=True,
    )
    def generate(self, task_name: str, payload: Dict[str, Any]) -> GenerationResult:
        """
        Submit an image modification job and return the resulting image payload.

        Expected payload keys:
        - image: base64-encoded source image (required)
        - mask: base64-encoded mask (optional)
        - prompt / other edit parameters
        """
        request_body = payload.copy()
        request_body.setdefault("model", self._config.deployment)

        image_b64 = request_body.get("image")
        if not image_b64:
            raise RuntimeError("Azure modification payload missing required 'image' base64 data")
        if not request_body.get("prompt"):
            raise RuntimeError("Azure modification payload missing required 'prompt' value")

        if not request_body.get("mask"):
            request_body.pop("mask", None)

        response = self._session.post(
            self._modification_path,
            params={"api-version": self._api_version},
            json=request_body,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Azure Flux modification endpoint not found at {self._modification_path}. "
                    "Ensure AZURE_FLUX_ENDPOINT is the full edits URL from the Azure portal "
                    "(.../openai/deployments/<name>/images/edits?api-version=...)."
                ) from exc
            raise

        response_data = response.json()
        operation_url = (
            response.headers.get("Operation-Location")
            or response.headers.get("operation-location")
            or response.headers.get("Azure-AsyncOperation")
        )

        if operation_url:
            if not operation_url.startswith("http"):
                operation_url = str(self._session.base_url.join(operation_url))
            response_data = self._poll_operation(operation_url)

        return self._extract_image(task_name, response_data, payload)

    def __enter__(self) -> "AzureFluxClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401 - standard context manager
        self.close()