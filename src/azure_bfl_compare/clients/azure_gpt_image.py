from __future__ import annotations

import base64
import binascii
import io
import time
from typing import Any, Dict

import httpx
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import AzureGPTImageConfig
from ..types import GenerationResult


class AzureGPTImageClient:
    """Client for invoking Azure GPT-Image-1 image edits endpoint."""

    def __init__(self, config: AzureGPTImageConfig) -> None:
        self._config = config
        url = httpx.URL(config.endpoint)

        base_url = url.copy_with(path="/", query=None, fragment=None)
        self._session = httpx.Client(
            base_url=str(base_url),
            headers={
                "api-key": config.api_key,
                "Accept": "application/json",
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
                raise RuntimeError(f"Azure cloth swap failed with status '{status}': {data}")

            time.sleep(self._config.poll_interval_seconds)

        raise TimeoutError(
            f"Azure operation did not complete after {self._config.max_poll_attempts} polls."
        )

    def _extract_image(self, task_name: str, data: Dict[str, Any], payload: Dict[str, Any]) -> GenerationResult:
        """Extract image from GPT-Image-1 response."""
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
        elif "image" in data:
            # GPT-Image-1 might return image directly
            image_base64 = data["image"]
            image_data = data

        if not image_base64:
            raise RuntimeError(f"Azure GPT-Image-1 response missing image data. Response format: {list(data.keys())}")

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode base64 image data: {exc}") from exc

        metadata = {
            "provider": "azure_gpt_image_1",
            "status": data.get("status"),
            "usage": data.get("usage"),
            "response_format": list(data.keys()),
        }
        sanitized_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"image", "mask", "person_image", "garment_image"}
        }
        metadata.update(sanitized_payload)

        return GenerationResult(
            provider="azure_gpt_image_1",
            task_name=task_name,
            image_bytes=image_bytes,
            metadata=metadata,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
        reraise=True,
    )
    def cloth_swap(self, task_name: str, payload: Dict[str, Any]) -> GenerationResult:
        """
        Submit an image edit job using GPT-Image-1 and return the resulting image.

        Expected payload keys:
        - image: base64-encoded source image (required)
        - mask: base64-encoded mask image (optional)
        - prompt: positive guidance text (required)
        
        Note: The /images/edits endpoint does NOT support:
        - model (already in URL path)
        - negative_prompt (not part of OpenAI spec)
        """
        request_body = payload.copy()
        payload_keys_for_error = list(request_body.keys())
        
        # Remove parameters not supported by /images/edits endpoint
        request_body.pop("model", None)
        request_body.pop("negative_prompt", None)

        image_b64 = request_body.get("image")
        if not image_b64:
            raise RuntimeError("GPT-Image-1 payload missing required 'image' base64 data")
        if not request_body.get("prompt"):
            raise RuntimeError("GPT-Image-1 payload missing required 'prompt' value")

        # Remove mask if empty/None
        if not request_body.get("mask"):
            request_body.pop("mask", None)

        def decode_image_field(field_name: str, data_b64: str) -> bytes:
            try:
                return base64.b64decode(data_b64)
            except (ValueError, binascii.Error) as exc:
                raise RuntimeError(f"Invalid base64 data provided for '{field_name}'.") from exc

        def detect_file_metadata(field_name: str, data: bytes) -> tuple[str, str]:
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                return f"{field_name}.png", "image/png"
            if data[:3] == b"\xff\xd8\xff":
                return f"{field_name}.jpg", "image/jpeg"
            if data[:2] == b"BM":
                return f"{field_name}.bmp", "image/bmp"
            if data.startswith(b"GIF8"):
                return f"{field_name}.gif", "image/gif"
            return f"{field_name}.bin", "application/octet-stream"

        data_fields: Dict[str, str] = {}
        for key in ("prompt", "n", "size", "response_format", "user"):
            value = request_body.get(key)
            if value is not None:
                data_fields[key] = str(value)

        image_bytes = decode_image_field("image", image_b64)
        image_filename, image_content_type = detect_file_metadata("image", image_bytes)
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("image", (image_filename, image_bytes, image_content_type))
        ]

        mask_b64 = request_body.get("mask")
        if mask_b64:
            mask_bytes = decode_image_field("mask", mask_b64)
            with Image.open(io.BytesIO(mask_bytes)) as mask_image:
                mask_l = mask_image.convert("L")
                alpha = mask_l.point(lambda value: 0 if value > 127 else 255)
                mask_rgba = Image.new("RGBA", mask_l.size, (255, 255, 255, 255))
                mask_rgba.putalpha(alpha)
                buffer = io.BytesIO()
                mask_rgba.save(buffer, format="PNG")
                mask_bytes = buffer.getvalue()
            files.append(("mask", ("mask.png", mask_bytes, "image/png")))

        response = self._session.post(
            self._modification_path,
            params={"api-version": self._api_version},
            data=data_fields or None,
            files=files,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_detail = ""
            try:
                error_body = exc.response.json()
                error_detail = f"\nError response: {error_body}"
            except Exception:
                error_detail = f"\nError text: {exc.response.text}"
            
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Azure GPT-Image-1 edits endpoint not found at {self._modification_path}. "
                    f"Ensure AZURE_GPT_IMAGE_ENDPOINT includes the deployment edits path.{error_detail}"
                ) from exc
            elif exc.response.status_code == 400:
                raise RuntimeError(
                    f"Bad Request to Azure GPT-Image-1 at {self._modification_path}. "
                    f"Request payload keys: {payload_keys_for_error}{error_detail}"
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

    def __enter__(self) -> "AzureGPTImageClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401 - standard context manager
        self.close()
