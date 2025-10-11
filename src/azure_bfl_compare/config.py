from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator


class AzureFluxConfig(BaseModel):
    """Settings required to access the Azure-hosted Flux deployment."""

    endpoint: str = Field(..., description="Azure Cognitive Services endpoint URL")
    api_key: str = Field(..., description="Azure API key with permissions for the Flux deployment")
    deployment: str = Field(..., description="Azure Flux deployment name")
    api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure OpenAI API version to use for Flux image operations",
    )
    poll_interval_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Delay between polling attempts when waiting for Azure operations",
    )
    max_poll_attempts: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Maximum polling attempts before timing out Azure generation jobs",
    )


class AzureGPTImageConfig(BaseModel):
    """Settings required to access Azure GPT-Image-1 deployment for cloth swap."""

    endpoint: str = Field(..., description="Azure GPT-Image-1 endpoint URL")
    api_key: str = Field(..., description="Azure API key for GPT-Image-1")
    deployment: str = Field(..., description="Azure GPT-Image-1 deployment name")
    api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure API version for GPT-Image-1 operations",
    )
    poll_interval_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Delay between polling attempts when waiting for operations",
    )
    max_poll_attempts: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Maximum polling attempts before timing out",
    )


class BFLFluxConfig(BaseModel):
    """Settings required to access the official BFL Flux API."""

    api_url: str = Field(
        default="https://api.bfl.ai/v1/images",
        description="Base URL for the BFL Flux image generation endpoint",
    )
    api_key: str = Field(..., description="BFL Flux API key")


class OutputConfig(BaseModel):
    """Configuration for storing generated artifacts."""

    root_dir: Path = Field(default_factory=lambda: Path("output"))
    include_metadata: bool = Field(default=True, description="Persist provider metadata alongside images")


class AppConfig(BaseModel):
    """Top-level configuration object consumed by the batch runner."""

    azure: AzureFluxConfig | None = None
    azure_gpt_image: AzureGPTImageConfig | None = None
    bfl: BFLFluxConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    enable_azure_flux: bool = Field(
        default=True,
        description="Enable Azure Flux API integration for image generation",
    )
    enable_bfl: bool = Field(
        default=True,
        description="Enable BFL Flux API integration alongside Azure Flux comparisons",
    )
    enable_gpt_image: bool = Field(
        default=False,
        description="Enable Azure GPT-Image-1 for cloth swap operations",
    )

    @model_validator(mode="after")
    def _validate_providers(self) -> "AppConfig":
        if self.enable_azure_flux and self.azure is None:
            raise ValueError("Azure configuration is required when enable_azure_flux is True")
        if self.enable_bfl and self.bfl is None:
            raise ValueError("BFL configuration is required when enable_bfl is True")
        if self.enable_gpt_image and self.azure_gpt_image is None:
            raise ValueError("Azure GPT-Image configuration is required when enable_gpt_image is True")
        return self


def _bool_from_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float value: {value}") from exc


def _int_from_env(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value: {value}") from exc


def load_config(dotenv_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from environment variables (optionally seeded by a .env file).

    Parameters
    ----------
    dotenv_path:
        Optional override for the .env file location. Defaults to project root.

    Raises
    ------
    RuntimeError
        If required configuration values are missing.
    """
    env_path = Path(dotenv_path) if dotenv_path else Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    enable_azure_flux = _bool_from_env(os.getenv("ENABLE_AZURE_FLUX"), True)
    enable_bfl = _bool_from_env(os.getenv("ENABLE_BFL_FLUX"), True)
    enable_gpt_image = _bool_from_env(os.getenv("ENABLE_AZURE_GPT_IMAGE"), False)

    azure_data: dict[str, object] | None
    if enable_azure_flux:
        azure_data = {
            "endpoint": os.getenv("AZURE_FLUX_ENDPOINT"),
            "api_key": os.getenv("AZURE_FLUX_API_KEY"),
            "deployment": os.getenv("AZURE_FLUX_DEPLOYMENT"),
            "api_version": os.getenv("AZURE_FLUX_API_VERSION", "2024-12-01-preview"),
            "poll_interval_seconds": _float_from_env(
                os.getenv("AZURE_FLUX_POLL_INTERVAL"), 2.0
            ),
            "max_poll_attempts": _int_from_env(
                os.getenv("AZURE_FLUX_MAX_POLL_ATTEMPTS"), 60
            ),
        }
    else:
        azure_data = None

    bfl_data: dict[str, object] | None
    if enable_bfl:
        bfl_data = {
            "api_url": os.getenv("BFL_FLUX_API_URL", "https://api.bfl.ai/v1/images"),
            "api_key": os.getenv("BFL_FLUX_API_KEY"),
        }
    else:
        bfl_data = None

    gpt_image_data: dict[str, object] | None
    if enable_gpt_image:
        gpt_image_data = {
            "endpoint": os.getenv("AZURE_GPT_IMAGE_ENDPOINT"),
            "api_key": os.getenv("AZURE_GPT_IMAGE_API_KEY"),
            "deployment": os.getenv("AZURE_GPT_IMAGE_DEPLOYMENT", "gpt-image-1"),
            "api_version": os.getenv("AZURE_GPT_IMAGE_API_VERSION", "2024-12-01-preview"),
            "poll_interval_seconds": _float_from_env(
                os.getenv("AZURE_GPT_IMAGE_POLL_INTERVAL"), 2.0
            ),
            "max_poll_attempts": _int_from_env(
                os.getenv("AZURE_GPT_IMAGE_MAX_POLL_ATTEMPTS"), 60
            ),
        }
    else:
        gpt_image_data = None

    data = {
        "azure": azure_data,
        "azure_gpt_image": gpt_image_data,
        "bfl": bfl_data,
        "output": {
            "root_dir": Path(os.getenv("OUTPUT_ROOT_DIR", "output")),
            "include_metadata": _bool_from_env(os.getenv("OUTPUT_INCLUDE_METADATA"), True),
        },
        "enable_azure_flux": enable_azure_flux,
        "enable_bfl": enable_bfl,
        "enable_gpt_image": enable_gpt_image,
    }

    try:
        config = AppConfig.model_validate(data)
    except ValidationError as exc:
        missing = {"/".join(str(part) for part in err["loc"]) for err in exc.errors()}
        missing_str = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing configuration values: {missing_str}") from exc

    config.output.root_dir.mkdir(parents=True, exist_ok=True)
    return config
