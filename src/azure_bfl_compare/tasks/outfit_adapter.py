from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from .outfit_transfer import EnhancedOutfitTransferPipeline, OutfitPreparationResult


@dataclass(slots=True)
class OutfitJob:
    """Parameters required to run a single outfit swap."""

    task_name: str
    original_image: str
    clothes_image: str
    mask_image: str | None = None
    output_dir: Path | None = None
    strength_override: float | None = None
    sam_checkpoint: str | None = None
    sam_model_type: str | None = None
    device: str | None = None


@dataclass(slots=True)
class OutfitExecutionResult:
    """Captured result bytes and metadata from an Azure execution."""

    image_bytes: bytes
    metadata: MutableMapping[str, object]
    saved_files: MutableMapping[str, str]


class OutfitSwapAdapter:
    """
    Thin orchestration layer that mirrors scripts/outfit_swap_with_mask.py logic.

    The adapter is intentionally stateful so batch execution and the CLI share the same
    preparation and Azure invocation flow.
    """

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = dict(env or os.environ)
        self.use_flux = self._flag("ENABLE_AZURE_FLUX", True)
        self.use_gpt_image = self._flag("ENABLE_AZURE_GPT_IMAGE", False)

        self._pipelines: dict[str, EnhancedOutfitTransferPipeline] = {}
        self._preparations: dict[str, OutfitPreparationResult] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prepare(self, job: OutfitJob) -> OutfitPreparationResult:
        """Run the deterministic preparation pipeline."""
        pipeline = self._build_pipeline(job)

        output_dir = job.output_dir or Path("output") / "manual_outfit" / job.task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        preparation = pipeline.prepare(
            task_name=job.task_name,
            original_image=job.original_image,
            clothes_image=job.clothes_image,
            output_dir=output_dir,
            strength_override=job.strength_override,
            original_mask_path=job.mask_image,
        )

        self._pipelines[job.task_name] = pipeline
        self._preparations[job.task_name] = preparation
        return preparation

    def execute_flux(self, task_name: str, *, seed: int | None = None) -> OutfitExecutionResult:
        """Invoke Azure Flux using the prepared artifacts."""
        if not self.use_flux:
            raise RuntimeError("Azure Flux execution is disabled via environment flags.")

        pipeline = self._pipelines.get(task_name)
        preparation = self._preparations.get(task_name)
        if pipeline is None or preparation is None:
            raise RuntimeError(f"No prepared outfit swap found for task '{task_name}'.")

        response = pipeline.call_azure_flux_api(
            preparation,
            execute=True,
            seed=seed,
        )

        saved_files = dict(response.get("saved_files") or {})
        output_path_value = saved_files.get("output_image") or saved_files.get("output_flux")
        if not output_path_value:
            raise RuntimeError("Azure Flux response did not provide an output image path.")

        output_path = Path(output_path_value)
        if not output_path.exists():
            raise RuntimeError(f"Azure Flux output image not found at {output_path}")

        image_bytes = output_path.read_bytes()
        metadata: MutableMapping[str, object] = {
            "provider": "azure_flux",
            "status": response.get("status"),
            "saved_files": saved_files,
            "response": {
                key: value
                for key, value in response.items()
                if key not in {"saved_files"}
            },
            "_outfit_adapter": True,
        }
        return OutfitExecutionResult(
            image_bytes=image_bytes,
            metadata=metadata,
            saved_files=saved_files,
        )

    def execute_gpt_image(self, task_name: str) -> OutfitExecutionResult:
        """Invoke Azure GPT-Image-1 edits endpoint using prepared assets."""
        pipeline = self._pipelines.get(task_name)
        preparation = self._preparations.get(task_name)
        if pipeline is None or preparation is None:
            raise RuntimeError(f"No prepared outfit swap found for task '{task_name}'.")

        response = pipeline.call_azure_gpt_image_api(
            preparation,
            execute=True,
        )

        saved_files = dict(response.get("saved_files") or {})
        output_path_value = saved_files.get("output_image")
        if not output_path_value:
            raise RuntimeError("Azure GPT-Image response did not provide an output image path.")

        output_path = Path(output_path_value)
        if not output_path.exists():
            raise RuntimeError(f"Azure GPT-Image output image not found at {output_path}")

        image_bytes = output_path.read_bytes()
        metadata: MutableMapping[str, object] = {
            "provider": "azure_gpt_image",
            "status": response.get("status"),
            "saved_files": saved_files,
            "response": {
                key: value
                for key, value in response.items()
                if key not in {"saved_files"}
            },
            "_outfit_adapter": True,
        }
        return OutfitExecutionResult(
            image_bytes=image_bytes,
            metadata=metadata,
            saved_files=saved_files,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _flag(self, name: str, default: bool) -> bool:
        value = self._env.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _build_pipeline(self, job: OutfitJob) -> EnhancedOutfitTransferPipeline:
        """Instantiate a pipeline with env + job level overrides."""
        return EnhancedOutfitTransferPipeline(
            sam_checkpoint=job.sam_checkpoint,
            sam_model_type=job.sam_model_type,
            device=job.device,
            azure_gpt_endpoint=self._env.get("AZURE_GPT_ENDPOINT"),
            azure_gpt_key=self._env.get("AZURE_GPT_API_KEY"),
            azure_gpt_deployment=self._env.get("AZURE_GPT_DEPLOYMENT"),
            azure_gpt_api_version=self._env.get("AZURE_GPT_API_VERSION"),
            azure_gpt_image_endpoint=self._env.get("AZURE_GPT_IMAGE_ENDPOINT"),
            azure_gpt_image_key=self._env.get("AZURE_GPT_IMAGE_API_KEY"),
            azure_gpt_image_deployment=self._env.get("AZURE_GPT_IMAGE_DEPLOYMENT"),
            azure_gpt_image_api_version=self._env.get("AZURE_GPT_IMAGE_API_VERSION"),
            use_gpt_image=self.use_gpt_image or not self.use_flux,
            azure_flux_endpoint=self._env.get("AZURE_FLUX_ENDPOINT") if self.use_flux else None,
            azure_flux_key=self._env.get("AZURE_FLUX_API_KEY") if self.use_flux else None,
            azure_flux_deployment=self._env.get("AZURE_FLUX_DEPLOYMENT") if self.use_flux else None,
            azure_flux_api_version=self._env.get("AZURE_FLUX_API_VERSION") if self.use_flux else None,
            deeplab_model_path=self._env.get("DEEPLAB_ONNX_PATH"),
        )

