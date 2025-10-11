from __future__ import annotations

import base64
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from ..config import AppConfig
from ..clients.azure_flux import AzureFluxClient
from ..clients.azure_gpt_image import AzureGPTImageClient
from ..clients.bfl_flux import BFLFluxClient
from ..output.comparer import OutputStore
from ..types import GenerationResult
from .outfit_transfer import EnhancedOutfitTransferPipeline, OutfitPreparationResult
from .prompt_plan import BatchDefinition, BatchTask


class BatchRunner:
    """Coordinate Azure/BFL generations for a batch of tasks."""

    @staticmethod
    def _encode_file(path: str, kind: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            raise RuntimeError(f"{kind} file not found: {file_path}")
        return base64.b64encode(file_path.read_bytes()).decode("utf-8")

    def __init__(
        self,
        config: AppConfig,
        batch: BatchDefinition,
        batch_name: str,
        console: Console | None = None,
    ) -> None:
        self._config = config
        self._batch = batch
        self._batch_name = batch_name
        self._console = console or Console()
        self._store = OutputStore(config.output.root_dir, batch_name=batch_name)
        self._outfit_context: dict[str, OutfitPreparationResult] = {}

    def _prepare_tasks(self, tasks: Iterable[BatchTask]) -> None:
        """Expand enhanced outfit transfer tasks into combined image workflows."""

        for task in tasks:
            outfit_config = task.payload.outfit
            if outfit_config is None:
                continue

            original_path = task.payload.image_path
            original_mask_path = task.payload.mask_path
            if not original_path:
                raise RuntimeError(
                    f"Task '{task.name}' requires 'image_path' when outfit transfer is enabled."
                )

            clothes_path = outfit_config.clothes_image_path
            if not Path(clothes_path).exists():
                raise RuntimeError(
                    f"Task '{task.name}' clothing reference not found: {clothes_path}"
                )

            output_subdir = outfit_config.output_subdir or task.name
            output_dir = self._store.base_dir / "outfit" / output_subdir

            pipeline = EnhancedOutfitTransferPipeline(
                sam_checkpoint=outfit_config.sam_checkpoint,
                sam_model_type=outfit_config.sam_model_type,
                device=outfit_config.device,
                azure_gpt_endpoint=os.getenv("AZURE_GPT_ENDPOINT"),
                azure_gpt_key=os.getenv("AZURE_GPT_API_KEY"),
                azure_gpt_deployment=os.getenv("AZURE_GPT_DEPLOYMENT"),
                azure_gpt_api_version=os.getenv("AZURE_GPT_API_VERSION"),
                azure_gpt_image_endpoint=os.getenv("AZURE_GPT_IMAGE_ENDPOINT"),
                azure_gpt_image_key=os.getenv("AZURE_GPT_IMAGE_API_KEY"),
                azure_gpt_image_deployment=os.getenv("AZURE_GPT_IMAGE_DEPLOYMENT"),
                azure_gpt_image_api_version=os.getenv("AZURE_GPT_IMAGE_API_VERSION"),
                use_gpt_image=os.getenv("ENABLE_AZURE_GPT_IMAGE", "false").lower() == "true",
                deeplab_model_path=os.getenv("DEEPLAB_ONNX_PATH"),
            )

            preparation = pipeline.prepare(
                task_name=task.name,
                original_image=original_path,
                clothes_image=clothes_path,
                output_dir=output_dir,
                strength_override=outfit_config.strength,
                original_mask_path=original_mask_path,
            )

            new_extra = self._merge_strength(task.payload.extra, preparation.strength)
            updated_payload = task.payload.model_copy(
                update={
                    "prompt": preparation.prompt,
                    "negative_prompt": preparation.negative_prompt,
                    "image_path": str(preparation.combined_image_path),
                    "mask_path": str(preparation.combined_mask_path),
                    "extra": new_extra,
                }
            )
            task.payload = updated_payload
            self._outfit_context[task.name] = preparation

    @staticmethod
    def _merge_strength(extra: dict[str, object], strength: float) -> dict[str, object]:
        """Inject the computed strength value into the provider extra payload."""

        merged = dict(extra or {})
        input_block = dict(merged.get("input") or {})
        input_block["strength"] = round(strength, 4)
        merged["input"] = input_block
        return merged

    def _post_process_result(self, task_name: str, result: GenerationResult) -> GenerationResult:
        """Crop combined outputs and attach metadata for outfit transfer tasks."""

        preparation = self._outfit_context.get(task_name)
        if preparation is None:
            return result

        metadata = result.mutable_metadata()
        for key in ("image", "mask", "person_image", "garment_image"):
            metadata.pop(key, None)
        outfit_meta = metadata.setdefault("outfit_transfer", {})
        outfit_meta.update(preparation.metadata())
        outfit_meta["strength"] = preparation.strength

        processed_bytes = preparation.crop_output(result.image_bytes)

        return GenerationResult(
            provider=result.provider,
            task_name=result.task_name,
            image_bytes=processed_bytes,
            metadata=metadata,
        )

    def _run_for_provider(
        self,
        client_name: str,
        tasks: Iterable[BatchTask],
        run_callable,
        include_metadata: bool,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        tasks_list = list(tasks)
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]{client_name}[/bold]"),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("queued", total=len(tasks_list))
            for item in tasks_list:
                progress.update(task_id, description=f"{item.name}", advance=0)
                result = run_callable(item)
                result = self._post_process_result(item.name, result)
                results.append(result)
                self._store.save(result, include_metadata=include_metadata)
                progress.advance(task_id)
        return results

    def run(self) -> None:
        """Execute the batch for both providers."""
        include_metadata = self._config.output.include_metadata
        azure_enabled = self._config.enable_azure_flux and self._config.azure is not None
        bfl_enabled = self._config.enable_bfl

        with ExitStack() as stack:
            azure_client: AzureFluxClient | None = None
            if azure_enabled:
                azure_client = stack.enter_context(AzureFluxClient(self._config.azure))
            azure_gpt_client: AzureGPTImageClient | None = None
            if self._config.enable_gpt_image and self._config.azure_gpt_image is not None:
                azure_gpt_client = stack.enter_context(
                    AzureGPTImageClient(self._config.azure_gpt_image)
                )
            bfl_client: BFLFluxClient | None = None
            if bfl_enabled and self._config.bfl is not None:
                bfl_client = stack.enter_context(BFLFluxClient(self._config.bfl))

            tasks = list(self._batch)
            self._prepare_tasks(tasks)
            gpt_ready_tasks = [task for task in tasks if task.payload.image_path]

            if azure_enabled and azure_client is not None:

                def run_azure(task: BatchTask) -> GenerationResult:
                    payload = task.payload.model_dump(mode="python")
                    extra = payload.pop("extra", {})
                    image_path = payload.pop("image_path", None)
                    mask_path = payload.pop("mask_path", None)
                    payload.pop("outfit", None)

                    if not image_path:
                        raise RuntimeError(
                            f"Task '{task.name}' requires 'image_path' for modification workflow."
                        )

                    azure_payload: dict[str, object] = {k: v for k, v in payload.items() if v is not None}
                    azure_payload.update(extra)
                    azure_payload["image"] = self._encode_file(image_path, "Base image")
                    if mask_path:
                        azure_payload["mask"] = self._encode_file(mask_path, "Mask image")

                    return azure_client.generate(task.name, azure_payload)

                self._console.rule("Azure Flux")
                self._run_for_provider("Azure", tasks, run_azure, include_metadata)
            else:
                self._console.print(
                    "[yellow]Skipping Azure Flux generation (disabled or missing configuration).[/yellow]"
                )

            if azure_gpt_client is not None and gpt_ready_tasks:

                def run_gpt_image(task: BatchTask) -> GenerationResult:
                    payload = task.payload.model_dump(mode="python")
                    extra = payload.pop("extra", {})
                    image_path = payload.pop("image_path", None)
                    mask_path = payload.pop("mask_path", None)
                    payload.pop("outfit", None)

                    gpt_payload: dict[str, object] = {k: v for k, v in payload.items() if v is not None}
                    gpt_payload.update(extra)

                    preparation = self._outfit_context.get(task.name)
                    if preparation is not None:
                        input_block = extra.get("input") if isinstance(extra, dict) else None
                        if isinstance(input_block, dict) and "strength" in input_block:
                            gpt_payload["image_strength"] = input_block["strength"]
                        gpt_payload.pop("input", None)
                        image_source = str(preparation.combined_image_path)
                        mask_source = str(preparation.combined_mask_path)
                    else:
                        gpt_payload.pop("input", None)
                        image_source = image_path or task.payload.image_path
                        mask_source = mask_path or task.payload.mask_path

                    if not image_source:
                        raise RuntimeError(
                            f"Task '{task.name}' requires 'image_path' for GPT-Image workflow."
                        )

                    gpt_payload["image"] = self._encode_file(image_source, "Base image")
                    if mask_source:
                        gpt_payload["mask"] = self._encode_file(mask_source, "Mask image")

                    # Azure GPT-Image edits endpoint follows OpenAI spec, which only accepts:
                    # image, prompt, mask, n, size, response_format, user
                    # Drop all other parameters that cause 400 Bad Request
                    allowed_gpt_keys = {
                        "prompt",
                        "image",
                        "mask",
                        "model",
                        "n",
                        "size",
                        "response_format",
                        "user",
                    }
                    gpt_payload = {
                        key: value
                        for key, value in gpt_payload.items()
                        if key in allowed_gpt_keys
                    }
                    
                    # Remove model if it's already in the deployment path
                    # The endpoint URL already includes the deployment name
                    gpt_payload.pop("model", None)

                    return azure_gpt_client.cloth_swap(task.name, gpt_payload)

                self._console.rule("Azure GPT-Image")
                self._run_for_provider(
                    "Azure (GPT-Image)", gpt_ready_tasks, run_gpt_image, include_metadata
                )
            elif azure_gpt_client is not None:
                self._console.print(
                    "[yellow]Skipping Azure GPT-Image run (no tasks with image data).[/yellow]"
                )

            if not bfl_client:
                self._console.print(
                    "[yellow]Skipping BFL Flux generation (enable_bfl is False).[/yellow]"
                )
            else:

                def run_bfl(task: BatchTask) -> GenerationResult:
                    payload = task.payload.model_dump(mode="python")
                    extra = payload.pop("extra", {})
                    image_path = payload.pop("image_path", None)
                    mask_path = payload.pop("mask_path", None)
                    payload.pop("outfit", None)

                    if not image_path:
                        raise RuntimeError(
                            f"Task '{task.name}' requires 'image_path' for modification workflow."
                        )

                    bfl_input: dict[str, object] = {"prompt": payload["prompt"]}
                    if payload.get("negative_prompt"):
                        bfl_input["negative_prompt"] = payload["negative_prompt"]
                    if payload.get("guidance_scale") is not None:
                        bfl_input["guidance_scale"] = payload["guidance_scale"]
                    if payload.get("num_inference_steps") is not None:
                        bfl_input["num_inference_steps"] = payload["num_inference_steps"]
                    if payload.get("size"):
                        bfl_input["size"] = payload["size"]

                    bfl_input["image"] = self._encode_file(image_path, "Base image")
                    if mask_path:
                        bfl_input["mask"] = self._encode_file(mask_path, "Mask image")

                    bfl_payload: dict[str, object] = {"input": bfl_input}
                    bfl_payload.update(extra)
                    return bfl_client.generate(task.name, bfl_payload)

                self._console.rule("BFL Flux")
                self._run_for_provider("BFL", tasks, run_bfl, include_metadata)

        self._console.print(f"[green]Outputs saved to[/green] {self._store.base_dir}")
