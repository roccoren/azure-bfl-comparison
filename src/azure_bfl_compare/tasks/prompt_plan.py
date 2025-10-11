from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Literal, Sequence

import json
from pydantic import BaseModel, Field, RootModel, ValidationError


class OutfitTransferConfig(BaseModel):
    """Configuration flags for enhanced outfit transfer preparation."""

    clothes_image_path: str = Field(
        ..., description="Path to the clothing reference image used for analysis"
    )
    output_subdir: str | None = Field(
        default=None,
        description="Optional custom subdirectory for intermediate artifacts",
    )
    strength: float | None = Field(
        default=None,
        description="Manual override for the auto-calculated Flux strength",
    )
    sam_checkpoint: str | None = Field(
        default=None,
        description="Optional SAM checkpoint override for segmentation",
    )
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = Field(
        default="vit_h", description="SAM model variant to load when available"
    )
    device: str | None = Field(
        default=None,
        description="Execution device override (e.g., 'cuda' or 'cpu')",
    )


class TaskPayload(BaseModel):
    """Payload forwarded to each provider per task."""

    prompt: str = Field(..., description="Primary text prompt")
    negative_prompt: str | None = Field(default=None)
    guidance_scale: float | None = Field(default=None)
    num_inference_steps: int | None = Field(default=None)
    size: str | None = Field(default=None, description="Size spec e.g. '1024x1024'")
    image_path: str | None = Field(
        default=None,
        description="Path to the base image used for modification (required for edits)",
    )
    mask_path: str | None = Field(
        default=None,
        description="Optional mask image path; transparent regions will be regenerated",
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific overrides")
    outfit: OutfitTransferConfig | None = Field(
        default=None,
        description="Enhanced outfit transfer configuration; enables automated prompt and mask preparation",
    )


class BatchTask(BaseModel):
    """Single generation task identified by name."""

    name: str
    payload: TaskPayload


class BatchImage(BaseModel):
    """Image source reused across multiple prompt profiles."""

    name: str = Field(..., description="Identifier applied to task names when combined with a profile")
    image_path: str = Field(..., description="Path to the base image that will be edited")
    mask_path: str | None = Field(
        default=None, description="Optional mask restricting the editable region for this image"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific overrides merged into the profile payload for this image",
    )
    payload_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Shallow overrides applied to the composed payload (e.g., guidance_scale adjustments)",
    )


class BatchProfile(BaseModel):
    """Prompt profile applied to one or more images."""

    name: str = Field(..., description="Identifier applied to task names when combined with an image")
    payload: TaskPayload = Field(
        ...,
        description="Base payload describing the prompt characteristics (image fields will be supplied per image)",
    )


class BatchMatrixDefinition(BaseModel):
    """
    Matrix-style batch definition that composes images with shared prompt profiles.

    Each profile is executed for every image, producing the cartesian product of tasks.
    """

    images: List[BatchImage]
    profiles: List[BatchProfile]
    name_pattern: str = Field(
        default="{image}-{profile}",
        description=(
            "Python str.format template used to derive task names. "
            "Available placeholders: {image}, {profile}, {index}, {index1}, "
            "{image_index}, {image_index1}, {profile_index}, {profile_index1}."
        ),
    )

    def to_batch_tasks(self) -> List[BatchTask]:
        """Expand the matrix definition into individual batch tasks."""
        tasks: List[BatchTask] = []
        for image_index, image in enumerate(self.images):
            for profile_index, profile in enumerate(self.profiles):
                context = {
                    "image": image.name,
                    "profile": profile.name,
                    "index": len(tasks),
                    "index1": len(tasks) + 1,
                    "image_index": image_index,
                    "image_index1": image_index + 1,
                    "profile_index": profile_index,
                    "profile_index1": profile_index + 1,
                }
                try:
                    task_name = self.name_pattern.format(**context)
                except KeyError as exc:  # pragma: no cover - defensive guard
                    missing = exc.args[0]
                    raise RuntimeError(
                        f"Invalid name_pattern placeholder '{missing}'. "
                        "Supported placeholders: {image}, {profile}, {index}, {index1}, "
                        "{image_index}, {image_index1}, {profile_index}, {profile_index1}."
                    ) from exc

                payload_dict = profile.payload.model_copy(deep=True).model_dump(mode="python")

                # Ensure image & mask fields reflect the current image.
                payload_dict["image_path"] = image.image_path
                mask_value = image.mask_path if image.mask_path is not None else payload_dict.get("mask_path")
                payload_dict["mask_path"] = mask_value

                # Merge extras with image-level overrides taking precedence.
                base_extra = payload_dict.get("extra") or {}
                if image.extra:
                    merged_extra = {**base_extra, **image.extra}
                else:
                    merged_extra = base_extra
                payload_dict["extra"] = merged_extra

                if image.payload_overrides:
                    payload_dict.update(image.payload_overrides)

                payload = TaskPayload.model_validate(payload_dict)
                tasks.append(BatchTask(name=task_name, payload=payload))
        return tasks


class BatchDefinition(RootModel[List[BatchTask]]):
    """Collection of tasks loaded from a batch JSON document."""

    @classmethod
    def from_tasks(cls, tasks: Sequence[BatchTask]) -> "BatchDefinition":
        return cls(root=list(tasks))

    def task_names(self) -> Sequence[str]:
        return [task.name for task in self.root]

    def __iter__(self) -> Iterable[BatchTask]:
        return iter(self.root)


def load_batch_definition(path: str | Path) -> BatchDefinition:
    """Load and validate a batch task file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        try:
            return BatchDefinition.model_validate(data)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid batch definition at {path}") from exc

    if isinstance(data, dict):
        try:
            matrix = BatchMatrixDefinition.model_validate(data)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid matrix batch definition at {path}") from exc
        return BatchDefinition.from_tasks(matrix.to_batch_tasks())

    raise RuntimeError(
        f"Unsupported batch definition structure at {path}; expected list or matrix-style object."
    )
