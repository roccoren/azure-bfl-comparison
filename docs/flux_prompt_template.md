# Flux Prompt Template Reference

This document describes each field in [`examples/flux_prompt_template.json`](examples/flux_prompt_template.json:1-86) so you can confidently customize the batch configuration for Flux-powered image editing pipelines.

## Top-Level Structure

| Field | Type | Description |
| --- | --- | --- |
| `template_name` | string | Human-readable identifier for this template configuration. |
| `description` | string | Summary of the template’s purpose or usage scenario. |
| `metadata` | object | Records template metadata such as versioning, authorship, and timestamps. |
| `defaults` | object | Global fallback values applied to every job when the job omits the corresponding field. |
| `jobs` | array<object> | List of job payload blueprints. Duplicate a job object per image or prompt variation. |
| `notes` | array<string> | Guidance for customizing or extending the template. |

## `metadata` Block

| Field | Type | Description |
| --- | --- | --- |
| `version` | string | Semantic version of the template. Update when structure or defaults change. |
| `created_utc` | string (ISO 8601 UTC) | Creation timestamp for auditing. |
| `author` | string | Person or system responsible for authoring the template. |

## `defaults` Block

These values apply across all jobs unless overridden inside a job’s `payload`.

| Field | Type | Description |
| --- | --- | --- |
| `provider` | string | Backend provider (e.g., `azure`). |
| `response_format` | array<string> | Response fields expected from the provider. |
| `negative_prompt` | string | Global negative prompt to suppress unwanted artifacts. |
| `guidance_scale` | number | Controls adherence to the prompt; higher means stricter guidance. |
| `num_inference_steps` | integer | Number of diffusion steps per job. |
| `image_path` | string | Default path to the source image. |
| `configuration` | object | Provider-specific tweaks. |
| `configuration.safety` | string | Safety mode (e.g., `standard`). |
| `configuration.edit_type` | string | Editing strategy (e.g., `color_adjustment`). |
| `input` | object | Parameters passed to the model input block. |
| `input.strength` | number | Degree of transformation applied to the source image. |
| `extra` | object | Optional advanced controls. |
| `extra.scheduler` | string | Diffusion scheduler selection (e.g., `euler_a`). |
| `extra.output_format` | string | Expected output file format (`png`, `jpg`, etc.). |
| `extra.metadata` | object | Free-form metadata for downstream systems. |
| `extra.metadata.subject` | string | Subject focus tag. |
| `extra.metadata.style` | string | Style or art direction tag. |
| `extra.metadata.notes` | string | Additional remarks. |
| `seed` | integer or null | Seed for reproducibility. |
| `enabled` | boolean | Toggle to activate or skip all jobs by default. |
| `mask_path` | string | Default path to the mask image paired with the source asset. |

## `jobs` Array

Each entry represents a single Flux invocation blueprint. Replace placeholder tokens (`{{...}}`) with actual values or remove keys you do not need.

### Job-Level Fields

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Unique identifier for this job instance. |
| `payload` | object | Request payload sent to the Flux backend. |
| `output` | object | Output naming conventions for generated assets. |
| `enabled` | boolean or string token | Job-level toggle overriding the default `enabled` flag. |

### `payload` Fields

| Field | Type | Description |
| --- | --- | --- |
| `prompt` | string | Primary text prompt describing the desired transformation. |
| `negative_prompt` | string | Job-specific negative prompt (falls back to default if omitted). |
| `guidance_scale` | number or token | Overrides the default guidance scale. |
| `num_inference_steps` | integer or token | Overrides the default inference steps. |
| `image_path` | string | Path to the source image; can differ per job. |
| `response_format` | array<string> | Response fields requested from the backend. |
| `provider` | string | Provider override if a job targets a different backend. |
| `configuration.safety` | string | Job-specific safety override. |
| `configuration.edit_type` | string | Job-specific edit strategy override. |
| `input.strength` | number | Strength override. |
| `extra.scheduler` | string | Scheduler override. |
| `extra.output_format` | string | Desired output format override. |
| `extra.metadata.subject` | string | Subject override. |
| `extra.metadata.style` | string | Style override. |
| `extra.metadata.notes` | string | Additional notes override. |
| `mask_path` | string | Mask image to apply for this job (overrides the default mask). |
| `seed` | integer or token | Seed override for deterministic runs. |

### `output` Fields

| Field | Type | Description |
| --- | --- | --- |
| `directory` | string | Destination directory for generated outputs. |
| `filename` | string | Base filename (without extension) for the generated asset. |

## Usage Notes

1. Duplicate the job object for each variation and replace placeholders with concrete values.  
2. Any omitted job field inherits from `defaults`.  
3. Remove unused keys entirely to avoid sending empty strings to the backend.  
4. Ensure image and mask paths are accessible to the Flux runtime in your deployment environment.