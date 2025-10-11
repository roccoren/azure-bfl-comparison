# Azure & BFL Flux Batch Comparator

This project compares image outputs from Microsoft Azure Flux API and the official Black Forest Labs Flux API using a shared batch definition.

## Features

- Load `.env` credentials for both providers.
- Define batch tasks (prompts, operations) in JSON.
- Execute Azure and BFL requests per task.
- Persist generated images and metadata for side-by-side evaluation.
- Rich console progress display and retry logic.
- Output storage grouped by batch name and provider with JSON metadata.

## Quickstart

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e '.[dev]'
   ```
> **Note:** zsh treats square brackets as glob patterns—quote the extras spec exactly as shown.

2. Copy `.env.example` to `.env` and set the provider secrets.
3. Edit `examples/single_image_batch.json` to match your prompts.
4. Run the batch:

   ```bash
   python scripts/run_batch.py examples/single_image_batch.json
   ```

Outputs are saved to `output/` with timestamped folders for each provider.

## Configuration

Environment variables are loaded from `.env` (copy [`.env.example`](.env.example)). Required keys:

- `AZURE_FLUX_ENDPOINT` (use the FULL Azure image generation URL from the portal, e.g. `https://resource.services.ai.azure.com/openai/deployments/FLUX.1-pro/images/generations?api-version=2025-04-01-preview`)
- `AZURE_FLUX_API_KEY`
- `AZURE_FLUX_DEPLOYMENT` (your Flux deployment name)
- `BFL_FLUX_API_KEY`

Optional overrides:

- `AZURE_FLUX_API_VERSION` (extracted from endpoint URL or defaults to `2024-12-01-preview`)
- `AZURE_FLUX_POLL_INTERVAL` (seconds between status checks, default `2.0`)
- `AZURE_FLUX_MAX_POLL_ATTEMPTS` (default `60`)
- `BFL_FLUX_API_URL` (defaults to the public API)
- `ENABLE_BFL_FLUX` (`true` by default; set `false` to skip BFL requests and only call Azure)
- `OUTPUT_ROOT_DIR`
- `OUTPUT_INCLUDE_METADATA` (`true`/`false`)

**Important:** For Azure, copy the complete image generation endpoint URL from your Azure portal, including the deployment name and API version query parameter.

## Batch definition format

Batch files are JSON arrays. Each entry is a task:

```json
{
  "name": "task-identifier",
  "payload": {
    "prompt": "primary description",
    "negative_prompt": "optional negatives",
    "guidance_scale": 8.0,
    "num_inference_steps": 30,
    "size": "1024x1024",
    "extra": {
      "...": "provider-specific configuration"
    }
  }
}
```

- `prompt` is required. Other keys are optional.
- `extra` is merged directly into provider payloads, letting you supply Azure `configuration` options or BFL-specific flags.

See [`examples/single_image_batch.json`](examples/single_image_batch.json) for a full sample.

## Output structure

Each run produces `output/{batch_name}_{timestamp}/`. Inside this directory, provider folders contain:

- `{task}.png` — generated image.
- `{task}.json` — metadata including request payload (if `OUTPUT_INCLUDE_METADATA=true`).

This layout makes it easy to compare results from Azure and BFL for the same task names.