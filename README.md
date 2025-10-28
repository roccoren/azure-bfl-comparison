# Azure & BFL Flux Batch Comparator

This project compares image outputs from Microsoft Azure Flux API and the official Black Forest Labs Flux API using a shared batch definition.

## Features

- Load `.env` credentials for both providers.
- Define batch tasks (prompts, operations) in JSON.
- Execute Azure and BFL requests per task.
- Persist generated images and metadata for side-by-side evaluation.
- Rich console progress display and retry logic.
- Output storage grouped by batch name and provider with JSON metadata.
- Manual outfit swap CLI that reuses the enhanced pipeline with a user-supplied mask.

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

## Manual outfit swap with a provided mask

When you already have an inpainting mask for the target person, you can reuse the enhanced
pipeline without running any automatic mask generation. The helper script below prepares the
combined panel, prompts, and (optionally) submits the request to Azure Flux:

```bash
python scripts/outfit_swap_with_mask.py \
  --original path/to/person.jpg \
  --mask path/to/person_mask.png \
  --clothes path/to/garment.jpg \
  --task-name sample_swap \
  --execute
```

- Drop `--execute` for a dry run that only writes the intermediate artifacts under
  `output/manual_outfit/sample_swap/`.
- The script automatically loads `.env` from the project root when present; pass `--dotenv`
  to point at a different credentials file.
- Supply `--strength` to override the computed inpainting strength or `--seed` to make the
  Azure Flux call deterministic.
- Execution respects the feature flags: set `ENABLE_AZURE_FLUX=false` and
  `ENABLE_AZURE_GPT_IMAGE=true` to call GPT-Image-1 directly.
- Prefer batching? Point `scripts/run_batch.py` at a JSON task file that includes both
  `image_path` and `mask_path`. See [`examples/outfit_swap_with_masks.json`](examples/outfit_swap_with_masks.json)
  for a template. With `ENABLE_AZURE_GPT_IMAGE=true`, the runner will reuse the
  provided mask and call GPT-Image-1 for each task.

### Batch outfit swap via JSON

To batch multiple swaps, add tasks to a JSON definition (see `examples/outfit_swap_try_on.json`).
Each task should specify:

- `image_path`: original person photo
- `mask_path`: binary/white mask covering only the garment region
- `outfit.clothes_image_path`: reference garment

Then run:

```bash
python scripts/run_batch.py examples/outfit_swap_try_on.json --batch-name try_on --dotenv .env
```

For datasets that already include hand-crafted masks, you can reuse the manual CLI via the helper runner:

```bash
python scripts/run_outfit_swap_with_masks.py examples/outfit_swap_with_masks.json --execute --dotenv .env
```

Set `ENABLE_AZURE_FLUX=false` and `ENABLE_AZURE_GPT_IMAGE=true` in `.env` to route calls to GPT-Image-1; the runner
will convert supplied masks to transparent regions so the model only edits the white area.

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
