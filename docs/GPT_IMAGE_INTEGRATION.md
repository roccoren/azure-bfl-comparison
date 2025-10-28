# GPT-Image-1 Integration for Cloth Swap

This document describes the integration of Azure GPT-Image-1 model for cloth swap operations in the outfit transfer pipeline.

## Overview

The outfit transfer pipeline now supports two modes for cloth swap:

1. **Azure Flux Kontext** (default) - Uses mask-based inpainting with detailed prompts
2. **Azure GPT-Image-1** (optional) - Direct cloth swap using AI-powered garment transfer

## Architecture

### New Components

1. **AzureGPTImageClient** (`src/azure_bfl_compare/clients/azure_gpt_image.py`)
   - Dedicated client for GPT-Image-1 cloth swap endpoint
   - Handles base64 encoding/decoding of images
   - Supports async operations with polling

2. **AzureGPTImageConfig** (`src/azure_bfl_compare/config.py`)
   - Configuration model for GPT-Image-1 credentials
   - Validation and defaults

3. **Enhanced Pipeline** (`src/azure_bfl_compare/tasks/outfit_transfer.py`)
   - `call_azure_gpt_image_api()` - New method for GPT-Image-1 calls
   - Automatic routing based on `use_gpt_image` flag

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Enable GPT-Image-1 for cloth swap
ENABLE_AZURE_GPT_IMAGE=true

# GPT-Image-1 credentials
AZURE_GPT_IMAGE_ENDPOINT=https://your-resource.openai.azure.com
AZURE_GPT_IMAGE_API_KEY=your-api-key-here
AZURE_GPT_IMAGE_DEPLOYMENT=gpt-image-1
AZURE_GPT_IMAGE_API_VERSION=2024-12-01-preview

# Optional: Polling configuration
AZURE_GPT_IMAGE_POLL_INTERVAL=2.0
AZURE_GPT_IMAGE_MAX_POLL_ATTEMPTS=60

# Optional: Parallel edit configuration
AZURE_GPT_IMAGE_MAX_CONCURRENCY=5
```

### Configuration in Code

```python
from azure_bfl_compare.tasks.outfit_transfer import EnhancedOutfitTransferPipeline

# Create pipeline with GPT-Image-1 enabled
pipeline = EnhancedOutfitTransferPipeline(
    use_gpt_image=True,
    azure_gpt_image_endpoint="https://your-resource.openai.azure.com",
    azure_gpt_image_key="your-key",
    azure_gpt_image_deployment="gpt-image-1",
)

# Prepare outfit transfer
preparation = pipeline.prepare(
    task_name="my_outfit_swap",
    original_image="path/to/person.jpg",
    clothes_image="path/to/garment.jpg",
    output_dir=Path("output/my_task"),
)

# Execute with GPT-Image-1
result = pipeline.call_azure_flux_api(preparation, execute=True)
```

## How It Works

### GPT-Image-1 Mode

When `ENABLE_AZURE_GPT_IMAGE=true`:

1. **Preparation Phase** (unchanged)
   - Loads person and garment images
   - Detects person orientation
   - Analyzes clothing attributes
   - Generates prompts

2. **API Call Phase** (new routing)
   - Encodes person image to base64
   - Encodes garment image to base64
   - Calls GPT-Image-1 cloth-swap endpoint with:
     - `person_image`: Original person photo
     - `garment_image`: Clothing reference
     - `prompt`: Generated description
     - `negative_prompt`: Preservation constraints

3. **Output Phase**
   - Receives swapped image directly
   - No mask-based cropping needed
   - Saves to `output_gpt_image.jpg`

### Prompt Strategy for GPT-Image-1

GPT-Image-1 expects edit-style instructions that stress preservation. The pipeline now composes a specialised prompt bundle that:

- Directs the model to edit the original `person_image`, not generate a new subject or scene.
- Repeats preservation rules (identity, pose, background, lighting, framing) so the model keeps the source photo intact.
- Limits the allowed edit area to the clothing slots detected in Step 3.
- Appends negative prompts discouraging subject replacement, background changes, reframing, artifacts, and other regressions.
- Merges any additional negative phrases from the batch definition.

This prompt bundle is assembled in [`outfit_transfer.EnhancedOutfitTransferPipeline._build_gpt_image_prompt_bundle()`](src/azure_bfl_compare/tasks/outfit_transfer.py:1303-1340) and automatically supplied to GPT-Image-1 calls.

### Azure Flux Mode (Default)

When `ENABLE_AZURE_GPT_IMAGE=false` or not set:

1. Uses the traditional mask-based inpainting approach
2. Creates combined panel with person + garment reference
3. Applies mask to target clothing region
4. Calls Azure Flux Kontext edits endpoint

## Comparison: GPT-Image-1 vs Azure Flux

| Feature | GPT-Image-1 | Azure Flux Kontext |
|---------|-------------|-------------------|
| **Input** | Person + Garment images | Person + Garment + Mask |
| **Approach** | Direct cloth swap AI | Mask-based inpainting |
| **Prompt Complexity** | Simple description | Detailed with preservation |
| **Mask Required** | No | Yes |
| **Person Preservation** | AI-based | Mask-controlled |
| **Best For** | Full garment replacement | Precise regional edits |

## Benefits of GPT-Image-1

1. **Simpler Pipeline**
   - No mask generation needed
   - Direct image-to-image transfer

2. **Better Understanding**
   - AI understands garment structure
   - Natural fitting to body shape

3. **Reduced Complexity**
   - No combined panel creation
   - No letterboxing/cropping

4. **Potential Quality**
   - May handle complex garments better
   - Could preserve person features more naturally

## Testing

Run the integration test:

```bash
.venv/bin/python test_gpt_image_integration.py
```

Expected output:
```
✅ All configuration tests passed!
✓ GPT-Image-1 client created successfully
✓ Configuration supports GPT-Image-1 credentials
✓ Pipeline can be configured to use GPT-Image-1
✓ Method routing for cloth swap is in place
```

## Troubleshooting

### GPT-Image-1 Not Being Used

**Problem**: Pipeline still uses Azure Flux despite configuration

**Solution**: 
- Verify `ENABLE_AZURE_GPT_IMAGE=true` in `.env`
- Check logs for "🎨 Using GPT-Image-1 for cloth swap..."
- Ensure endpoint and API key are correctly set

### Endpoint Not Found (404)

**Problem**: `Azure GPT-Image-1 cloth swap endpoint not found`

**Solution**:
- Verify deployment name matches your Azure resource
- Check API version is supported
- Ensure cloth-swap endpoint exists for your deployment

### Image Quality Issues

**Problem**: Generated images have artifacts or poor quality

**Solution**:
- Try adjusting the prompt/negative_prompt
- Check input image quality and resolution
- Verify garment image has clear, visible clothing

## API Reference

### GPT-Image-1 Request Format

```json
{
  "model": "gpt-image-1",
  "person_image": "base64_encoded_person_image",
  "garment_image": "base64_encoded_garment_image",
  "prompt": "Description of desired outcome",
  "negative_prompt": "Things to avoid"
}
```

### GPT-Image-1 Response Format

```json
{
  "image": "base64_encoded_result",
  "status": "succeeded"
}
```

## Future Enhancements

Potential improvements for GPT-Image-1 integration:

1. **Parallel Comparison**
   - Generate both GPT-Image-1 and Flux outputs
   - Side-by-side quality comparison

2. **Hybrid Approach**
   - Use GPT-Image-1 for initial swap
   - Apply Flux for refinement

3. **Quality Metrics**
   - Automatic evaluation of results
   - SSIM/PSNR comparisons

4. **Batch Processing**
   - Support multiple garments per person
   - Matrix testing across models

## References

- Azure OpenAI GPT-Image documentation
- Outfit Transfer Pipeline: `docs/ORIENTATION_DETECTION_USAGE.md`
- Prompt Engineering: `docs/PROMPT_SIMPLIFICATION_SUMMARY.md`
