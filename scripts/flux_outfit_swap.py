from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageOps

from azure_bfl_compare.clients.azure_flux import AzureFluxClient
from azure_bfl_compare.config import AzureFluxConfig


try:  # Pillow 10 renamed the filter enum
    _LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - legacy Pillow
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class CombinedAssets:
    """Paths and sizing data for the combined person + clothes panel."""

    combined_image_path: Path
    combined_mask_path: Path
    combined_size: Tuple[int, int]
    person_region: Tuple[int, int, int, int]
    original_size: Tuple[int, int]


@dataclass(frozen=True, slots=True)
class AzureGPTConfig:
    """Minimal settings for Azure GPT-4o-mini chat completions."""

    endpoint: str
    api_key: str
    deployment: str
    api_version: str


def _as_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a combined person + clothing image panel and submit it to the Azure Flux API "
            "for outfit swapping."
        )
    )
    parser.add_argument("--original", required=True, type=_as_path, help="Path to the person/original image.")
    parser.add_argument("--mask", required=True, type=_as_path, help="Binary mask image for the person.")
    parser.add_argument("--clothes", required=True, type=_as_path, help="Reference clothing image.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional manual prompt. If omitted, Azure GPT-4o-mini generates one automatically.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional manual negative prompt (auto-generated when not provided).",
    )
    parser.add_argument("--strength", type=float, default=0.85, help="Inpainting strength (0-1).")
    parser.add_argument("--task-name", default="manual-outfit", help="Identifier for saved artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional destination directory (defaults to output/manual_outfit/<task-name>).",
    )
    parser.add_argument(
        "--clothes-scale",
        type=float,
        default=0.6,
        help="Relative height of the clothing panel compared to the person image (default: 0.6).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=48,
        help="Horizontal padding (in pixels) inserted between the person and clothing panels.",
    )
    parser.add_argument(
        "--background-color",
        default="#FFFFFF",
        help="Background color for the combined canvas (hex RGB, default: #FFFFFF).",
    )
    parser.add_argument(
        "--skip-flux",
        action="store_true",
        help="Prepare the assets but do not submit them to Azure Flux.",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=None,
        help="Optional path to a .env file containing AZURE_FLUX_* credentials.",
    )
    parser.add_argument(
        "--skip-gpt",
        action="store_true",
        help="Disable automatic prompt generation via GPT-4o-mini.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="Maximum dimension for the combined canvas after letterboxing (default: 2048).",
    )
    return parser.parse_args()


def _load_flux_config() -> AzureFluxConfig:
    endpoint = os.getenv("AZURE_FLUX_ENDPOINT")
    api_key = os.getenv("AZURE_FLUX_API_KEY")
    deployment = os.getenv("AZURE_FLUX_DEPLOYMENT")
    api_version = os.getenv("AZURE_FLUX_API_VERSION", "2024-12-01-preview")

    missing = [name for name, value in [
        ("AZURE_FLUX_ENDPOINT", endpoint),
        ("AZURE_FLUX_API_KEY", api_key),
        ("AZURE_FLUX_DEPLOYMENT", deployment),
    ] if not value]

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Azure Flux configuration is incomplete. Missing environment variables: {joined}"
        )

    return AzureFluxConfig(
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        api_version=api_version,
    )


def _load_gpt_config() -> AzureGPTConfig | None:
    endpoint = os.getenv("AZURE_GPT_ENDPOINT")
    api_key = os.getenv("AZURE_GPT_API_KEY")
    deployment = os.getenv("AZURE_GPT_DEPLOYMENT")
    api_version = os.getenv("AZURE_GPT_API_VERSION", "2024-08-01-preview")

    if not all([endpoint, api_key, deployment]):
        return None

    return AzureGPTConfig(
        endpoint=str(endpoint),
        api_key=str(api_key),
        deployment=str(deployment),
        api_version=str(api_version),
    )


def _apply_background(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) not in {3, 6}:
        raise ValueError(f"Invalid background color '{color}'. Expected #RGB or #RRGGBB.")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i : i + 2], 16) for i in range(0, 6, 2))  # type: ignore[return-value]


def _determine_clothes_mask(image: Image.Image) -> Image.Image:
    """Create an opaque mask for the resized clothing image."""
    if image.mode == "RGBA":
        alpha = image.split()[-1]
        return alpha.point(lambda p: 255 if p > 0 else 0)
    grayscale = ImageOps.grayscale(image)
    # Use a modest threshold: anything darker than near-white becomes editable.
    return grayscale.point(lambda p: 255 if p < 250 else 0)


def create_combined_assets(
    original_path: Path,
    mask_path: Path,
    clothes_path: Path,
    output_dir: Path,
    *,
    clothes_scale: float,
    padding: int,
    background_color: str,
    max_size: int,
) -> CombinedAssets:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(original_path) as original_img:
        person_rgb = original_img.convert("RGB")
    with Image.open(mask_path) as mask_img:
        person_mask = mask_img.convert("L")

    if person_mask.size != person_rgb.size:
        person_mask = person_mask.resize(person_rgb.size, Image.NEAREST)

    with Image.open(clothes_path) as clothes_img:
        clothes_rgba = clothes_img.convert("RGBA")

    person_offset_x = 0
    person_offset_y = 0
    person_width = person_rgb.width
    person_height = person_rgb.height

    clothes_scale = max(0.05, clothes_scale)
    target_height = max(1, int(round(person_rgb.height * clothes_scale)))
    if target_height <= 0:
        raise ValueError("Clothes scale resulted in zero height. Increase --clothes-scale.")
    scale_factor = target_height / clothes_rgba.height
    resized_width = max(1, int(round(clothes_rgba.width * scale_factor)))
    clothes_resized = clothes_rgba.resize((resized_width, target_height), _LANCZOS)

    canvas_height = max(person_rgb.height, clothes_resized.height)
    canvas_width = person_rgb.width + padding + clothes_resized.width

    background = Image.new("RGB", (canvas_width, canvas_height), _apply_background(background_color))
    background.paste(person_rgb, (0, 0))

    clothes_offset_x = person_rgb.width + padding
    clothes_offset_y = max(0, (canvas_height - clothes_resized.height) // 2)
    background.paste(clothes_resized.convert("RGB"), (clothes_offset_x, clothes_offset_y), clothes_resized)

    combined_mask = Image.new("L", (canvas_width, canvas_height), 0)
    combined_mask.paste(person_mask, (0, 0))
    clothes_mask = _determine_clothes_mask(clothes_resized)
    combined_mask.paste(clothes_mask, (clothes_offset_x, clothes_offset_y))

    square_size = max(canvas_width, canvas_height)
    if square_size != canvas_width or square_size != canvas_height:
        square_rgb = Image.new("RGB", (square_size, square_size), _apply_background(background_color))
        offset_x = (square_size - canvas_width) // 2
        offset_y = (square_size - canvas_height) // 2
        square_rgb.paste(background, (offset_x, offset_y))
        background = square_rgb
        person_offset_x += offset_x
        person_offset_y += offset_y

        square_mask = Image.new("L", (square_size, square_size), 0)
        square_mask.paste(combined_mask, (offset_x, offset_y))
        combined_mask = square_mask

    max_size = max(256, max_size)
    if max(background.size) > max_size:
        scale = max_size / float(max(background.size))
        new_size = (
            max(1, int(round(background.width * scale))),
            max(1, int(round(background.height * scale))),
        )
        background = background.resize(new_size, _LANCZOS)
        combined_mask = combined_mask.resize(new_size, Image.NEAREST)
        person_offset_x = int(round(person_offset_x * scale))
        person_offset_y = int(round(person_offset_y * scale))
        person_width = max(1, int(round(person_width * scale)))
        person_height = max(1, int(round(person_height * scale)))

    combined_image_path = output_dir / "combined_input.png"
    combined_mask_path = output_dir / "combined_mask.png"
    background.save(combined_image_path)
    combined_mask.save(combined_mask_path)

    combined_width, combined_height = background.size
    clip_x = max(0, min(person_offset_x, combined_width - 1))
    clip_y = max(0, min(person_offset_y, combined_height - 1))
    right = min(clip_x + person_width, combined_width)
    bottom = min(clip_y + person_height, combined_height)
    crop_width = max(1, right - clip_x)
    crop_height = max(1, bottom - clip_y)

    return CombinedAssets(
        combined_image_path=combined_image_path,
        combined_mask_path=combined_mask_path,
        combined_size=(combined_width, combined_height),
        person_region=(clip_x, clip_y, crop_width, crop_height),
        original_size=(person_rgb.width, person_rgb.height),
    )


def build_flux_payload(
    assets: CombinedAssets,
    *,
    prompt: str,
    negative_prompt: str | None,
    strength: float,
) -> dict[str, object]:
    combined_image_bytes = assets.combined_image_path.read_bytes()
    combined_mask_bytes = assets.combined_mask_path.read_bytes()

    payload: dict[str, object] = {
        "prompt": prompt,
        "image": base64.b64encode(combined_image_bytes).decode("utf-8"),
        "mask": base64.b64encode(combined_mask_bytes).decode("utf-8"),
        "strength": max(0.0, min(1.0, strength)),
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload


def _encode_image_to_data_url(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        with BytesIO() as buffer:
            rgb.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _gpt_prompt_messages(
    *,
    person_data_url: str,
    clothes_data_url: str,
) -> list[dict[str, object]]:
    instructions = (
        "You are a precise fashion prompt engineer. "
        "Study the person photo and the clothing reference image. "
        "Return a JSON object with exactly two keys: prompt and negative_prompt. "
        "The prompt must only describe replacing the clothing on the person with the garments exactly as they appear in "
        "the reference image (colors, materials, silhouette, notable details) while preserving the person's identity, "
        "pose, lighting, background, and every element not covered by the reference garment. "
        "Do not mention or suggest any additional garments, accessories, alternate colors, or styling advice. "
        "Explicitly instruct the model to keep all other visual aspects unchanged. "
        "The negative_prompt should focus on avoiding structural distortions, background changes, artifacts, or any "
        "alteration beyond the targeted garment. Respond with valid JSON only."
    )
    return [
        {"role": "system", "content": "You craft meticulous prompts for diffusion-based outfit swaps."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {"type": "text", "text": "Person reference image:"},
                {"type": "image_url", "image_url": {"url": person_data_url}},
                {"type": "text", "text": "Clothing reference image:"},
                {"type": "image_url", "image_url": {"url": clothes_data_url}},
            ],
        },
    ]


def generate_prompts_via_gpt(
    *,
    gpt_config: AzureGPTConfig,
    person_path: Path,
    clothes_path: Path,
) -> tuple[str, str | None]:
    person_data = _encode_image_to_data_url(person_path)
    clothes_data = _encode_image_to_data_url(clothes_path)
    messages = _gpt_prompt_messages(person_data_url=person_data, clothes_data_url=clothes_data)

    url = (
        f"{gpt_config.endpoint}/openai/deployments/{gpt_config.deployment}/chat/completions"
        f"?api-version={gpt_config.api_version}"
    )

    payload: Dict[str, object] = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            url,
            headers={"api-key": gpt_config.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GPT-4o-mini returned non-JSON content: {content}") from exc

    prompt = str(parsed.get("prompt") or "").strip()
    negative_prompt = parsed.get("negative_prompt")
    if not prompt:
        raise RuntimeError("GPT-4o-mini response did not include a 'prompt' value.")
    if isinstance(negative_prompt, str) and not negative_prompt.strip():
        negative_prompt = None
    return prompt, negative_prompt if isinstance(negative_prompt, str) else None


def _fallback_prompts() -> tuple[str, str]:
    prompt = (
        "High-end fashion editorial photo of the same person wearing the provided reference outfit exactly. "
        "Keep identity, pose, lighting, camera angle, and environment identical. "
        "Match the garment's colors, materials, and silhouette precisely with natural draping and shadows while leaving all other elements untouched."
    )
    negative = (
        "blurry, text, logo, watermark, extra limbs, distorted body, different background, different face, cropped head"
    )
    return prompt, negative


def main() -> None:
    args = parse_args()

    if args.dotenv:
        load_dotenv(args.dotenv, override=True)
    else:
        default_env = Path(".env")
        if default_env.exists():
            load_dotenv(default_env, override=True)

    output_dir = args.output_dir or Path("output") / "manual_outfit" / args.task_name
    output_dir = output_dir.expanduser()

    assets = create_combined_assets(
        args.original,
        args.mask,
        args.clothes,
        output_dir,
        clothes_scale=args.clothes_scale,
        padding=args.padding,
        background_color=args.background_color,
        max_size=args.max_size,
    )

    print("✓ Combined panel prepared.")
    print(f"  Image: {assets.combined_image_path}")
    print(f"  Mask : {assets.combined_mask_path}")
    print(f"  Size : {assets.combined_size[0]}x{assets.combined_size[1]}")

    prompt = args.prompt
    negative_prompt = args.negative_prompt

    prompt_source = "manual"
    if not prompt:
        prompt_source = "fallback"
        if args.skip_gpt:
            prompt, fallback_negative = _fallback_prompts()
            if negative_prompt is None:
                negative_prompt = fallback_negative
        else:
            gpt_config = _load_gpt_config()
            if gpt_config is None:
                print("⚠️  Azure GPT environment variables missing – using fallback prompt.")
                prompt, fallback_negative = _fallback_prompts()
                if negative_prompt is None:
                    negative_prompt = fallback_negative
            else:
                try:
                    prompt, inferred_negative = generate_prompts_via_gpt(
                        gpt_config=gpt_config,
                        person_path=args.original,
                        clothes_path=args.clothes,
                    )
                    if negative_prompt is None:
                        negative_prompt = inferred_negative
                    prompt_source = "azure_gpt_4o_mini"
                    print("✓ Prompt generated via Azure GPT-4o-mini.")
                except Exception as exc:
                    print(f"⚠️  GPT prompt generation failed ({exc}); using fallback prompt.")
                    prompt, fallback_negative = _fallback_prompts()
                    if negative_prompt is None:
                        negative_prompt = fallback_negative

    prompts_path = output_dir / "prompts.txt"
    prompts_path.write_text(
        f"Prompt (source={prompt_source}):\n{prompt}\n\nNegative Prompt:\n{negative_prompt or '<none>'}\n"
    )
    print(f"  Prompts saved to {prompts_path}")

    payload = build_flux_payload(
        assets,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=args.strength,
    )

    preview = {
        key: ("<omitted base64>" if key in {"image", "mask"} else value)
        for key, value in payload.items()
    }
    preview_path = output_dir / "payload_preview.json"
    preview_path.write_text(json.dumps(preview, indent=2))
    print(f"  Preview payload written to {preview_path}")

    if args.skip_flux:
        print("ℹ️  Skipping Azure Flux submission (--skip-flux enabled).")
        return

    config = _load_flux_config()
    with AzureFluxClient(config) as client:
        print("🚀 Submitting request to Azure Flux…")
        result = client.generate(args.task_name, payload)

    raw_panel_path = output_dir / "output_flux_panel.png"
    raw_panel_path.write_bytes(result.image_bytes)

    with Image.open(BytesIO(result.image_bytes)) as panel_image:
        panel_rgb = panel_image.convert("RGB")
        if panel_rgb.size != assets.combined_size:
            panel_rgb = panel_rgb.resize(assets.combined_size, _LANCZOS)

        with BytesIO() as buffer:
            panel_rgb.save(buffer, format="PNG")
            cropped_bytes = buffer.getvalue()

    output_image_path = output_dir / "output_flux.png"
    output_image_path.write_bytes(cropped_bytes)
    print(f"✓ Flux panel saved to {output_image_path}")
    print(f"  Raw panel preserved at {raw_panel_path}")

    metadata_path = output_dir / "flux_metadata.json"
    metadata = dict(result.metadata)
    metadata["prompt_source"] = prompt_source
    metadata["prompt"] = prompt
    metadata["negative_prompt"] = negative_prompt
    metadata["combined_size"] = {"width": assets.combined_size[0], "height": assets.combined_size[1]}
    metadata["person_region"] = {
        "x": assets.person_region[0],
        "y": assets.person_region[1],
        "width": assets.person_region[2],
        "height": assets.person_region[3],
    }
    metadata["original_size"] = {"width": assets.original_size[0], "height": assets.original_size[1]}
    metadata["raw_panel"] = str(raw_panel_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Metadata stored at {metadata_path}")


if __name__ == "__main__":
    main()
