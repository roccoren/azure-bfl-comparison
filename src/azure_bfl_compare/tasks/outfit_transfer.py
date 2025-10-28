from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNXRUNTIME = False


try:  # Pillow 10 renamed resampling filters.
    LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - legacy Pillow
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


@dataclass(slots=True)
class OutfitPreparationResult:
    """Artifacts produced for a single outfit transfer task."""

    prompt: str
    negative_prompt: str
    strength: float
    combined_image_path: Path
    combined_mask_path: Path
    original_size: Tuple[int, int]
    combined_canvas_size: Tuple[int, int]
    letterbox_offset: Tuple[int, int]
    analysis_path: Path
    prompts_path: Path
    clothes_mask_path: Path
    clothes_alpha_path: Path
    clothing_info: Dict[str, object]
    clothes_image_path: Path
    original_image_path: Path
    target_mask_path: Path
    original_mask_path: Path | None

    def crop_output(self, image_bytes: bytes) -> bytes:
        """Crop the provider output back to the original frame."""
        with Image.open(io.BytesIO(image_bytes)) as image:  # type: ignore[name-defined]
            canvas_width, canvas_height = self.combined_canvas_size

            if image.size == self.original_size:
                output_image = image.copy()
            else:
                if image.size != (canvas_width, canvas_height):
                    resized = image.resize((canvas_width, canvas_height), resample=LANCZOS)
                else:
                    resized = image

                offset_x, offset_y = self.letterbox_offset
                left = offset_x
                upper = offset_y
                right = min(left + self.original_size[0], canvas_width)
                lower = min(upper + self.original_size[1], canvas_height)

                cropped = resized.crop((left, upper, right, lower))
                if cropped.size != self.original_size:
                    cropped = cropped.resize(self.original_size, resample=LANCZOS)
                output_image = cropped

            with io.BytesIO() as buffer:  # type: ignore[name-defined]
                output_image.save(buffer, format="PNG")
                return buffer.getvalue()

    def compose_flux_output(self, image_bytes: bytes) -> tuple[bytes, bytes]:
        """
        Composite a Flux output onto the original frame using the prepared mask.

        Returns
        -------
        tuple[bytes, bytes]
            (composited_image_bytes, raw_flux_bytes)
        """
        raw_flux_bytes = image_bytes
        cropped_bytes = self.crop_output(image_bytes)

        with Image.open(io.BytesIO(cropped_bytes)) as flux_image:  # type: ignore[name-defined]
            flux_rgba = flux_image.convert("RGBA")
        with Image.open(self.original_image_path) as original_image:
            original_rgba = original_image.convert("RGBA")
        with Image.open(self.target_mask_path) as mask_image:
            mask_l = mask_image.convert("L")

        if flux_rgba.size != original_rgba.size:
            flux_rgba = flux_rgba.resize(original_rgba.size, LANCZOS)
        if mask_l.size != original_rgba.size:
            mask_l = mask_l.resize(original_rgba.size, LANCZOS)

        preserve_flux_only = (
            os.getenv("FLUX_PRESERVE_MODEL_EDITS", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if not preserve_flux_only:
            flux_array = np.asarray(flux_rgba, dtype=np.uint8).copy()
            original_array = np.asarray(original_rgba, dtype=np.uint8)
            mask_array = np.asarray(mask_l, dtype=np.uint8)

            white_regions = (
                (flux_array[..., 0] > 245)
                & (flux_array[..., 1] > 245)
                & (flux_array[..., 2] > 245)
                & (mask_array > 0)
            )

            remaining_white = white_regions.copy()
            if remaining_white.any():
                try:
                    with Image.open(self.clothes_image_path) as clothes_img:
                        clothes_rgba_src = clothes_img.convert("RGBA")
                    with Image.open(self.clothes_mask_path) as clothes_mask_img:
                        clothes_mask = clothes_mask_img.convert("L")
                except Exception:
                    clothes_rgba_src = None
                    clothes_mask = None

                if clothes_rgba_src is not None and clothes_mask is not None:
                    if clothes_rgba_src.size != clothes_mask.size:
                        clothes_mask = clothes_mask.resize(clothes_rgba_src.size, LANCZOS)
                    clothes_rgba_src = clothes_rgba_src.copy()
                    clothes_rgba_src.putalpha(clothes_mask)

                    overlay_canvas = Image.new("RGBA", original_rgba.size, (0, 0, 0, 0))
                    coords = np.argwhere(mask_array > 0)
                    if coords.size > 0:
                        top = int(coords[:, 0].min())
                        bottom = int(coords[:, 0].max()) + 1
                        left = int(coords[:, 1].min())
                        right = int(coords[:, 1].max()) + 1
                        target_width = max(1, right - left)
                        target_height = max(1, bottom - top)

                        resized_clothes = clothes_rgba_src.resize((target_width, target_height), LANCZOS)
                        overlay_canvas.paste(resized_clothes, (left, top), resized_clothes)

                        overlay_array = np.asarray(overlay_canvas, dtype=np.uint8)
                        overlay_alpha = overlay_array[..., 3]
                        overlay_mask = overlay_alpha > 0
                        fill_mask = remaining_white & overlay_mask
                        if fill_mask.any():
                            flux_array[..., :3][fill_mask] = overlay_array[..., :3][fill_mask]
                            flux_array[..., 3][fill_mask] = overlay_alpha[fill_mask]
                            remaining_white = remaining_white & ~fill_mask

            if remaining_white.any():
                flux_array[..., :3][remaining_white] = original_array[..., :3][remaining_white]
                flux_array[..., 3][remaining_white] = original_array[..., 3][remaining_white]

            flux_rgba = Image.fromarray(flux_array, mode="RGBA")

        composite = original_rgba.copy()
        refined_mask = mask_l.filter(ImageFilter.MinFilter(3))
        refined_mask = refined_mask.filter(ImageFilter.GaussianBlur(radius=2))
        composite.paste(flux_rgba, mask=refined_mask)

        with io.BytesIO() as buffer:  # type: ignore[name-defined]
            composite.convert("RGB").save(buffer, format="PNG")
            final_bytes = buffer.getvalue()

        return final_bytes, raw_flux_bytes

    def metadata(self) -> Dict[str, object]:
        """Expose auxiliary paths and analysis data for metadata storage."""
        data: Dict[str, object] = {
            "analysis_file": str(self.analysis_path),
            "prompts_file": str(self.prompts_path),
            "combined_image": str(self.combined_image_path),
            "combined_mask": str(self.combined_mask_path),
            "combined_canvas_size": {
                "width": self.combined_canvas_size[0],
                "height": self.combined_canvas_size[1],
            },
            "letterbox_offset": {
                "x": self.letterbox_offset[0],
                "y": self.letterbox_offset[1],
            },
            "clothes_mask": str(self.clothes_mask_path),
            "clothes_alpha": str(self.clothes_alpha_path),
            "clothing_info": self.clothing_info,
            "original_size": {"width": self.original_size[0], "height": self.original_size[1]},
            "original_image": str(self.original_image_path),
            "clothes_image": str(self.clothes_image_path),
            "target_mask": str(self.target_mask_path),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "strength": self.strength,
        }
        if self.original_mask_path:
            data["original_mask"] = str(self.original_mask_path)
        return data

    def as_flux_payload(self, *, combined: bool = True) -> Dict[str, object]:
        """Build a ready-to-send Azure Flux payload with base64 encoded assets."""
        def encode(path: Path) -> str:
            return base64.b64encode(path.read_bytes()).decode("utf-8")

        if combined:
            image_source = self.combined_image_path
            mask_source = self.combined_mask_path
        else:
            image_source = self.original_image_path
            mask_source = self.target_mask_path

        payload: Dict[str, object] = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image": encode(image_source),
            "mask": encode(mask_source),
            "strength": round(self.strength, 4),
        }

        if not combined and self.original_mask_path:
            payload["original_mask"] = encode(self.original_mask_path)

        return payload

@dataclass(slots=True)
class Step0Inputs:
    """Artifacts loaded during Step 0 (input ingestion)."""

    original_rgb: Image.Image
    clothes_rgba: Image.Image
    original_path: Path
    clothes_path: Path
    original_size: Tuple[int, int]


@dataclass(slots=True)
class ClothesSegmentationArtifacts:
    """Artifacts persisted during Step 1 (clothes segmentation)."""

    mask_path: Path
    alpha_path: Path


@dataclass(slots=True)
class ClothingAnalysisResult:
    """Structured holder for Step 2 analysis output."""

    info: Dict[str, object]


@dataclass(slots=True)
class MaskGenerationArtifacts:
    """Artifacts produced during Step 3 (target mask creation)."""

    mask_array: np.ndarray
    mask_path: Path
    source: str


@dataclass(slots=True)
class PromptArtifacts:
    """Prompt bundle generated during Step 4."""

    prompt: str
    negative_prompt: str
    strength: float

class EnhancedOutfitTransferPipeline:
    """CPU-only outfit transfer pipeline with placeholder ML stages."""

    _COLOR_SWATCHES: Dict[str, Tuple[int, int, int]] = {
        "black": (16, 16, 16),
        "white": (240, 240, 240),
        "warm gray": (176, 170, 165),
        "cool gray": (160, 170, 180),
        "navy blue": (25, 54, 112),
        "royal blue": (45, 86, 163),
        "sky blue": (110, 160, 220),
        "teal": (30, 140, 140),
        "emerald green": (0, 120, 80),
        "olive": (110, 120, 55),
        "maroon": (120, 28, 45),
        "burgundy": (138, 21, 56),
        "crimson": (184, 20, 38),
        "sunset orange": (220, 90, 40),
        "mustard": (198, 158, 34),
        "lavender": (170, 140, 210),
        "violet": (110, 60, 160),
        "blush pink": (225, 180, 190),
        "champagne": (235, 215, 175),
    }

    def __init__(
        self,
        *,
        azure_gpt_endpoint: Optional[str] = None,
        azure_gpt_key: Optional[str] = None,
        azure_gpt_deployment: Optional[str] = None,
        azure_gpt_api_version: Optional[str] = None,
        azure_gpt_image_endpoint: Optional[str] = None,
        azure_gpt_image_key: Optional[str] = None,
        azure_gpt_image_deployment: Optional[str] = None,
        azure_gpt_image_api_version: Optional[str] = None,
        use_gpt_image: bool = False,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_h",
        device: Optional[str] = None,
        azure_flux_endpoint: Optional[str] = None,
        azure_flux_key: Optional[str] = None,
        azure_flux_deployment: Optional[str] = None,
        azure_flux_api_version: Optional[str] = None,
        deeplab_model_path: Optional[str] = None,
    ) -> None:
        self.azure_gpt_endpoint = (azure_gpt_endpoint or os.getenv("AZURE_GPT_ENDPOINT", "")).rstrip("/")
        self.azure_gpt_key = azure_gpt_key or os.getenv("AZURE_GPT_API_KEY")
        self.azure_gpt_deployment = azure_gpt_deployment or os.getenv("AZURE_GPT_DEPLOYMENT", "gpt-4o-mini")
        self.azure_gpt_api_version = azure_gpt_api_version or os.getenv("AZURE_GPT_API_VERSION", "2024-02-15-preview")

        flux_endpoint_env = (
            azure_flux_endpoint
            or os.getenv("AZURE_FLUX_ENDPOINT")
            or os.getenv("AZURE_FLUX_KONTEXT_ENDPOINT")
            or ""
        )
        self.azure_flux_endpoint = flux_endpoint_env.rstrip("/")
        self.azure_flux_key = (
            azure_flux_key
            or os.getenv("AZURE_FLUX_API_KEY")
            or os.getenv("AZURE_FLUX_KONTEXT_KEY")
        )
        self.azure_flux_deployment = azure_flux_deployment or os.getenv("AZURE_FLUX_DEPLOYMENT")
        self.azure_flux_api_version = azure_flux_api_version or os.getenv("AZURE_FLUX_API_VERSION", "2024-12-01-preview")

        # GPT-Image-1 configuration for cloth swap
        self.use_gpt_image = use_gpt_image or os.getenv("ENABLE_AZURE_GPT_IMAGE", "false").lower() == "true"
        self.azure_gpt_image_endpoint = (azure_gpt_image_endpoint or os.getenv("AZURE_GPT_IMAGE_ENDPOINT", "")).rstrip("/")
        self.azure_gpt_image_key = azure_gpt_image_key or os.getenv("AZURE_GPT_IMAGE_API_KEY")
        self.azure_gpt_image_deployment = azure_gpt_image_deployment or os.getenv("AZURE_GPT_IMAGE_DEPLOYMENT", "gpt-image-1")
        self.azure_gpt_image_api_version = azure_gpt_image_api_version or os.getenv("AZURE_GPT_IMAGE_API_VERSION", "2024-12-01-preview")

        resolved_model_path: Optional[Path] = None
        env_model_path = os.getenv("DEEPLAB_ONNX_PATH")
        if deeplab_model_path:
            resolved_model_path = Path(deeplab_model_path)
        elif env_model_path:
            resolved_model_path = Path(env_model_path)
        self.deeplab_model_path = resolved_model_path
        self._deeplab_session: Optional["ort.InferenceSession"] = None  # type: ignore[name-defined]
        self._deeplab_input_name: Optional[str] = None
        self._deeplab_output_name: Optional[str] = None

        # Retain the legacy constructor parameters so existing callers continue to work.
        self._legacy_sam_checkpoint = sam_checkpoint
        self._legacy_sam_model_type = sam_model_type
        self._legacy_device = device or "cpu"

    # ------------------------------------------------------------------
    # Orientation detection
    # ------------------------------------------------------------------
    def detect_person_orientation(self, original_rgb: Image.Image) -> Dict[str, str]:
        """
        Detect person's orientation using GPT-4o-mini Vision API.
        
        Returns:
            {
                "orientation": "front-facing" | "back-facing" | "left-profile" | "right-profile" | "three-quarter-left" | "three-quarter-right",
                "confidence": "high" | "medium" | "low",
                "description": "detailed description of visible features"
            }
        """
        print("🧭 Step 0: Detecting person orientation via GPT-4o-mini...")
        
        if not self.azure_gpt_endpoint or not self.azure_gpt_key:
            print("   ⚠️ Azure GPT credentials missing – skipping orientation detection.")
            return {
                "orientation": None,
                "confidence": None,
                "description": "Orientation detection skipped (no credentials)",
            }
        
        buffered = io.BytesIO()
        original_rgb.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        url = (
            f"{self.azure_gpt_endpoint}/openai/deployments/{self.azure_gpt_deployment}/chat/completions"
            f"?api-version={self.azure_gpt_api_version}"
        )
        
        instructions = (
            "Analyze this image of a person and determine their viewing orientation/angle. "
            "Focus on which direction the person is facing.\n\n"
            "Classify into ONE of these orientations:\n"
            "1. front-facing: Person's face is clearly visible, looking toward camera\n"
            "2. back-facing: Person's back is visible, face not shown at all\n"
            "3. left-profile: Person's left side of face visible (facing left)\n"
            "4. right-profile: Person's right side of face visible (facing right)\n"
            "5. three-quarter-left: Mostly face visible but angled left\n"
            "6. three-quarter-right: Mostly face visible but angled right\n\n"
            "Provide confidence level: high (very clear), medium (somewhat clear), low (ambiguous)\n\n"
            "Response format (exactly):\n"
            "ORIENTATION: <orientation>\n"
            "CONFIDENCE: <confidence>\n"
            "DESCRIPTION: <what you see - describe visible body parts, face visibility, etc.>"
        )
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing human pose and orientation in images.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                },
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }
        
        try:
            response = requests.post(
                url,
                headers={"api-key": self.azure_gpt_key, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = self._parse_orientation_response(content)
            
            if parsed is None:
                raise ValueError("GPT response did not match expected format.")
            
            print(f"   ✓ Detected orientation: {parsed['orientation']} (confidence: {parsed['confidence']})")
            if len(parsed['description']) > 80:
                preview = parsed['description'][:80] + "..."
            else:
                preview = parsed['description']
            print(f"   ✓ Description: {preview}")
            
            return parsed
            
        except Exception as exc:
            print(f"   ⚠️ Orientation detection failed ({exc}); using generic orientation preservation.")
            return {
                "orientation": None,
                "confidence": None,
                "description": f"Detection failed: {exc}",
            }
    
    @staticmethod
    def _parse_orientation_response(content: str) -> Optional[Dict[str, str]]:
        """Parse GPT orientation detection response."""
        result = {
            "orientation": "",
            "confidence": "",
            "description": "",
        }
        
        valid_orientations = {
            "front-facing", "back-facing", "left-profile",
            "right-profile", "three-quarter-left", "three-quarter-right"
        }
        valid_confidence = {"high", "medium", "low"}
        
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("ORIENTATION:"):
                value = line.split(":", 1)[1].strip().lower()
                # Normalize variations
                if "back" in value:
                    result["orientation"] = "back-facing"
                elif "front" in value:
                    result["orientation"] = "front-facing"
                elif "left" in value and "three" in value:
                    result["orientation"] = "three-quarter-left"
                elif "right" in value and "three" in value:
                    result["orientation"] = "three-quarter-right"
                elif "left" in value:
                    result["orientation"] = "left-profile"
                elif "right" in value:
                    result["orientation"] = "right-profile"
                else:
                    # Try exact match
                    if value in valid_orientations:
                        result["orientation"] = value
            elif line.startswith("CONFIDENCE:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in valid_confidence:
                    result["confidence"] = value
            elif line.startswith("DESCRIPTION:"):
                result["description"] = line.split(":", 1)[1].strip()
        
        if not result["orientation"] or not result["description"]:
            return None
        
        if not result["confidence"]:
            result["confidence"] = "medium"
        
        return result
    
    def _get_orientation_instruction(self, orientation: Optional[str], confidence: Optional[str]) -> str:
        """Generate orientation-specific instruction for prompt."""
        if not orientation:
            return "Keep the same viewing angle and body orientation."
        
        orientation_map = {
            "front-facing": "Keep front-facing view (face visible).",
            "back-facing": "Keep back view (face NOT visible).",
            "left-profile": "Keep left profile view.",
            "right-profile": "Keep right profile view.",
            "three-quarter-left": "Keep three-quarter left view.",
            "three-quarter-right": "Keep three-quarter right view.",
        }
        
        return orientation_map.get(orientation, "Keep the same viewing angle.")

    # ------------------------------------------------------------------
    # Public orchestration
    # ------------------------------------------------------------------
    def prepare(
        self,
        *,
        task_name: str,
        original_image: str,
        clothes_image: str,
        output_dir: Path,
        strength_override: Optional[float] = None,
        original_mask_path: Optional[str] = None,
    ) -> OutfitPreparationResult:
        """Compute prompts, masks, and intermediate assets for a task."""
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'=' * 60}\n🧵 Preparing outfit transfer task: {task_name}\n{'=' * 60}")
 
        inputs = self._load_step0_inputs(original_image=original_image, clothes_image=clothes_image)
        
        # Step 0: Detect person orientation
        orientation_info = self.detect_person_orientation(inputs.original_rgb)
 
        segmentation = self._run_step1_clothes_segmentation(inputs, output_dir)
        analysis = self._run_step2_clothing_analysis(inputs)
        mask_artifacts = self._run_step3_mask_generation(
            inputs,
            analysis.info,
            segmentation,
            output_dir,
            original_mask_path,
        )
        prompt_artifacts = self._run_step4_prompt(
            analysis.info,
            strength_override,
            orientation_info["orientation"],
            orientation_info["confidence"],
        )
 
        combined_image_path = output_dir / "combined_input.png"
        combined_mask_path = output_dir / "combined_mask.png"
        canvas_size, letterbox_offset = self._save_combined_panel(
            inputs.original_rgb,
            segmentation.alpha_path,
            mask_artifacts.mask_array,
            combined_image_path,
            combined_mask_path,
        )
 
        analysis_path = output_dir / "analysis.json"
        analysis_payload = {
            "task_name": task_name,
            "input": {
                "original_image": str(inputs.original_path),
                "clothes_image": str(inputs.clothes_path),
            },
            "orientation": orientation_info,
            "clothing": analysis.info,
            "steps": {
                "step_0": {
                    "orientation": orientation_info["orientation"],
                    "confidence": orientation_info["confidence"],
                    "description": orientation_info["description"],
                },
                "step_1": {
                    "clothes_mask": str(segmentation.mask_path),
                    "clothes_alpha": str(segmentation.alpha_path),
                },
                "step_2": {
                    "description": analysis.info["description"],
                    "type": analysis.info["clothing_type"],
                },
                "step_3": {
                    "target_mask": str(mask_artifacts.mask_path),
                    "mask_generator": mask_artifacts.source,
                },
                "step_4": {
                    "prompt": prompt_artifacts.prompt,
                    "negative_prompt": prompt_artifacts.negative_prompt,
                },
                "step_5": {"api_ready_payload": True},
            },
        }
        analysis_path.write_text(json.dumps(analysis_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Analysis summary saved to {analysis_path}")
 
        prompts_path = output_dir / "prompts.txt"
        prompts_path.write_text(
            f"Positive Prompt:\n{prompt_artifacts.prompt}\n\nNegative Prompt:\n{prompt_artifacts.negative_prompt}\n",
            encoding="utf-8",
        )
        print(f"✓ Prompt snapshot saved to {prompts_path}")
 
        return OutfitPreparationResult(
            prompt=prompt_artifacts.prompt,
            negative_prompt=prompt_artifacts.negative_prompt,
            strength=prompt_artifacts.strength,
            combined_image_path=combined_image_path,
            combined_mask_path=combined_mask_path,
            original_size=inputs.original_size,
            combined_canvas_size=canvas_size,
            letterbox_offset=letterbox_offset,
            analysis_path=analysis_path,
            prompts_path=prompts_path,
            clothes_mask_path=segmentation.mask_path,
            clothes_alpha_path=segmentation.alpha_path,
            clothing_info=analysis.info,
            clothes_image_path=inputs.clothes_path,
            original_image_path=inputs.original_path,
            target_mask_path=mask_artifacts.mask_path,
            original_mask_path=Path(original_mask_path) if original_mask_path else None,
        )

    def _load_step0_inputs(self, *, original_image: str, clothes_image: str) -> Step0Inputs:
        """Step 0: Load and validate input assets."""
        original_path = Path(original_image)
        clothes_path = Path(clothes_image)
        self._validate_input(original_path, "original image")
        self._validate_input(clothes_path, "clothing image")

        with Image.open(original_path) as original_im:
            original_rgb = original_im.convert("RGB")
        with Image.open(clothes_path) as clothes_im:
            clothes_rgba = clothes_im.convert("RGBA")

        return Step0Inputs(
            original_rgb=original_rgb,
            clothes_rgba=clothes_rgba,
            original_path=original_path,
            clothes_path=clothes_path,
            original_size=original_rgb.size,
        )

    def _run_step1_clothes_segmentation(
        self,
        inputs: Step0Inputs,
        output_dir: Path,
    ) -> ClothesSegmentationArtifacts:
        """Step 1: Segment the clothing reference image."""
        mask_path = output_dir / "clothes_mask.png"
        alpha_path = output_dir / "clothes_alpha.png"
        self._segment_clothing(inputs.clothes_rgba, mask_path, alpha_path)
        return ClothesSegmentationArtifacts(mask_path=mask_path, alpha_path=alpha_path)

    def _run_step2_clothing_analysis(self, inputs: Step0Inputs) -> ClothingAnalysisResult:
        """Step 2: Analyze the clothing reference."""
        info = self._analyze_clothing(inputs.clothes_rgba)
        return ClothingAnalysisResult(info=info)

    def _run_step3_mask_generation(
        self,
        inputs: Step0Inputs,
        clothing_info: Dict[str, object],
        segmentation: ClothesSegmentationArtifacts,
        output_dir: Path,
        original_mask_path: Optional[str],
    ) -> MaskGenerationArtifacts:
        """Step 3: Produce or normalize the target inpainting mask."""
        mask_path = output_dir / "target_mask.png"

        base_type = str(clothing_info.get("clothing_type", "upper"))
        if base_type == "shoes":
            clothing_slots: Tuple[str, ...] = ("shoes",)
        else:
            derived_slots = self._derive_clothing_slots(segmentation.mask_path)
            if base_type == "lower":
                clothing_slots = ("lower",)
            elif base_type == "upper":
                clothing_slots = ("upper", "lower") if "lower" in derived_slots else ("upper",)
            else:
                clothing_slots = tuple(slot for slot in derived_slots if slot in {"upper", "lower"}) or (base_type,)
        clothing_info["clothing_slots"] = list(clothing_slots)

        parsing_map = self._run_deeplab_segmentation(inputs.original_rgb)
        body_slots = self._body_slots_from_parsing(parsing_map, inputs.original_rgb.size)
        clothing_info["body_slots"] = body_slots

        target_slots: List[str] = []
        for slot in clothing_slots:
            if slot == "upper" and body_slots.get("upper", True):
                target_slots.append("upper")
            elif slot in {"lower", "shoes"} and body_slots.get("lower", False):
                target_slots.append(slot)
        if not target_slots:
            fallback_slot = "upper" if body_slots.get("upper", True) else "lower"
            target_slots = [fallback_slot]
        clothing_info["target_slots"] = target_slots

        print(f"   ℹ️ Clothing coverage detected: {', '.join(clothing_slots)}")
        print(
            "   ℹ️ Person body coverage detected: "
            f"upper={body_slots.get('upper', False)} lower={body_slots.get('lower', False)}"
        )
        print(f"   🎯 Targeting slots for replacement: {', '.join(target_slots)}")

        if original_mask_path:
            mask_array = self.prepare_existing_mask(inputs.original_size, Path(original_mask_path), mask_path)
            source = "provided_mask"
        else:
            mask_array, source = self._generate_target_mask(
                inputs.original_rgb,
                tuple(target_slots),
                parsing_map=parsing_map,
            )
            Image.fromarray(mask_array, mode="L").save(mask_path)
        clothing_info["mask_source"] = source
        print(f"✓ Target mask saved to {mask_path} ({source})")
        return MaskGenerationArtifacts(mask_array=mask_array, mask_path=mask_path, source=source)

    def _derive_clothing_slots(self, mask_path: Path) -> Tuple[str, ...]:
        """Infer which body slots the clothing reference likely covers."""
        try:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        except FileNotFoundError:
            return ("upper",)

        total_pixels = mask.size
        if mask.size == 0 or int(mask.sum()) == 0:
            return ("upper",)

        coverage_ratio = float((mask > 127).sum()) / float(total_pixels)
        if coverage_ratio >= 0.85:
            # Segmentation likely captured the full canvas (white background, noisy mask).
            # Treat as upper-only to avoid targeting lower body unintentionally.
            return ("upper",)

        height, width = mask.shape
        binary = mask > 127

        rows = np.where(binary.any(axis=1))[0]
        if rows.size == 0:
            return ("upper",)

        top = rows[0]
        bottom = rows[-1]
        coverage_height = (bottom - top + 1) / float(height)

        slots: list[str] = []
        if top <= int(height * 0.55):
            slots.append("upper")

        if (
            coverage_height >= 0.7
            and bottom >= int(height * 0.9)
            and coverage_ratio >= 0.25
        ):
            slots.append("lower")

        if not slots:
            slots = ["upper"]

        # Preserve ordering and uniqueness
        seen: set[str] = set()
        ordered_slots = []
        for slot in slots:
            if slot not in seen:
                ordered_slots.append(slot)
                seen.add(slot)
        return tuple(ordered_slots)

    def _body_slots_from_parsing(
        self,
        parsing_map: Optional[np.ndarray],
        size: Tuple[int, int],
    ) -> Dict[str, bool]:
        """Estimate whether upper/lower body regions are visible in the person parsing map."""
        if parsing_map is None:
            return {"upper": True, "lower": True}

        height, width = parsing_map.shape
        if height != size[1] or width != size[0]:
            # Ensure alignment with the original resolution
            parsing_resized = Image.fromarray(parsing_map, mode="L").resize(size, LANCZOS)
            parsing_map = np.asarray(parsing_resized, dtype=np.uint8)
            height, width = parsing_map.shape

        person_mask = parsing_map == 15  # PASCAL VOC person class index
        if not person_mask.any():
            return {"upper": True, "lower": True}

        upper_region = person_mask[: int(height * 0.55), :]
        lower_region = person_mask[int(height * 0.45) :, :]

        upper_present = bool(upper_region.any())
        lower_present = bool(lower_region.any())

        return {"upper": upper_present, "lower": lower_present}

    def _run_step4_prompt(
        self,
        clothing_info: Dict[str, object],
        strength_override: Optional[float],
        orientation: Optional[str] = None,
        orientation_confidence: Optional[str] = None,
    ) -> PromptArtifacts:
        """Step 4: Build prompts and compute inpainting strength."""
        prompt, negative_prompt = self._build_flux_prompt(
            clothing_info,
            orientation,
            orientation_confidence,
        )
        if strength_override is not None:
            strength = float(strength_override)
            print(f"   🎯 Using manual strength override: {strength:.2f}")
        else:
            strength = self._estimate_strength(clothing_info)
        return PromptArtifacts(prompt=prompt, negative_prompt=negative_prompt, strength=strength)

    def call_azure_gpt_image_api(
        self,
        preparation: OutfitPreparationResult,
        *,
        execute: bool = False,
    ) -> Dict[str, object]:
        """Call Azure GPT-Image-1 cloth swap API.
        
        This method uses GPT-Image-1 for direct cloth swap without masks.
        """
        if not execute:
            print("⚠️  Azure GPT-Image-1 API execution skipped (placeholder).")
            return {
                "status": "skipped",
                "reason": "Placeholder execution. Enable execute=True to submit the request.",
            }

        if not self.azure_gpt_image_endpoint or not self.azure_gpt_image_key:
            raise RuntimeError("Azure GPT-Image-1 credentials are not configured for execution.")

        combined_image = preparation.combined_image_path.read_bytes()
        combined_mask = preparation.combined_mask_path.read_bytes()

        url = (
            f"{self.azure_gpt_image_endpoint.rstrip('/')}"
            f"/openai/deployments/{self.azure_gpt_image_deployment}/images/edits"
        )
        params = {"api-version": self.azure_gpt_image_api_version}

        gpt_prompt, gpt_negative_prompt = self._build_gpt_image_prompt_bundle(preparation)
        if gpt_negative_prompt:
            gpt_prompt = f"{gpt_prompt} Avoid: {gpt_negative_prompt}."

        def _detect_mime(name: str, data: bytes) -> tuple[str, str]:
            if name.endswith(".png") or data.startswith(b"\x89PNG\r\n\x1a\n"):
                return name if name.endswith(".png") else f"{name}.png", "image/png"
            if name.endswith(".jpg") or name.endswith(".jpeg") or data[:3] == b"\xff\xd8\xff":
                base = name if name.endswith(".jpg") or name.endswith(".jpeg") else f"{name}.jpg"
                return base, "image/jpeg"
            if name.endswith(".bmp") or data[:2] == b"BM":
                return name if name.endswith(".bmp") else f"{name}.bmp", "image/bmp"
            if name.endswith(".gif") or data.startswith(b"GIF8"):
                return name if name.endswith(".gif") else f"{name}.gif", "image/gif"
            return f"{name}.bin", "application/octet-stream"

        image_name = preparation.combined_image_path.name
        image_filename, image_mime = _detect_mime(image_name, combined_image)

        with Image.open(io.BytesIO(combined_mask)) as mask_image:
            mask_gray = mask_image.convert("L")
        mask_array = np.asarray(mask_gray, dtype=np.uint8)
        alpha_array = np.where(mask_array > 127, 0, 255).astype(np.uint8)
        mask_rgba = Image.new("RGBA", mask_gray.size, (255, 255, 255, 255))
        mask_rgba.putalpha(Image.fromarray(alpha_array, mode="L"))
        mask_buffer = io.BytesIO()
        mask_rgba.save(mask_buffer, format="PNG")
        mask_bytes = mask_buffer.getvalue()
        mask_buffer.close()
        mask_filename, mask_mime = "mask.png", "image/png"

        headers = {"api-key": self.azure_gpt_image_key, "Accept": "application/json"}
        files = {
            "image": (image_filename, combined_image, image_mime),
            "mask": (mask_filename, mask_bytes, mask_mime),
        }
        data = {"prompt": gpt_prompt}

        print(f"🚀 Calling Azure GPT-Image-1 edits endpoint at {url}")
        response = requests.post(url, headers=headers, params=params, data=data, files=files, timeout=120)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = f" Response: {exc.response.json()}"
            except Exception:
                if exc.response is not None:
                    detail = f" Body: {exc.response.text}"
            raise RuntimeError(f"Azure GPT-Image-1 request failed ({exc.response.status_code}).{detail}") from exc

        result_json = response.json()
        operation_url = response.headers.get("Operation-Location") or response.headers.get("Azure-AsyncOperation")
        if operation_url:
            if not operation_url.lower().startswith("http"):
                operation_url = f"{self.azure_gpt_image_endpoint.rstrip('/')}/{operation_url.lstrip('/')}"

            poll_interval = float(os.getenv("AZURE_GPT_IMAGE_POLL_INTERVAL", "2.0"))
            max_attempts = int(os.getenv("AZURE_GPT_IMAGE_MAX_POLL_ATTEMPTS", "60"))
            poll_headers = {"api-key": self.azure_gpt_image_key, "Accept": "application/json"}
            poll_params = None if "api-version=" in operation_url else {"api-version": self.azure_gpt_image_api_version}

            for attempt in range(max_attempts):
                poll_response = requests.get(
                    operation_url,
                    headers=poll_headers,
                    params=poll_params,
                    timeout=120,
                )
                poll_response.raise_for_status()
                result_json = poll_response.json()
                status = str(result_json.get("status", "")).lower()
                if status in {"succeeded", "success"}:
                    break
                if status in {"failed", "cancelled", "canceled"}:
                    raise RuntimeError(f"Azure GPT-Image-1 operation failed with status '{status}': {result_json}")
                time.sleep(poll_interval)
            else:
                raise TimeoutError(
                    f"Azure GPT-Image-1 operation did not complete after {max_attempts} attempts."
                )

        def _extract_base64(payload: Dict[str, object]) -> tuple[str | None, Dict[str, object]]:
            candidates = []
            if isinstance(payload.get("data"), list) and payload["data"]:
                first = payload["data"][0]
                if isinstance(first, dict):
                    candidates.append(first)
            if isinstance(payload.get("result"), dict):
                inner = payload["result"]
                if isinstance(inner.get("data"), list) and inner["data"]:
                    entry = inner["data"][0]
                    if isinstance(entry, dict):
                        candidates.append(entry)
            if isinstance(payload.get("images"), list) and payload["images"]:
                entry = payload["images"][0]
                if isinstance(entry, dict):
                    candidates.append(entry)
            if isinstance(payload.get("result"), dict):
                image_value = payload["result"].get("image")
                if isinstance(image_value, (str, bytes)):
                    candidates.append({"image": image_value})
            flat_candidates = candidates + [payload]

            for candidate in flat_candidates:
                if not isinstance(candidate, dict):
                    continue
                if "b64_json" in candidate and candidate["b64_json"]:
                    return candidate["b64_json"], candidate
                if "image" in candidate and candidate["image"]:
                    return candidate["image"], candidate
                if "image_base64" in candidate and candidate["image_base64"]:
                    return candidate["image_base64"], candidate
            return None, payload

        image_b64, _source = _extract_base64(result_json)
        if not image_b64:
            raise RuntimeError(f"Azure GPT-Image-1 response missing image data: {result_json}")

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode GPT-Image-1 output: {exc}") from exc

        def _redact_images(node: object) -> object:
            if isinstance(node, dict):
                cleaned: Dict[str, object] = {}
                for key, value in node.items():
                    if key in {"b64_json", "image", "image_base64"} and isinstance(value, (str, bytes)):
                        cleaned[key] = "<omitted>"
                    else:
                        cleaned[key] = _redact_images(value)
                return cleaned
            if isinstance(node, list):
                return [_redact_images(item) for item in node]
            return node

        sanitized_response = _redact_images(result_json)

        output_dir = preparation.target_mask_path.parent
        output_path = output_dir / "output_gpt_image.png"
        output_path.write_bytes(image_bytes)
        print(f"✓ GPT-Image-1 output saved to {output_path}")

        return {
            "status": result_json.get("status", "succeeded"),
            "saved_files": {"output_image": str(output_path)},
            "response": sanitized_response,
        }

    def call_azure_flux_api(
        self,
        preparation: OutfitPreparationResult,
        *,
        execute: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """Placeholder helper for Step 5 to call the Azure Flux Kontext API.

        By default the request is NOT executed. Pass execute=True once credentials
        are configured to run the real HTTP call. The returned dictionary mirrors
        a minimal subset of the expected REST response and records the saved file paths.
        
        If use_gpt_image is enabled, this will call GPT-Image-1 instead.
        """
        # Use GPT-Image-1 if enabled
        if self.use_gpt_image:
            print("🎨 Using GPT-Image-1 for cloth swap...")
            return self.call_azure_gpt_image_api(preparation, execute=execute)

        # Otherwise use Azure Flux
        payload = preparation.as_flux_payload()
        if seed is not None:
            payload["seed"] = seed

        if not execute:
            payload_preview = {
                key: ("<omitted base64>" if key in {"image", "mask", "original_mask"} else value)
                for key, value in payload.items()
            }
            print("⚠️  Azure Flux API execution skipped (placeholder).")
            return {
                "status": "skipped",
                "reason": "Placeholder execution. Enable execute=True to submit the request.",
                "payload_preview": payload_preview,
            }

        if not self.azure_flux_endpoint or not self.azure_flux_key:
            raise RuntimeError("Azure Flux credentials are not configured for execution.")

        endpoint = self.azure_flux_endpoint.rstrip("/")
        params: Dict[str, str] = {}
        if "openai/deployments" in endpoint and "/images/" in endpoint:
            url = endpoint
            if "api-version=" not in endpoint and self.azure_flux_api_version:
                params["api-version"] = self.azure_flux_api_version
        else:
            if not self.azure_flux_deployment:
                raise RuntimeError(
                    "Azure Flux deployment name is required when AZURE_FLUX_ENDPOINT does not include it."
                )
            url = (
                f"{endpoint}/openai/deployments/{self.azure_flux_deployment}/images/generations"
            )
            if self.azure_flux_api_version:
                params["api-version"] = self.azure_flux_api_version

        headers = {
            "api-key": self.azure_flux_key,
            "Content-Type": "application/json",
        }
        if self.azure_flux_deployment:
            payload.setdefault("model", self.azure_flux_deployment)
        print(f"🚀 Calling Azure Flux Kontext API at {url}")
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            params=params or None,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        operation_url = (
            response.headers.get("Operation-Location")
            or response.headers.get("operation-location")
            or response.headers.get("Azure-AsyncOperation")
        )
        if operation_url:
            if not operation_url.lower().startswith("http"):
                operation_url = f"{self.azure_flux_endpoint.rstrip('/')}/{operation_url.lstrip('/')}"
            poll_headers = {"api-key": self.azure_flux_key, "Accept": "application/json"}
            poll_params = None if "api-version=" in operation_url else params or {"api-version": self.azure_flux_api_version}
            poll_interval = float(os.getenv("AZURE_FLUX_POLL_INTERVAL", "2.0"))
            max_attempts = int(os.getenv("AZURE_FLUX_MAX_POLL_ATTEMPTS", "60"))
            for attempt in range(max_attempts):
                poll_response = requests.get(
                    operation_url,
                    headers=poll_headers,
                    params=poll_params,
                    timeout=120,
                )
                poll_response.raise_for_status()
                result = poll_response.json()
                status = str(result.get("status", "")).lower()
                if status in {"succeeded", "success"}:
                    break
                if status in {"failed", "cancelled", "canceled"}:
                    raise RuntimeError(f"Azure Flux operation failed with status '{status}': {result}")
                time.sleep(poll_interval)
            else:
                raise TimeoutError("Azure Flux operation did not complete within the allotted attempts.")

        def _extract_flux_base64(payload: Dict[str, object]) -> tuple[str | None, Dict[str, object]]:
            candidates = []
            if isinstance(payload.get("data"), list) and payload["data"]:
                first = payload["data"][0]
                if isinstance(first, dict):
                    candidates.append(first)
            if isinstance(payload.get("result"), dict):
                inner = payload["result"]
                if isinstance(inner.get("data"), list) and inner["data"]:
                    entry = inner["data"][0]
                    if isinstance(entry, dict):
                        candidates.append(entry)
            if isinstance(payload.get("images"), list) and payload["images"]:
                entry = payload["images"][0]
                if isinstance(entry, dict):
                    candidates.append(entry)
            if "image_base64" in payload and payload["image_base64"]:
                candidates.append({"image_base64": payload["image_base64"]})
            flat_candidates = candidates + [payload]
            for candidate in flat_candidates:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("b64_json"):
                    return candidate["b64_json"], candidate
                if candidate.get("image_base64"):
                    return candidate["image_base64"], candidate
                if candidate.get("image"):
                    return candidate["image"], candidate
            return None, payload

        image_b64, _source = _extract_flux_base64(result)
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as exc:
                raise RuntimeError(f"Failed to decode Azure Flux image data: {exc}") from exc
            output_dir = preparation.target_mask_path.parent
            final_bytes, raw_bytes = preparation.compose_flux_output(image_bytes)
            raw_path = output_dir / "output_flux_raw.png"
            raw_path.write_bytes(raw_bytes)

            output_path = output_dir / "output_flux.png"
            output_path.write_bytes(final_bytes)

            saved_files = result.setdefault("saved_files", {})
            saved_files["output_image"] = str(output_path)
            saved_files["output_image_raw"] = str(raw_path)
            print(f"✓ Output image saved to {output_path}")
        else:
            print("⚠️  Azure Flux response did not include image data.")

        return result

    # ------------------------------------------------------------------
    # Placeholder implementations for each pipeline step
    # ------------------------------------------------------------------
    def _validate_input(self, path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label.capitalize()} not found: {path}")

    def _segment_clothing(self, clothes_rgba: Image.Image, mask_path: Path, alpha_path: Path) -> None:
        print("✂️  Step 1: Segmenting clothing and creating masked version...")

        clothes_rgb = clothes_rgba.convert("RGB")
        grayscale = clothes_rgba.convert("L")
        values = np.asarray(grayscale, dtype=np.uint8)
        threshold = self._otsu_threshold(values)

        mask: np.ndarray
        grabcut_used = False

        if HAS_CV2:
            try:
                image_bgr = cv2.cvtColor(np.asarray(clothes_rgb), cv2.COLOR_RGB2BGR)
                height, width = values.shape

                initial_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
                probable_foreground = values < max(250, int(threshold * 1.05))
                initial_mask[probable_foreground] = cv2.GC_PR_FGD

                border = max(4, min(height, width) // 25)
                initial_mask[:border, :] = cv2.GC_BGD
                initial_mask[-border:, :] = cv2.GC_BGD
                initial_mask[:, :border] = cv2.GC_BGD
                initial_mask[:, -border:] = cv2.GC_BGD

                fg_rows = np.where(probable_foreground.any(axis=1))[0]
                fg_cols = np.where(probable_foreground.any(axis=0))[0]
                if fg_rows.size > 0 and fg_cols.size > 0:
                    top, bottom = fg_rows[0], fg_rows[-1]
                    left, right = fg_cols[0], fg_cols[-1]
                    rect = (max(0, left - border), max(0, top - border),
                            min(width - 1, right + border) - max(0, left - border),
                            min(height - 1, bottom + border) - max(0, top - border))
                else:
                    margin_x = width // 10
                    margin_y = height // 10
                    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)

                cv2.grabCut(image_bgr, initial_mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
                cv2.grabCut(image_bgr, initial_mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)

                mask = np.where((initial_mask == cv2.GC_FGD) | (initial_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                if mask.sum() > 0:
                    grabcut_used = True
                else:
                    raise ValueError("GrabCut produced empty mask")
            except Exception as exc:  # pragma: no cover - runtime fallback
                grabcut_used = False
                print(f"   ⚠️ GrabCut segmentation failed ({exc}); falling back to Otsu threshold.")

        if grabcut_used:
            print("   ✓ GrabCut segmentation applied for clothing reference.")
        if not HAS_CV2 or not grabcut_used:
            mask = np.where(values > threshold, 255, 0).astype(np.uint8)
            if mask.sum() == 0:
                mask.fill(255)
            if not HAS_CV2:
                print("   ⚠️ OpenCV not available – using Otsu threshold segmentation.")
            else:
                print("   ✓ Otsu threshold segmentation used as fallback.")

        Image.fromarray(mask, mode="L").save(mask_path)

        clothes_array = np.asarray(clothes_rgb, dtype=np.uint8)
        mask_3channel = np.stack([mask, mask, mask], axis=-1) / 255.0
        masked_clothes = (clothes_array * mask_3channel + 255 * (1 - mask_3channel)).astype(np.uint8)
        
        clothes_masked_rgb = Image.fromarray(masked_clothes, mode="RGB")
        clothes_masked_rgb.save(alpha_path, format="PNG")
        
        print(f"   ✓ Clothes mask saved to {mask_path}")
        print(f"   ✓ Clothes with white background (pure RGB, no color loss) saved to {alpha_path}")

    def _analyze_clothing(self, clothes_rgba: Image.Image) -> Dict[str, str]:
        print("🔍 Step 2: Clothing analysis via Azure GPT-4o-mini...")
        if not self.azure_gpt_endpoint or not self.azure_gpt_key:
            print("   ⚠️ Azure GPT credentials missing – using heuristic fallback.")
            return self._heuristic_clothing_analysis(clothes_rgba, log=True)

        heuristics = self._heuristic_clothing_analysis(clothes_rgba, log=False)

        buffered = io.BytesIO()
        clothes_rgba.convert("RGB").save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = (
            f"{self.azure_gpt_endpoint}/openai/deployments/{self.azure_gpt_deployment}/chat/completions"
            f"?api-version={self.azure_gpt_api_version}"
        )

        instructions = (
            "Analyze this clothing reference image in precise detail and respond exactly in the required format.\n\n"
            "Provide:\n"
            "1. CLOTHING_TYPE: one of [upper, lower, shoes]\n"
            "2. DESCRIPTION: vivid paragraph covering colors, patterns, texture, fabric behaviour\n"
            "3. FINE_DETAILS: bullet-style sentence capturing trims, embellishments, closures, silhouettes\n"
            "4. STYLE: choose from [casual, formal, athletic, traditional, vintage, modern, bohemian]\n\n"
            "Response format (exactly):\n"
            "CLOTHING_TYPE: <type>\n"
            "DESCRIPTION: <detailed description>\n"
            "FINE_DETAILS: <concise but comprehensive fine details>\n"
            "STYLE: <style>"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert fashion analyst and prompt engineer.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                },
            ],
            "max_tokens": 500,
            "temperature": 0.15,
        }

        try:
            response = requests.post(
                url,
                headers={"api-key": self.azure_gpt_key, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = self._parse_gpt_clothing_response(content)
            if parsed is None:
                raise ValueError("GPT response did not match the expected format.")
        except Exception as exc:
            print(f"   ⚠️ Azure GPT analysis failed ({exc}); using heuristic fallback.")
            return self._heuristic_clothing_analysis(clothes_rgba, log=True)

        info = dict(heuristics)
        info.update(
            clothing_type=parsed["clothing_type"],
            description=parsed["description"],
            fine_details=parsed["fine_details"],
            style=parsed["style"] or heuristics["style"],
            analysis_source="azure_gpt_4o_mini",
        )

        print(f"   ✓ GPT detected clothing type: {info['clothing_type']}")
        preview = info["description"][:80]
        suffix = "..." if len(info["description"]) > 80 else ""
        print(f"   ✓ GPT description: {preview}{suffix}")
        return info

    def _heuristic_clothing_analysis(self, clothes_rgba: Image.Image, *, log: bool) -> Dict[str, str]:
        if log:
            print("   ℹ️  Running heuristic color/shape fallback analysis...")
        rgb_image = clothes_rgba.convert("RGB")
        small = rgb_image.resize((64, 64), LANCZOS)
        sample = np.asarray(small, dtype=np.float32)

        average_rgb = sample.reshape(-1, 3).mean(axis=0)
        dominant_color = self._describe_color(tuple(average_rgb))
        saturation = self._approximate_saturation(sample)
        brightness = float(np.clip(sample.mean(), 0, 255))

        clothing_type = self._guess_clothing_type(rgb_image.size)
        style = "formal" if saturation < 22.0 else "casual"

        garment_label = {
            "upper": "upper garment",
            "lower": "lower garment",
            "shoes": "pair of shoes",
        }[clothing_type]

        description = (
            f"a {dominant_color} {style} {garment_label} with a smooth placeholder texture and natural sheen"
        )
        fine_details = (
            "Fallback estimate of trims and embellishments derived from image statistics; "
            "replace with model-generated insights when available."
        )

        info = {
            "clothing_type": clothing_type,
            "description": description,
            "fine_details": fine_details,
            "style": style,
            "base_color": dominant_color,
            "average_brightness": round(brightness, 2),
            "average_saturation": round(saturation, 2),
            "analysis_source": "heuristic_placeholder",
        }

        if log:
            print(f"   ✓ Estimated clothing type: {clothing_type}")
            print(f"   ✓ Dominant color estimate: {dominant_color}")
        return info

    @staticmethod
    def _parse_gpt_clothing_response(content: str) -> Optional[Dict[str, str]]:
        result = {
            "clothing_type": "",
            "description": "",
            "fine_details": "",
            "style": "",
        }

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("CLOTHING_TYPE:"):
                value = line.split(":", 1)[1].strip().lower()
                if "shoe" in value or "boot" in value:
                    result["clothing_type"] = "shoes"
                elif "pant" in value or "trouser" in value or "skirt" in value:
                    result["clothing_type"] = "lower"
                else:
                    result["clothing_type"] = "upper"
            elif line.startswith("DESCRIPTION:"):
                result["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("FINE_DETAILS:"):
                result["fine_details"] = line.split(":", 1)[1].strip()
            elif line.startswith("STYLE:"):
                result["style"] = line.split(":", 1)[1].strip().lower()

        if not result["description"]:
            return None
        if not result["clothing_type"]:
            result["clothing_type"] = "upper"
        if result["style"] not in {"casual", "formal", "athletic", "traditional", "vintage", "modern", "bohemian"}:
            result["style"] = ""
        return result

    def _generate_target_mask(
        self,
        original_rgb: Image.Image,
        target_slots: Tuple[str, ...],
        *,
        parsing_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str]:
        if not target_slots:
            target_slots = ("upper",)

        if parsing_map is None:
            parsing_map = self._run_deeplab_segmentation(original_rgb)

        width, height = original_rgb.size
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        if parsing_map is not None:
            for slot in target_slots:
                region_mask = self._extract_region_mask(parsing_map, slot, original_rgb.size)
                if region_mask is not None:
                    combined_mask = np.maximum(combined_mask, region_mask)
            if combined_mask.sum() > 0:
                print("✓ DeepLab v3+ MobileNetV2 parsing applied for target mask.")
                return combined_mask, "deeplab_v3_plus_mobilenet_v2"
            print("   ⚠️ DeepLab parsing did not yield a valid region mask; falling back to heuristic hint.")
        else:
            print("   ⚠️ DeepLab v3+ MobileNetV2 unavailable; using heuristic region hint.")

        heuristic_mask = self._create_target_mask_for_slots(original_rgb.size, target_slots)
        return heuristic_mask, "heuristic_region_hint"

    def _load_deeplab_session(self) -> bool:
        if not HAS_ONNXRUNTIME:
            return False
        if self._deeplab_session is not None:
            return True
        if not self.deeplab_model_path or not self.deeplab_model_path.exists():
            return False
        try:
            self._deeplab_session = ort.InferenceSession(  # type: ignore[attr-defined]
                str(self.deeplab_model_path),
                providers=["CPUExecutionProvider"],
            )
            self._deeplab_input_name = self._deeplab_session.get_inputs()[0].name
            self._deeplab_output_name = self._deeplab_session.get_outputs()[0].name
            print(f"   ✓ DeepLab v3+ MobileNetV2 model loaded from {self.deeplab_model_path}")
            return True
        except Exception as exc:
            print(f"   ⚠️ Failed to load DeepLab v3+ MobileNetV2 model: {exc}")
            self._deeplab_session = None
            self._deeplab_input_name = None
            self._deeplab_output_name = None
            return False

    def _run_deeplab_segmentation(self, original_rgb: Image.Image) -> Optional[np.ndarray]:
        if not self._load_deeplab_session():
            return None
        assert self._deeplab_session is not None
        assert self._deeplab_input_name is not None
        assert self._deeplab_output_name is not None

        input_resolution = 513
        resized = original_rgb.resize((input_resolution, input_resolution), LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        array = array[None, ...]  # NHWC format expected by mobile DeepLab exports

        try:
            outputs = self._deeplab_session.run([self._deeplab_output_name], {self._deeplab_input_name: array})
        except Exception as exc:
            print(f"   ⚠️ DeepLab inference failed: {exc}")
            return None
        if not outputs:
            return None

        logits = outputs[0]
        # The model output is already ArgMax result with shape [1, H, W]
        if logits.ndim == 3 and logits.shape[0] == 1:
            parsing_map = logits[0].astype(np.uint8)
        # Handle case where output is logits [1, H, W, C] or [1, C, H, W]
        elif logits.ndim == 4 and logits.shape[0] == 1:
            if logits.shape[1] <= 32:
                parsing_map = logits[0].argmax(axis=0).astype(np.uint8)
            else:
                parsing_map = logits[0].argmax(axis=-1).astype(np.uint8)
        else:
            print(f"   ⚠️ Unexpected DeepLab output shape: {logits.shape}")
            return None
        
        parsing_image = Image.fromarray(parsing_map, mode="L").resize(original_rgb.size, LANCZOS)
        return np.asarray(parsing_image, dtype=np.uint8)

    def _extract_region_mask(
        self,
        parsing_map: np.ndarray,
        target_slot: str,
        size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        height, width = parsing_map.shape
        person_label = 15  # PASCAL VOC person class for DeepLab
        region = parsing_map == person_label
        coverage = float(region.sum()) / float(region.size)
        if coverage < 0.02:
            return None

        mask = np.zeros_like(parsing_map, dtype=np.uint8)
        mask[region] = 255
        
        # More precise masking: only the clothing coverage area, not entire torso
        if target_slot == "upper":
            # Upper clothing: chest and shoulders area, avoiding face/neck and arms
            # Vertical bounds: from slightly below shoulders (22%) to mid-hip (72%)
            mask[: int(height * 0.22), :] = 0
            mask[int(height * 0.72) :, :] = 0

            # Horizontal bounds: center torso area, reducing arm coverage
            # Keep center ~56% to focus on torso clothing
            left_margin = int(width * 0.22)
            right_margin = int(width * 0.22)
            mask[:, :left_margin] = 0  # Remove left arm region
            mask[:, width - right_margin:] = 0  # Remove right arm region

        elif target_slot == "lower":
            # Lower clothing: waist to ankles, avoiding upper body
            mask[: int(height * 0.40), :] = 0  # Remove upper body
            mask[int(height * 0.92) :, :] = 0  # Remove feet area
            
            # Keep center 70% to focus on pants/skirt
            left_margin = int(width * 0.15)
            right_margin = int(width * 0.15)
            mask[:, :left_margin] = 0
            mask[:, width - right_margin:] = 0
            
        elif target_slot == "shoes":
            # Shoes: only ankle and below
            mask[: int(height * 0.75), :] = 0  # Only bottom 25% for shoes

        if mask.sum() == 0:
            return None

        # Apply morphological operations to refine mask edges
        mask_image = Image.fromarray(mask, mode="L")
        
        # Erode slightly to avoid edges, then blur for smooth transitions
        from PIL import ImageFilter
        mask_image = mask_image.filter(ImageFilter.MinFilter(3))  # Erode
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=5))
        mask_image = mask_image.point(lambda value: 255 if value > 100 else 0)
        
        return np.asarray(mask_image, dtype=np.uint8)

    def prepare_existing_mask(
        self,
        original_size: Tuple[int, int],
        provided_mask_path: Path,
        output_path: Path,
    ) -> np.ndarray:
        print("🎯 Using provided mask for target region...")
        with Image.open(provided_mask_path) as mask_image:
            mask_gray = mask_image.convert("L")
            if mask_gray.size != original_size:
                mask_gray = mask_gray.resize(original_size, LANCZOS)
            mask_binary = mask_gray.point(lambda value: 255 if value > 127 else 0)
            mask_binary.save(output_path)
        print(f"   ✓ Normalised mask saved to {output_path}")
        return np.asarray(mask_binary, dtype=np.uint8)

    def _create_target_mask_for_slots(self, size: Tuple[int, int], target_slots: Sequence[str]) -> np.ndarray:
        width, height = size
        combined = np.zeros((height, width), dtype=np.uint8)
        for slot in target_slots:
            combined = np.maximum(combined, self._create_target_mask(size, slot))
        return combined

    def _create_target_mask(self, size: Tuple[int, int], clothing_type: str) -> np.ndarray:
        print(f"🎭 Step 3: Creating target mask for {clothing_type}...")
        width, height = size
        mask = Image.new("L", size, color=0)
        draw = ImageDraw.Draw(mask)

        if clothing_type == "upper":
            # More conservative upper mask: center torso only, avoiding face/neck and arms
            top_rect = (
                int(width * 0.28),
                int(height * 0.25),
                int(width * 0.72),
                int(height * 0.72),
            )
            draw.rounded_rectangle(top_rect, radius=int(min(width, height) * 0.08), fill=255)

        elif clothing_type == "lower":
            # More conservative lower mask: center legs area
            bottom_rect = (
                int(width * 0.25),  # More inward (was 0.18)
                int(height * 0.50),  # Higher start (was 0.45)
                int(width * 0.75),  # More inward (was 0.82)
                int(height * 0.88),  # Lower cutoff to avoid feet (was 0.95)
            )
            draw.rounded_rectangle(bottom_rect, radius=int(min(width, height) * 0.05), fill=255)
            
        elif clothing_type == "shoes":
            # Shoes: only bottom area
            shoe_rect = (
                int(width * 0.15),
                int(height * 0.82),  # Higher start (was 0.78)
                int(width * 0.85),
                int(height * 0.98),  # Avoid very bottom edge
            )
            draw.rounded_rectangle(shoe_rect, radius=int(min(width, height) * 0.03), fill=255)
        else:
            draw.rectangle((0, 0, width, height), fill=255)

        # Lighter blur for sharper edges, then erode slightly
        mask = mask.filter(ImageFilter.MinFilter(3))  # Erode to make mask smaller
        mask = mask.filter(ImageFilter.GaussianBlur(radius=4))  # Softer blur (was 6)
        mask = mask.point(lambda value: 255 if value > 120 else 0)  # Higher threshold (was 128)
        return np.asarray(mask, dtype=np.uint8)

    def _build_gpt_image_prompt_bundle(self, preparation: OutfitPreparationResult) -> Tuple[str, str]:
        """Compose GPT-Image-1 specific prompts that emphasise preservation of the source frame."""
        clothing_info = preparation.clothing_info or {}
        garment_description = str(clothing_info.get("description") or "").strip()
        fine_details = str(clothing_info.get("fine_details") or "").strip()

        if not garment_description:
            garment_description = preparation.prompt

        if fine_details:
            garment_description = f"{garment_description} Fine details to respect: {fine_details}."

        target_slots = clothing_info.get("target_slots") or []
        if isinstance(target_slots, (list, tuple)) and target_slots:
            slot_text = f" Limit edits strictly to the {', '.join(map(str, target_slots))} region."
        else:
            slot_text = " Limit edits strictly to the clothing area that is replaced."

        gpt_prompt = (
            "Edit the supplied person_image directly without generating a new scene."
            " Preserve the original subject's identity, face, skin tone, hair, body proportions, pose, hands, legs, and expression."
            " Maintain the original background, lighting, shadows, camera angle, framing, and image quality."
            " Only swap the clothing worn by the existing person with this garment description: "
            f"{garment_description}{slot_text}"
            " Ensure the garment follows the body contours naturally and blend seams cleanly."
            " Fill the entire masked area with the garment fabric so no blank, transparent, or white regions remain."
            " Do not add accessories unless required by the garment."
        )

        preservation_guards = (
            "different person, replaced subject, swapped identity, changed face, new hairstyle, skin tone change, body change, pose change, "
            "arm change, leg change, missing limbs, extra limbs, altered expression, background change, new environment, different camera angle, "
            "cropped composition, zoomed in, zoomed out, regenerated scene, displaced subject, floating clothes, garment disconnected, artifacts, text, watermark"
        )

        fill_guards = (
            "white patch, empty fabric, incomplete garment, transparent cloth, unfilled mask, missing garment coverage, blank clothing area"
        )

        negative_parts = [preservation_guards, fill_guards]
        if preparation.negative_prompt:
            negative_parts.append(preparation.negative_prompt)

        negative_prompt = ", ".join(part.strip(" ,") for part in negative_parts if part)

        return gpt_prompt, negative_prompt

    def _build_flux_prompt(
        self,
        clothing_info: Dict[str, object],
        orientation: Optional[str] = None,
        orientation_confidence: Optional[str] = None,
    ) -> Tuple[str, str]:
        print("📝 Step 4: Building Flux prompt...")
        description = str(clothing_info.get("description", ""))
        fine_details = str(clothing_info.get("fine_details", ""))
        target_slots = clothing_info.get("target_slots", [])
        slot_phrase = ""
        if target_slots:
            slot_phrase = f" Replace {', '.join(map(str, target_slots))} clothing only."
        
        # Get orientation-specific instruction
        orientation_instruction = self._get_orientation_instruction(orientation, orientation_confidence)

        # Precise inpainting prompt: focus on clothing replacement while preserving composition
        prompt = (
            f"Keep the exact same person, pose, and scene composition. "
            f"Only replace the clothing in the masked area with: {description}. "
            f"Details: {fine_details}. "
            f"The clothing should fit naturally on this specific person. "
            f"Completely fill the masked region with the garment so there are no blank, transparent, or white patches. "
            f"{orientation_instruction} "
            f"Maintain the original photo framing and show the full person. {slot_phrase}"
        )

        # Strong negative prompt emphasizing preservation and preventing zoom/crop
        negative_prompt = (
            "zoomed in, cropped, close-up of clothing only, showing only the garment, "
            "missing head, missing face, missing body parts, partial person, cut off person, "
            "changing the person, different person, removing person, no person visible, mannequin, "
            "face change, different face, facial modification, hair change, different hairstyle, hair color change, "
            "skin tone change, body shape change, pose change, different pose, "
            "background change, different background, environment change, "
            "turning around, rotation, flipped, different angle, different view, perspective change, "
            "body position change, stance change, arm position change, leg position change, "
            "extra limbs, missing limbs, deformed anatomy, distorted proportions, unnatural body, "
            "wrong gender, age change, child to adult, adult to child, "
            "changing non-masked areas, modifying unmasked regions, full body clothing change, "
            "reframing, different composition, altered framing, zooming, "
            "blurry, low quality, unrealistic, artificial, cartoon, illustration, "
            "white patch, empty fabric, incomplete garment, transparent cloth, unfilled mask, missing garment coverage, blank clothing area"
        )

        print("   ✓ Prompt prepared.")
        return prompt, negative_prompt

    def _estimate_strength(self, clothing_info: Dict[str, object]) -> float:
        print("⚖️  Estimating inpainting strength...")
        base_strength = 0.68
        adjustments = 0.0

        style = str(clothing_info.get("style", "")).lower()
        fine_details = str(clothing_info.get("fine_details", ""))
        average_saturation = float(clothing_info.get("average_saturation", 0.0))

        if style == "formal":
            adjustments += 0.05
        if "placeholder" not in fine_details.lower():
            adjustments += 0.04
        if average_saturation > 35.0:
            adjustments += 0.03

        strength = round(min(0.9, base_strength + adjustments), 4)
        print(f"   ✓ Strength set to {strength:.2f}")
        return strength

    def _save_combined_panel(
        self,
        original_rgb: Image.Image,
        clothes_alpha_path: Path,
        target_mask: np.ndarray,
        combined_image_path: Path,
        combined_mask_path: Path,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        print("🧩 Preparing combined panel with alpha-masked clothing for Azure Flux submission...")
        orig_width, orig_height = original_rgb.size

        with Image.open(clothes_alpha_path) as clothes_img:
            clothes_panel = clothes_img.convert("RGB")

        panel_width, panel_height = clothes_panel.size
        max_panel_width = int(orig_width * 0.45)
        max_panel_height = int(orig_height * 0.6)
        scale = min(
            max_panel_width / float(panel_width) if panel_width else 1.0,
            max_panel_height / float(panel_height) if panel_height else 1.0,
            1.0,
        )
        if scale < 1.0:
            new_size = (
                max(1, int(round(panel_width * scale))),
                max(1, int(round(panel_height * scale))),
            )
            clothes_panel = clothes_panel.resize(new_size, LANCZOS)
            panel_width, panel_height = clothes_panel.size

        lateral_gap = max(32, orig_width // 25)
        vertical_margin = max(32, orig_height // 30)

        left_margin = panel_width + lateral_gap
        canvas_width = left_margin + orig_width + lateral_gap + panel_width
        canvas_height = max(orig_height + 2 * vertical_margin, panel_height + 2 * vertical_margin)

        person_offset_x = left_margin
        person_offset_y = (canvas_height - orig_height) // 2
        patch_x = person_offset_x + orig_width + lateral_gap
        patch_y = (canvas_height - panel_height) // 2

        combined_rgb = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        combined_rgb.paste(original_rgb, (person_offset_x, person_offset_y))
        combined_rgb.paste(clothes_panel, (patch_x, patch_y))

        combined_mask = Image.new("L", (canvas_width, canvas_height), 0)
        combined_mask.paste(Image.fromarray(target_mask, mode="L"), (person_offset_x, person_offset_y))
        # Leave the clothing reference area unmasked so Azure focuses on the person region.

        square_size = max(canvas_width, canvas_height)
        offset_x = (square_size - canvas_width) // 2
        offset_y = (square_size - canvas_height) // 2
        if square_size != canvas_width or square_size != canvas_height:
            square_rgb = Image.new("RGB", (square_size, square_size), (255, 255, 255))
            square_rgb.paste(combined_rgb, (offset_x, offset_y))
            combined_rgb = square_rgb

            square_mask = Image.new("L", (square_size, square_size), 0)
            square_mask.paste(combined_mask, (offset_x, offset_y))
            combined_mask = square_mask
            print(
                f"   ✓ Letterboxed combined panel to square canvas {square_size}x{square_size} "
                "to stabilise Azure preprocessing."
            )

        total_offset = (offset_x + person_offset_x, offset_y + person_offset_y)

        max_canvas = int(os.getenv("OUTFIT_COMBINED_MAX_SIZE", "2048"))
        if max_canvas < 256:
            max_canvas = 256
        if max(combined_rgb.size) > max_canvas:
            scale = max_canvas / float(max(combined_rgb.size))
            new_size = (
                max(1, int(round(combined_rgb.width * scale))),
                max(1, int(round(combined_rgb.height * scale))),
            )
            combined_rgb = combined_rgb.resize(new_size, LANCZOS)
            combined_mask = combined_mask.resize(new_size, Image.NEAREST)
            total_offset = (
                int(round(total_offset[0] * scale)),
                int(round(total_offset[1] * scale)),
            )
            print(
                f"   ✓ Scaled combined panel to {new_size[0]}x{new_size[1]} "
                f"(max {max_canvas}) to satisfy Azure limits."
            )

        combined_rgb.save(combined_image_path)
        combined_mask.save(combined_mask_path)

        print(f"   ✓ Clothing reference inset at ({patch_x}, {patch_y}) with size {panel_width}x{panel_height}")
        print(f"   ✓ Combined image saved to {combined_image_path}")
        print(f"   ✓ Combined mask saved to {combined_mask_path}")

        return combined_rgb.size, total_offset

    # ------------------------------------------------------------------
    # Heuristic utilities
    # ------------------------------------------------------------------
    def _otsu_threshold(self, values: np.ndarray) -> int:
        hist, _ = np.histogram(values.flatten(), bins=256, range=(0, 256))
        total = values.size
        sum_total = np.dot(hist, np.arange(256))

        sum_background = 0.0
        weight_background = 0.0
        maximum_variance = 0.0
        threshold = 127

        for idx, weight in enumerate(hist):
            weight_background += weight
            if weight_background == 0:
                continue
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += idx * weight
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            if between_class_variance > maximum_variance:
                maximum_variance = between_class_variance
                threshold = idx

        return max(32, min(224, threshold))

    def _describe_color(self, rgb: Tuple[float, float, float]) -> str:
        distances = {
            name: np.linalg.norm(np.array(rgb) - np.array(value))
            for name, value in self._COLOR_SWATCHES.items()
        }
        return min(distances, key=distances.get)

    def _approximate_saturation(self, sample: np.ndarray) -> float:
        max_channel = sample.max(axis=2)
        min_channel = sample.min(axis=2)
        denom = np.where(max_channel == 0, 1.0, max_channel)
        saturation = (max_channel - min_channel) / denom
        return float(np.mean(saturation) * 100.0)

    def _guess_clothing_type(self, size: Tuple[int, int]) -> str:
        width, height = size
        if height >= width * 1.25:
            return "upper"
        if height <= width * 0.55:
            return "shoes"
        if height >= width * 0.9:
            return "lower"
        return "upper"
