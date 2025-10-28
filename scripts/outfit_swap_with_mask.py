from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from azure_bfl_compare.tasks.outfit_adapter import OutfitJob, OutfitSwapAdapter


def _positive_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single outfit swap using the enhanced pipeline with a provided inpainting mask.\n"
            "The script prepares prompts, normalizes the assets, and optionally calls Azure Flux."
        )
    )
    parser.add_argument("--original", required=True, type=_positive_path, help="Person/original image path.")
    parser.add_argument("--mask", required=True, type=_positive_path, help="Binary mask covering the swap region.")
    parser.add_argument("--clothes", required=True, type=_positive_path, help="Reference garment image path.")
    parser.add_argument("--task-name", default="manual-outfit", help="Identifier used for output folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for intermediate artifacts (defaults to output/manual_outfit/<task-name>).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Optional override for the computed inpainting strength.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Flux seed passed when executing the Azure request.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit the prepared payload to Azure Flux (requires configured credentials).",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=None,
        help="Optional path to a .env file containing Azure credentials and configuration.",
    )
    return parser.parse_args()


def _resolve_output_dir(task_name: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser()
    return Path("output") / "manual_outfit" / task_name


def main() -> None:
    args = parse_args()

    if args.dotenv:
        load_dotenv(args.dotenv, override=True)
    else:
        default_env = Path(".env")
        if default_env.exists():
            load_dotenv(default_env, override=True)

    output_dir = _resolve_output_dir(args.task_name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = OutfitSwapAdapter()
    job = OutfitJob(
        task_name=args.task_name,
        original_image=str(args.original),
        clothes_image=str(args.clothes),
        mask_image=str(args.mask),
        output_dir=output_dir,
        strength_override=args.strength,
    )

    try:
        preparation = adapter.prepare(job)
    except Exception as exc:  # pragma: no cover - CLI execution guard
        print(f"[!] Failed to prepare outfit swap pipeline: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("✓ Outfit swap assets prepared.")
    print(f"  Combined panel: {preparation.combined_image_path}")
    print(f"  Combined mask : {preparation.combined_mask_path}")
    print(f"  Analysis JSON : {preparation.analysis_path}")
    print(f"  Prompts file  : {preparation.prompts_path}")

    if not args.execute:
        print("ℹ️  Skipping Azure Flux execution (pass --execute to submit the request).")
        return

    try:
        if adapter.use_gpt_image or not adapter.use_flux:
            exec_result = adapter.execute_gpt_image(job.task_name)
        elif adapter.use_flux:
            exec_result = adapter.execute_flux(job.task_name, seed=args.seed)
        else:
            print("[!] Neither Azure Flux nor Azure GPT-Image-1 is enabled; nothing to execute.", file=sys.stderr)
            raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - network/runtime guard
        print(f"[!] Azure request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    saved_image = (exec_result.saved_files or {}).get("output_image")
    if saved_image:
        print(f"✓ Output saved to {saved_image}")
    else:
        print("⚠️  Response did not include an output image reference.")


if __name__ == "__main__":
    main()
