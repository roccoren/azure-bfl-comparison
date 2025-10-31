from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner that feeds multiple tasks from a JSON definition into "
            "scripts/flux_outfit_swap.py."
        )
    )
    parser.add_argument(
        "batch_file",
        type=Path,
        help="Path to the JSON array describing outfit swap jobs.",
    )
    parser.add_argument(
        "--flux-script",
        type=Path,
        default=Path(__file__).with_name("flux_outfit_swap.py"),
        help="Optional override for the flux outfit swap CLI.",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=None,
        help="Optional .env file forwarded to the flux CLI.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to process from the batch file.",
    )
    parser.add_argument(
        "--skip-gpt",
        action="store_true",
        help="Forward --skip-gpt so prompts are not auto-generated.",
    )
    parser.add_argument(
        "--skip-flux",
        action="store_true",
        help="Forward --skip-flux to prepare assets without submitting to Azure.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Override --max-size for the flux CLI (defaults to CLI default when omitted).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch on the first failure instead of continuing.",
    )
    return parser.parse_args()


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Failed to read batch file {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Batch file {path} contains invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Batch definition must be a JSON array, got {type(data).__name__}.")

    validated: list[dict[str, Any]] = []
    for idx, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"Entry #{idx} is not an object; found {type(entry).__name__}.")
        validated.append(entry)
    return validated


def _candidate_paths(base_dirs: Iterable[Path], relative: str) -> Iterable[Path]:
    candidate = Path(relative)
    if candidate.is_absolute():
        yield candidate
        return
    for directory in base_dirs:
        yield (directory / candidate).resolve()


def _resolve_asset(
    *,
    candidate: str | None,
    search_dirs: list[Path],
    label: str,
    task_name: str,
) -> Path:
    if not candidate:
        raise SystemExit(f"Task '{task_name}' is missing required field '{label}'.")

    attempts = list(_candidate_paths(search_dirs, candidate))
    for path in attempts:
        if path.exists():
            return path

    attempted = "\n  - ".join(str(path) for path in attempts)
    raise SystemExit(
        f"Task '{task_name}' could not locate '{candidate}' for {label}. "
        f"Tried:\n  - {attempted}"
    )


def main() -> None:
    args = parse_args()

    batch_path = args.batch_file
    if not batch_path.exists():
        raise SystemExit(f"Batch file not found: {batch_path}")

    flux_script = args.flux_script
    if not flux_script.exists():
        raise SystemExit(f"Flux CLI not found: {flux_script}")

    tasks = _load_tasks(batch_path)
    total = len(tasks)
    limit = args.limit if args.limit is not None else total

    base_dirs = [Path.cwd().resolve()]
    batch_dir = batch_path.parent.resolve()
    if batch_dir not in base_dirs:
        base_dirs.append(batch_dir)

    failures = 0
    for index, task in enumerate(tasks, start=1):
        if index > limit:
            break

        payload = task.get("payload")
        if not isinstance(payload, dict):
            raise SystemExit(f"Task #{index} is missing a 'payload' object.")

        outfit = payload.get("outfit") if isinstance(payload.get("outfit"), dict) else {}
        if not isinstance(outfit, dict):
            outfit = {}

        task_name = str(outfit.get("output_subdir") or task.get("name") or f"task-{index:02d}")
        original_path = _resolve_asset(
            candidate=payload.get("image_path"),
            search_dirs=base_dirs,
            label="payload.image_path",
            task_name=task_name,
        )
        mask_path = _resolve_asset(
            candidate=payload.get("mask_path"),
            search_dirs=base_dirs,
            label="payload.mask_path",
            task_name=task_name,
        )
        clothes_path = _resolve_asset(
            candidate=outfit.get("clothes_image_path"),
            search_dirs=base_dirs,
            label="payload.outfit.clothes_image_path",
            task_name=task_name,
        )

        command = [
            sys.executable,
            str(flux_script),
            "--original",
            str(original_path),
            "--mask",
            str(mask_path),
            "--clothes",
            str(clothes_path),
            "--task-name",
            task_name,
        ]

        if args.dotenv is not None:
            command.extend(["--dotenv", str(args.dotenv)])
        if args.skip_gpt:
            command.append("--skip-gpt")
        if args.skip_flux:
            command.append("--skip-flux")
        if args.max_size is not None:
            command.extend(["--max-size", str(args.max_size)])

        strength = outfit.get("strength")
        if strength is not None:
            command.extend(["--strength", str(strength)])

        prompt_override = payload.get("prompt")
        if isinstance(prompt_override, str) and prompt_override.strip():
            command.extend(["--prompt", prompt_override.strip()])

        negative_override = payload.get("negative_prompt")
        if isinstance(negative_override, str) and negative_override.strip():
            command.extend(["--negative-prompt", negative_override.strip()])

        print(f"[{index}/{limit}] Running task '{task_name}'...")
        result = subprocess.run(command, check=False)

        if result.returncode != 0:
            failures += 1
            print(f"[!] Task '{task_name}' failed with exit code {result.returncode}.")
            if args.stop_on_error:
                break

    processed = min(total, limit)
    if failures:
        print(f"Completed {processed} task(s) with {failures} failure(s).")
        raise SystemExit(1)

    print(f"Completed {processed} task(s) successfully.")


if __name__ == "__main__":
    main()

