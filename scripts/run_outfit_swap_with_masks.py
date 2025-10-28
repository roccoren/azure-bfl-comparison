from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner that feeds multiple tasks from a JSON definition into "
            "scripts/outfit_swap_with_mask.py."
        )
    )
    parser.add_argument(
        "batch_file",
        type=Path,
        help="Path to the JSON file describing the outfit swap tasks.",
    )
    parser.add_argument(
        "--outfit-script",
        type=Path,
        default=Path(__file__).with_name("outfit_swap_with_mask.py"),
        help="Optional override for the outfit swap CLI (defaults to scripts/outfit_swap_with_mask.py).",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=None,
        help="Optional .env file forwarded to the outfit swap CLI.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Forward the --execute flag so each task submits to Azure once prepared.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tasks to run from the JSON list.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch as soon as one task fails instead of continuing with the remainder.",
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
        raise SystemExit(f"Batch file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Expected the batch definition to be a JSON array but got {type(data).__name__}.")

    tasks: list[dict[str, Any]] = []
    for idx, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"Task #{idx} is not an object; found {type(entry).__name__}.")
        tasks.append(entry)
    return tasks


def _resolve_path(candidate: str | None, *, search_dirs: list[Path], field: str, task_name: str) -> Path:
    if not candidate:
        raise SystemExit(f"Task '{task_name}' is missing required field: {field}")

    path = Path(candidate)
    if path.is_absolute():
        if not path.exists():
            raise SystemExit(f"Task '{task_name}' points to a missing file: {path}")
        return path

    attempted: list[str] = []
    for base_dir in search_dirs:
        resolved = (base_dir / path).resolve()
        attempted.append(str(resolved))
        if resolved.exists():
            return resolved

    locations = "\n  - ".join(attempted)
    raise SystemExit(
        f"Task '{task_name}' could not locate '{candidate}' for {field}. "
        f"Tried:\n  - {locations}"
    )


def main() -> None:
    args = parse_args()

    batch_file = args.batch_file
    if not batch_file.exists():
        raise SystemExit(f"Batch file not found: {batch_file}")

    outfit_script = args.outfit_script
    if not outfit_script.exists():
        raise SystemExit(f"Outfit swap CLI not found: {outfit_script}")

    tasks = _load_tasks(batch_file)
    total = len(tasks)
    if args.limit is not None:
        total = min(total, args.limit)

    search_dirs = [Path.cwd().resolve()]
    batch_dir = batch_file.parent.resolve()
    if batch_dir not in search_dirs:
        search_dirs.append(batch_dir)
    failures = 0

    for index, task in enumerate(tasks, start=1):
        if args.limit is not None and index > args.limit:
            break

        payload = task.get("payload")
        if not isinstance(payload, dict):
            raise SystemExit(f"Task #{index} does not define a 'payload' object.")

        outfit = payload.get("outfit") if isinstance(payload.get("outfit"), dict) else {}
        if not isinstance(outfit, dict):
            outfit = {}

        task_name = str(outfit.get("output_subdir") or task.get("name") or f"task-{index:02d}")

        original_path = _resolve_path(
            payload.get("image_path"),
            search_dirs=search_dirs,
            field="payload.image_path",
            task_name=task_name,
        )
        mask_path = _resolve_path(
            payload.get("mask_path"),
            search_dirs=search_dirs,
            field="payload.mask_path",
            task_name=task_name,
        )
        clothes_path = _resolve_path(
            outfit.get("clothes_image_path"),
            search_dirs=search_dirs,
            field="payload.outfit.clothes_image_path",
            task_name=task_name,
        )

        command = [
            sys.executable,
            str(outfit_script),
            "--original",
            str(original_path),
            "--mask",
            str(mask_path),
            "--clothes",
            str(clothes_path),
            "--task-name",
            task_name,
        ]

        strength = outfit.get("strength")
        if strength is not None:
            command.extend(["--strength", str(strength)])

        if args.dotenv is not None:
            command.extend(["--dotenv", str(args.dotenv)])

        if args.execute:
            command.append("--execute")

        print(f"[{index}/{total}] Running task '{task_name}'...")
        result = subprocess.run(command, check=False)

        if result.returncode != 0:
            failures += 1
            print(f"[!] Task '{task_name}' failed with exit code {result.returncode}.")
            if args.stop_on_error:
                break

    completed = min(len(tasks), args.limit or len(tasks))
    if failures:
        print(f"Finished {completed} task(s) with {failures} failure(s).")
        raise SystemExit(1)

    print(f"Finished {completed} task(s) successfully.")


if __name__ == "__main__":
    main()
