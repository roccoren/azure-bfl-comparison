from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from azure_bfl_compare.config import load_config
from azure_bfl_compare.tasks.batch_runner import BatchRunner
from azure_bfl_compare.tasks.prompt_plan import load_batch_definition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a batch of prompts against Azure Flux and BFL Flux providers."
    )
    parser.add_argument(
        "batch_file",
        type=Path,
        help="Path to the JSON batch definition file.",
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Optional override for the batch name (defaults to batch file stem).",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=None,
        help="Optional path to a .env file containing provider credentials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    batch_path = args.batch_file
    if not batch_path.exists():
        console.print(f"[red]Batch file not found:[/red] {batch_path}")
        raise SystemExit(1)

    config = load_config(args.dotenv)
    batch = load_batch_definition(batch_path)

    batch_name = args.batch_name or batch_path.stem
    runner = BatchRunner(config=config, batch=batch, batch_name=batch_name, console=console)
    runner.run()


if __name__ == "__main__":
    main()