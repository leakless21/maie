#!/usr/bin/env python3
"""
Convert a Hugging Face Whisper (or compatible Transformers) checkpoint to
the CTranslate2 format expected by faster-whisper.

This is a thin Python wrapper around the official `ct2-transformers-converter`
CLI from the `ctranslate2` package. It validates inputs, forwards the common
arguments, and verifies the converted output contains `model.bin`.

Examples:
  # Convert a remote HF model repo to CT2 with float16 weights
  python scripts/convert_whisper_to_ct2.py \
    --model hangnguyen25/whisper-small-vi \
    --output-dir data/models/whisper-small-vi-ct2 \
    --quantization float16 --force

  # Convert from a local model directory (offline)
  python scripts/convert_whisper_to_ct2.py \
    --model /path/to/local/whisper-small-vi \
    --output-dir data/models/whisper-small-vi-ct2

Notes:
  - You need `ctranslate2` installed. If missing, install with:
      pip install ctranslate2 transformers sentencepiece huggingface_hub
  - For private HF repos, set env var `HUGGING_FACE_HUB_TOKEN` before running.
  - The output directory should be used with MAIE's Whisper backend via
    `--model-path` or `WHISPER_MODEL_PATH`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _which(prog: str) -> str | None:
    """Return absolute path to an executable if found in PATH."""
    return shutil.which(prog)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hugging Face Whisper/Transformers model to CTranslate2 format"
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model identifier or local path. For example: openai/whisper-small, "
            "hangnguyen25/whisper-small-vi, or /path/to/local/model"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the converted CT2 model will be written",
    )
    parser.add_argument(
        "--quantization",
        choices=[
            "int8",
            "int8_float32",
            "int8_float16",
            "int8_bfloat16",
            "int16",
            "float16",
            "bfloat16",
            "float32",
        ],
        default="float16",
        help="Weight quantization type for the converted model (default: float16)",
    )
    parser.add_argument(
        "--copy-files",
        nargs="*",
        default=[],
        help=(
            "Optional list of filenames to copy from the source model to the "
            "output directory (e.g., tokenizer.json preprocessor_config.json)"
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision on the HF Hub (tag, branch, or commit sha)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow converting models that require custom code",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Enable Transformers low_cpu_mem_usage flag during loading",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help=(
            "Optional HF cache directory (sets HF_HOME env only for this run). "
            "Defaults to current HF settings"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    src = Path(args.model)
    out_dir = Path(args.output_dir)

    # Basic validations
    if src.exists() and not src.is_dir():
        print(f"[ERROR] --model path exists but is not a directory: {src}", file=sys.stderr)
        return 2

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(
            f"[ERROR] Output directory already exists and is not empty: {out_dir}. "
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        return 2

    # Ensure ctranslate2 converter is available
    converter = _which("ct2-transformers-converter")
    if converter is None:
        print(
            "[ERROR] ct2-transformers-converter is not available.\n"
            "Install dependencies: pip install ctranslate2 transformers sentencepiece huggingface_hub",
            file=sys.stderr,
        )
        return 3

    # Build command
    cmd = [
        converter,
        "--model",
        args.model,
        "--output_dir",
        str(out_dir),
        "--quantization",
        args.quantization,
    ]

    if args.copy_files:
        cmd.extend(["--copy_files", *args.copy_files])

    if args.revision:
        cmd.extend(["--revision", args.revision])

    if args.trust_remote_code:
        cmd.append("--trust_remote_code")

    if args.low_cpu_mem_usage:
        cmd.append("--low_cpu_mem_usage")

    if args.force:
        cmd.append("--force")

    # Prepare environment (optionally override HF cache dir for this run)
    env = os.environ.copy()
    if args.hf_home:
        env["HF_HOME"] = str(Path(args.hf_home).expanduser())

    print("[INFO] Running:", " ".join(cmd))

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conversion failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1

    # Post-conversion sanity check
    model_bin = out_dir / "model.bin"
    if not model_bin.exists():
        print(
            f"[WARNING] Conversion finished but {model_bin} was not found.\n"
            "The output may be incomplete. Please check the converter logs.",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] Conversion complete: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

