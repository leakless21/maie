"""
Command-line interface for inspecting application configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .loader import get_settings
from .model import AppSettings


def _dump_settings(settings: AppSettings, show_secrets: bool) -> dict[str, Any]:
    if not show_secrets:
        return settings.model_dump(mode="json")
    python_dump = settings.model_dump(mode="python")

    def reveal(value: Any) -> Any:
        if hasattr(value, "get_secret_value"):
            try:
                return value.get_secret_value()
            except Exception:  # pragma: no cover
                return str(value)
        if isinstance(value, dict):
            return {k: reveal(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [reveal(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return reveal(python_dump)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MAIE configuration.")
    parser.add_argument(
        "-e",
        "--environment",
        help="Environment to load (development, production). Defaults to ENVIRONMENT variable.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload configuration instead of reusing cached instance.",
    )
    parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Display secret values instead of masked placeholders.",
    )
    args = parser.parse_args()

    settings = get_settings(environment=args.environment, reload=args.reload)
    payload = _dump_settings(settings, show_secrets=args.show_secrets)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
