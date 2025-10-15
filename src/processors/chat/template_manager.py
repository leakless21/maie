# Chat package removed — template manager no longer present.
# The project now relies solely on the default runtime chat template (e.g., vLLM's built-in template).
# This file is intentionally left as a lightweight placeholder so imports that target
# src.processors.chat.template_manager (if any remain) will still succeed.
#
# Do not add logic here; custom chat templates are deprecated and removed.

from typing import NoReturn


def deprecated_import() -> NoReturn:
    """Indicates that the chat template manager was removed.

    If code still attempts to use chat template functionality, it should be
    updated to rely on the default chat template provided at runtime.
    """
    raise RuntimeError(
        "ChatTemplateManager has been removed. Use the default runtime chat template "
        "and remove references to src.processors.chat.template_manager."
    )
