#!/usr/bin/env python3
"""
Show current LLM backend configuration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

print("=" * 70)
print("Current LLM Backend Configuration")
print("=" * 70)
print(f"Backend Mode: {settings.llm_backend.value}")
print()

if settings.llm_backend.value == "vllm_server":
    print("vLLM Server Mode:")
    print(f"  Enhancement Endpoint: {settings.llm_server.enhance_base_url}")
    print(f"  Enhancement Model: {settings.llm_server.enhance_model_name or 'Not specified'}")
    print(f"  Summary Endpoint: {settings.llm_server.summary_url}")
    print(f"  Summary Model: {settings.llm_server.summary_model_name or 'Not specified'}")
else:
    print("Local vLLM Mode:")
    print(f"  Enhancement Model: {settings.llm_enhance.model}")
    print(f"  Summary Model: {settings.llm_sum.model}")
    print(f"  GPU Memory Utilization: {settings.llm_enhance.gpu_memory_utilization}")
    print(f"  Max Model Length: {settings.llm_enhance.max_model_len}")

print("=" * 70)
