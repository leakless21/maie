#!/usr/bin/env python3
"""
MAIE LLM Processor Benchmark Script

Tests the LLM processor as used by MAIE to identify performance bottlenecks.
This simulates the actual usage pattern in the pipeline.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.processors.llm.processor import LLMProcessor


def benchmark_enhancement(
    processor: LLMProcessor,
    text: str,
    num_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark text enhancement task.

    Args:
        processor: LLM processor instance
        text: Input text to enhance
        num_iterations: Number of iterations to run

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n--- Benchmarking Enhancement Task ---")
    print(f"Input text length: {len(text)} characters")
    print(f"Iterations: {num_iterations}")
    
    results = []
    
    for i in range(num_iterations):
        print(f"  Iteration {i + 1}/{num_iterations}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = processor.execute(text, task="enhancement")
            end_time = time.time()
            
            elapsed = end_time - start_time
            tokens_used = result.tokens_used or 0
            throughput = tokens_used / elapsed if elapsed > 0 and tokens_used > 0 else 0
            
            results.append({
                "success": True,
                "elapsed": elapsed,
                "tokens_used": tokens_used,
                "throughput": throughput,
                "output_length": len(result.text) if result.text else 0,
            })
            
            print(f"✓ {elapsed:.2f}s, {throughput:.2f} tok/s")
            
        except Exception as e:
            end_time = time.time()
            results.append({
                "success": False,
                "elapsed": end_time - start_time,
                "error": str(e),
            })
            print(f"✗ Error: {e}")
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    if successful:
        avg_elapsed = sum(r["elapsed"] for r in successful) / len(successful)
        avg_tokens = sum(r["tokens_used"] for r in successful) / len(successful)
        avg_throughput = sum(r["throughput"] for r in successful) / len(successful)
        
        print(f"\n  Results (avg over {len(successful)} successful runs):")
        print(f"    Total Time: {avg_elapsed:.3f}s")
        print(f"    Tokens Used: {avg_tokens:.1f}")
        print(f"    Throughput: {avg_throughput:.2f} tokens/s")
        
        return {
            "task": "enhancement",
            "successful_runs": len(successful),
            "avg_elapsed": avg_elapsed,
            "avg_tokens": avg_tokens,
            "avg_throughput": avg_throughput,
        }
    else:
        print(f"\n  ✗ All iterations failed")
        return {
            "task": "enhancement",
            "successful_runs": 0,
            "error": "All iterations failed",
        }


def benchmark_summary(
    processor: LLMProcessor,
    text: str,
    template_id: str = "summary_v1",
    num_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark summary generation task.

    Args:
        processor: LLM processor instance
        text: Input text to summarize
        template_id: Template ID for summary
        num_iterations: Number of iterations to run

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n--- Benchmarking Summary Task ---")
    print(f"Template: {template_id}")
    print(f"Input text length: {len(text)} characters")
    print(f"Iterations: {num_iterations}")
    
    results = []
    
    for i in range(num_iterations):
        print(f"  Iteration {i + 1}/{num_iterations}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = processor.execute(
                text,
                task="summary",
                template_id=template_id,
            )
            end_time = time.time()
            
            elapsed = end_time - start_time
            tokens_used = result.tokens_used or 0
            throughput = tokens_used / elapsed if elapsed > 0 and tokens_used > 0 else 0
            
            # Try to parse JSON output
            output_valid = False
            try:
                if result.text:
                    json.loads(result.text)
                    output_valid = True
            except json.JSONDecodeError:
                pass
            
            results.append({
                "success": True,
                "elapsed": elapsed,
                "tokens_used": tokens_used,
                "throughput": throughput,
                "output_length": len(result.text) if result.text else 0,
                "output_valid": output_valid,
            })
            
            status = "✓" if output_valid else "⚠"
            print(f"{status} {elapsed:.2f}s, {throughput:.2f} tok/s")
            
        except Exception as e:
            end_time = time.time()
            results.append({
                "success": False,
                "elapsed": end_time - start_time,
                "error": str(e),
            })
            print(f"✗ Error: {e}")
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    if successful:
        avg_elapsed = sum(r["elapsed"] for r in successful) / len(successful)
        avg_tokens = sum(r["tokens_used"] for r in successful) / len(successful)
        avg_throughput = sum(r["throughput"] for r in successful) / len(successful)
        valid_outputs = sum(1 for r in successful if r.get("output_valid", False))
        
        print(f"\n  Results (avg over {len(successful)} successful runs):")
        print(f"    Total Time: {avg_elapsed:.3f}s")
        print(f"    Tokens Used: {avg_tokens:.1f}")
        print(f"    Throughput: {avg_throughput:.2f} tokens/s")
        print(f"    Valid JSON Outputs: {valid_outputs}/{len(successful)}")
        
        return {
            "task": "summary",
            "template_id": template_id,
            "successful_runs": len(successful),
            "avg_elapsed": avg_elapsed,
            "avg_tokens": avg_tokens,
            "avg_throughput": avg_throughput,
            "valid_outputs": valid_outputs,
        }
    else:
        print(f"\n  ✗ All iterations failed")
        return {
            "task": "summary",
            "template_id": template_id,
            "successful_runs": 0,
            "error": "All iterations failed",
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark MAIE LLM processor performance"
    )
    parser.add_argument(
        "--task",
        choices=["enhancement", "summary", "both"],
        default="both",
        help="Task to benchmark (default: both)",
    )
    parser.add_argument(
        "--template-id",
        default="summary_v1",
        help="Template ID for summary task (default: summary_v1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per test (default: 3)",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to input text file (optional)",
    )
    parser.add_argument(
        "--input-text",
        help="Input text to use (optional, overrides --input-file)",
    )

    args = parser.parse_args()

    # Determine input text
    if args.input_text:
        input_text = args.input_text
    elif args.input_file:
        if not args.input_file.exists():
            print(f"✗ Input file not found: {args.input_file}")
            sys.exit(1)
        input_text = args.input_file.read_text(encoding="utf-8")
    else:
        # Use default sample text
        input_text = """
        This is a sample transcript for testing the LLM processor.
        The quick brown fox jumps over the lazy dog.
        Machine learning is a subset of artificial intelligence.
        Natural language processing enables computers to understand human language.
        Deep learning uses neural networks with multiple layers.
        """

    print("=" * 70)
    print("MAIE LLM Processor Benchmark")
    print("=" * 70)
    print(f"Backend: {settings.llm_backend}")
    print(f"Server URL: {settings.llm_server.enhance_base_url}")
    print(f"Model: {settings.llm_server.enhance_model_name}")
    print(f"Input text length: {len(input_text)} characters")
    print("=" * 70)

    # Initialize processor
    print("\nInitializing LLM processor...")
    try:
        processor = LLMProcessor()
        print("✓ LLM processor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize LLM processor: {e}")
        sys.exit(1)

    benchmark_results = []

    # Run benchmarks
    if args.task in ["enhancement", "both"]:
        result = benchmark_enhancement(
            processor,
            input_text,
            num_iterations=args.iterations,
        )
        benchmark_results.append(result)

    if args.task in ["summary", "both"]:
        result = benchmark_summary(
            processor,
            input_text,
            template_id=args.template_id,
            num_iterations=args.iterations,
        )
        benchmark_results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    for result in benchmark_results:
        if result.get("successful_runs", 0) > 0:
            print(f"\n{result['task'].upper()}:")
            print(f"  Average Time: {result['avg_elapsed']:.3f}s")
            print(f"  Average Throughput: {result['avg_throughput']:.2f} tokens/s")
            if "valid_outputs" in result:
                print(f"  Valid Outputs: {result['valid_outputs']}/{result['successful_runs']}")
        else:
            print(f"\n{result['task'].upper()}: All runs failed")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
