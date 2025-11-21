#!/usr/bin/env python3
"""
vLLM Server Benchmark Script

Tests the performance of the vLLM server to diagnose slow inference speeds.
Measures:
- Time to First Token (TTFT)
- Tokens per second (throughput)
- End-to-end latency
- Server response times
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


class VLLMBenchmark:
    """Benchmark utility for vLLM server."""

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None):
        """
        Initialize benchmark client.

        Args:
            base_url: vLLM server base URL (e.g., http://localhost:8001/v1)
            model_name: Model name to use for requests
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def test_health(self) -> bool:
        """Test if server is healthy and responsive."""
        try:
            # Try models endpoint
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=5.0
            )
            if response.status_code == 200:
                print(f"✓ Server is healthy (status: {response.status_code})")
                models = response.json()
                print(f"  Available models: {json.dumps(models, indent=2)}")
                return True
            else:
                print(f"✗ Server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Server health check failed: {e}")
            return False

    def benchmark_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.5,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Benchmark a single completion request.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to use streaming

        Returns:
            Dictionary with benchmark metrics
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        start_time = time.time()
        first_token_time = None
        tokens_received = 0
        generated_text = ""

        try:
            if stream:
                # Streaming request
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=300.0,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            if first_token_time is None:
                                first_token_time = time.time()
                            
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                text = chunk["choices"][0].get("text", "")
                                generated_text += text
                                tokens_received += 1
                        except json.JSONDecodeError:
                            continue

            else:
                # Non-streaming request
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=300.0,
                )
                response.raise_for_status()
                
                result = response.json()
                first_token_time = time.time()  # For non-streaming, TTFT = total time
                
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0].get("text", "")
                    if "usage" in result:
                        tokens_received = result["usage"].get("completion_tokens", 0)

            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_time
            tokens_per_second = tokens_received / total_time if total_time > 0 else 0

            return {
                "success": True,
                "total_time": total_time,
                "ttft": ttft,
                "tokens_received": tokens_received,
                "tokens_per_second": tokens_per_second,
                "generated_text": generated_text,
                "prompt_length": len(prompt),
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time,
            }

    def benchmark_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.5,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Benchmark a chat completion request.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to use streaming

        Returns:
            Dictionary with benchmark metrics
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        start_time = time.time()
        first_token_time = None
        tokens_received = 0
        generated_text = ""

        try:
            if stream:
                # Streaming request
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=300.0,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            if first_token_time is None:
                                first_token_time = time.time()
                            
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                text = delta.get("content", "")
                                generated_text += text
                                if text:
                                    tokens_received += 1
                        except json.JSONDecodeError:
                            continue

            else:
                # Non-streaming request
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=300.0,
                )
                response.raise_for_status()
                
                result = response.json()
                first_token_time = time.time()  # For non-streaming, TTFT = total time
                
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    generated_text = message.get("content", "")
                    if "usage" in result:
                        tokens_received = result["usage"].get("completion_tokens", 0)

            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_time
            tokens_per_second = tokens_received / total_time if total_time > 0 else 0

            return {
                "success": True,
                "total_time": total_time,
                "ttft": ttft,
                "tokens_received": tokens_received,
                "tokens_per_second": tokens_per_second,
                "generated_text": generated_text,
                "messages_count": len(messages),
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time,
            }

    def run_benchmark_suite(
        self,
        num_iterations: int = 5,
        prompt_lengths: List[int] = [50, 200, 500],
        max_tokens: int = 100,
    ):
        """
        Run a comprehensive benchmark suite.

        Args:
            num_iterations: Number of iterations per test
            prompt_lengths: List of prompt lengths to test
            max_tokens: Maximum tokens to generate per request
        """
        print("\n" + "=" * 70)
        print("vLLM Server Benchmark Suite")
        print("=" * 70)
        print(f"Server: {self.base_url}")
        print(f"Model: {self.model_name}")
        print(f"Iterations per test: {num_iterations}")
        print(f"Max tokens per request: {max_tokens}")
        print("=" * 70 + "\n")

        # Test different prompt lengths
        for prompt_len in prompt_lengths:
            print(f"\n--- Testing with ~{prompt_len} character prompt ---")
            
            # Generate a prompt of approximately the desired length
            prompt = "Explain the concept of artificial intelligence. " * (prompt_len // 45)
            prompt = prompt[:prompt_len]
            
            results = []
            for i in range(num_iterations):
                print(f"  Iteration {i + 1}/{num_iterations}...", end=" ", flush=True)
                result = self.benchmark_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.5,
                    stream=False,
                )
                results.append(result)
                
                if result["success"]:
                    print(f"✓ {result['tokens_per_second']:.2f} tok/s")
                else:
                    print(f"✗ Error: {result.get('error', 'Unknown')}")

            # Calculate statistics
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                avg_ttft = sum(r["ttft"] for r in successful_results) / len(successful_results)
                avg_total_time = sum(r["total_time"] for r in successful_results) / len(successful_results)
                avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful_results) / len(successful_results)
                avg_tokens = sum(r["tokens_received"] for r in successful_results) / len(successful_results)

                print(f"\n  Results (avg over {len(successful_results)} successful runs):")
                print(f"    Time to First Token: {avg_ttft:.3f}s")
                print(f"    Total Time: {avg_total_time:.3f}s")
                print(f"    Tokens Generated: {avg_tokens:.1f}")
                print(f"    Throughput: {avg_tokens_per_sec:.2f} tokens/s")
            else:
                print(f"\n  ✗ All iterations failed")

        # Test chat completion
        print(f"\n--- Testing Chat Completion API ---")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning? Explain briefly."}
        ]
        
        chat_results = []
        for i in range(num_iterations):
            print(f"  Iteration {i + 1}/{num_iterations}...", end=" ", flush=True)
            result = self.benchmark_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.5,
                stream=False,
            )
            chat_results.append(result)
            
            if result["success"]:
                print(f"✓ {result['tokens_per_second']:.2f} tok/s")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown')}")

        # Calculate chat statistics
        successful_chat = [r for r in chat_results if r["success"]]
        if successful_chat:
            avg_ttft = sum(r["ttft"] for r in successful_chat) / len(successful_chat)
            avg_total_time = sum(r["total_time"] for r in successful_chat) / len(successful_chat)
            avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful_chat) / len(successful_chat)
            avg_tokens = sum(r["tokens_received"] for r in successful_chat) / len(successful_chat)

            print(f"\n  Results (avg over {len(successful_chat)} successful runs):")
            print(f"    Time to First Token: {avg_ttft:.3f}s")
            print(f"    Total Time: {avg_total_time:.3f}s")
            print(f"    Tokens Generated: {avg_tokens:.1f}")
            print(f"    Throughput: {avg_tokens_per_sec:.2f} tokens/s")
        else:
            print(f"\n  ✗ All iterations failed")

        print("\n" + "=" * 70)
        print("Benchmark Complete")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server performance"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="vLLM server base URL (default: from config)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name to use (default: from config)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (default: from config)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per test (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per request (default: 100)",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        help="Prompt lengths to test (default: 50 200 500)",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only check server health, don't run benchmarks",
    )

    args = parser.parse_args()

    # Use config defaults if not specified
    base_url = args.base_url or settings.llm_server.enhance_base_url
    model_name = args.model_name or settings.llm_server.enhance_model_name or "maie-enhance"
    api_key = args.api_key
    if api_key is None and settings.llm_server.enhance_api_key:
        api_key = settings.llm_server.enhance_api_key.get_secret_value()

    print(f"Connecting to vLLM server at: {base_url}")
    print(f"Using model: {model_name}")

    benchmark = VLLMBenchmark(base_url, model_name, api_key)

    # Test server health
    if not benchmark.test_health():
        print("\n✗ Server health check failed. Please ensure the vLLM server is running.")
        sys.exit(1)

    if args.health_only:
        print("\n✓ Health check passed. Exiting.")
        sys.exit(0)

    # Run benchmark suite
    benchmark.run_benchmark_suite(
        num_iterations=args.iterations,
        prompt_lengths=args.prompt_lengths,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
