"""
LLM processor module for MAIE.
Supports Qwen3-4B-Instruct AWQ-4bit model via direct vLLM integration.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from Levenshtein import distance as levenshtein_distance
from loguru import logger

from src.config import settings
from src.processors.base import LLMBackend, LLMResult
from src.processors.llm.config import GenerationConfig, build_generation_config
from src.processors.prompt.renderer import PromptRenderer
from src.processors.prompt.template_loader import TemplateLoader
from src.tooling.vllm_utils import apply_overrides_to_sampling, calculate_checkpoint_hash, get_model_info

from .schema_validator import (
    load_template_schema,
    retry_with_lower_temperature,
    validate_llm_output,
)


class LLMProcessor(LLMBackend):
    """
    LLM processor for text enhancement and structured summarization.
    Supports Qwen3-4B-Instruct AWQ-4bit model via direct vLLM integration.
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the LLM processor.

        Args:
            model_path: Path to the LLM model file (optional, uses config default)
            **kwargs: Additional backend-specific parameters
        """
        self.model_path = model_path or settings.llm_enhance_model
        self.model = None
        self.tokenizer = None
        self.checkpoint_hash = None
        self.model_info = None
        self.current_template_id = None
        self.current_schema_hash = None

        # Initialize prompt rendering system
        template_loader = TemplateLoader(settings.templates_dir / "prompts")
        self.prompt_renderer = PromptRenderer(template_loader)

        # Build environment configs for different tasks
        self.env_config_enhancement = GenerationConfig(
            temperature=settings.llm_enhance_temperature,
            top_p=settings.llm_enhance_top_p,
            top_k=settings.llm_enhance_top_k,
            max_tokens=settings.llm_enhance_max_tokens,
        )

        self.env_config_summary = GenerationConfig(
            temperature=settings.llm_sum_temperature,
            top_p=settings.llm_sum_top_p,
            top_k=settings.llm_sum_top_k,
            max_tokens=settings.llm_sum_max_tokens,
        )

        # Load model lazily (only when first needed)
        self._model_loaded = False

    def _detect_quantization_method(self, model_path: str) -> str | None:
        """
        Auto-detect quantization method from model configuration.

        Args:
            model_path: Path to the model directory

        Returns:
            Detected quantization method or None if not found
        """
        try:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                return None

            # Check for config.json
            config_file = model_path_obj / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Check for quantization config
                quantization_config = config.get("quantization_config", {})
                if quantization_config:
                    quant_method = quantization_config.get("quant_method")
                    if quant_method:
                        logger.info(
                            f"Auto-detected quantization method: {quant_method}"
                        )
                        return quant_method

                # Check for other quantization indicators
                if "compressed_tensors" in str(config).lower():
                    logger.info("Auto-detected quantization method: compressed-tensors")
                    return "compressed-tensors"
                elif "awq" in str(config).lower():
                    logger.info("Auto-detected quantization method: awq")
                    return "awq"

            # Default fallback
            logger.info("No quantization method detected, using default")
            return None

        except Exception as e:
            logger.warning(f"Failed to detect quantization method: {e}")
            return None

    def _load_model(self, **kwargs) -> None:
        """
        Load vLLM model with AWQ quantization.

        Args:
            **kwargs: Additional model loading parameters
        """
        if self._model_loaded:
            return
        try:
            from vllm import LLM

            # Determine model name based on task or use default
            model_name = kwargs.get("model_name", self.model_path)

            logger.info(f"Loading LLM model: {model_name}")

            # Load model with configurable quantization
            quantization_method = (
                settings.llm_enhance_quantization
                or self._detect_quantization_method(model_name)
            )

            # Build LLM arguments
            llm_args = {
                "model": model_name,
                "gpu_memory_utilization": settings.llm_enhance_gpu_memory_utilization,
                "max_model_len": settings.llm_enhance_max_model_len,
                "trust_remote_code": True,
                "enforce_eager": False,  # Use CUDA graphs for performance
            }

            # Only add quantization if specified
            if quantization_method:
                llm_args["quantization"] = quantization_method

            self.model = LLM(**llm_args)

            # Calculate checkpoint hash for versioning (NFR-1)
            try:
                model_path = Path(model_name)
                if model_path.exists():
                    self.checkpoint_hash = calculate_checkpoint_hash(model_path)
                    self.model_info = get_model_info(model_path)
                else:
                    # For HuggingFace models, use model name as hash
                    self.checkpoint_hash = f"hf:{model_name}"
                    self.model_info = {
                        "model_name": model_name,
                        "checkpoint_hash": self.checkpoint_hash,
                    }
            except Exception as e:
                logger.warning(f"Failed to calculate model hash: {e}")
                self.checkpoint_hash = f"unknown:{model_name}"
                self.model_info = {
                    "model_name": model_name,
                    "checkpoint_hash": self.checkpoint_hash,
                }

            self._model_loaded = True
            logger.info(f"LLM model loaded successfully: {model_name}")

        except ImportError:
            logger.warning("vLLM not installed, LLM features will be unavailable")
            self.model = None
            self.checkpoint_hash = "vllm_not_installed"
            self.model_info = {
                "model_name": "unavailable",
                "reason": "vLLM not installed",
            }
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
            self.checkpoint_hash = f"load_error:{str(e)[:50]}"
            self.model_info = {
                "model_name": "unavailable",
                "reason": f"Load error: {e}",
            }
            raise

    def execute(self, text: str, **kwargs) -> LLMResult:
        """
        Execute LLM processing on text for enhancement and summarization.

        Args:
            text: Input text to process
            **kwargs: Additional parameters including task type

        Returns:
            LLMResult with processed text and metadata
        """
        task = kwargs.get("task", "general")

        # Load model if not already loaded
        if not self._model_loaded:
            self._load_model(**kwargs)

        if self.model is None:
            return LLMResult(
                text=text,
                tokens_used=None,
                model_info=self.model_info
                or {"model_name": "unavailable", "reason": "vLLM not installed"},
                metadata={"task": task, "fallback": True},
            )

        # Use text as prompt directly for basic execution
        prompt_text = text

        # Select environment config based on task
        env_config = (
            self.env_config_enhancement
            if task == "enhancement"
            else self.env_config_summary
        )

        # Extract runtime overrides from kwargs
        candidate_keys = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "max_new_tokens",
            "stop",
            "stop_token_ids",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
            "logprobs",
            "prompt_logprobs",
        }
        runtime_overrides_dict = {k: kwargs[k] for k in candidate_keys if k in kwargs}
        runtime_config = GenerationConfig(**runtime_overrides_dict)

        # Extract guided_decoding if present
        guided_decoding = kwargs.get("guided_decoding", None)

        # Build final config using hierarchy
        model_path = Path(self.model_path) if Path(self.model_path).exists() else None
        final_config = build_generation_config(
            model_path=model_path,
            env_overrides=env_config,
            runtime_overrides=runtime_config,
        )

        # Convert to SamplingParams
        sampling_params_dict = final_config.to_sampling_params()

        try:
            from vllm import SamplingParams

            sampling = SamplingParams(
                **sampling_params_dict, guided_decoding=guided_decoding
            )

            # Generate text
            outputs = self.model.generate([prompt_text], sampling)
            generated_text = outputs[0].outputs[0].text if outputs else ""

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            generated_text = text

        return LLMResult(
            text=generated_text or text,
            tokens_used=None,
            model_info=self.model_info or {"model_name": "unknown"},
            metadata={"task": task, "config": sampling_params_dict},
        )

    def enhance_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Enhance text with proper punctuation and capitalization using LLM.

        Args:
            text: Raw text to enhance (e.g., from ASR without punctuation)
            **kwargs: Additional parameters for enhancement

        Returns:
            Dictionary containing enhanced text and enhancement metrics
        """
        # Load model if not already loaded
        if not self._model_loaded:
            self._load_model(**kwargs)

        if self.model is None:
            return {
                "enhanced_text": text,
                "original_text": text,
                "enhancement_applied": False,
                "edit_distance": 0,
                "model_info": self.model_info or {"model_name": "unavailable"},
            }

        # Render enhancement prompt
        try:
            prompt = self.prompt_renderer.render(
                "text_enhancement_v1", text_input=text, add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Failed to render enhancement prompt: {e}")
            return {
                "enhanced_text": text,
                "original_text": text,
                "enhancement_applied": False,
                "edit_distance": 0,
                "model_info": self.model_info or {"model_name": "unavailable"},
            }

        # Generate enhanced text
        result = self.execute(prompt, task="enhancement", **kwargs)
        enhanced_text = result.text.strip()

        # Calculate edit distance for metrics
        edit_distance = levenshtein_distance(text, enhanced_text)
        max_length = max(len(text), len(enhanced_text))
        edit_rate = edit_distance / max_length if max_length > 0 else 0

        return {
            "enhanced_text": enhanced_text,
            "original_text": text,
            "enhancement_applied": True,
            "edit_distance": edit_distance,
            "edit_rate": edit_rate,
            "model_info": result.model_info,
        }

    def needs_enhancement(self, asr_backend: str) -> bool:
        """
        Determine if text enhancement is needed based on ASR backend.

        Args:
            asr_backend: Name of the ASR backend used

        Returns:
            True if enhancement is needed, False otherwise
        """
        # Whisper with erax-wow-turbo variant has native punctuation, so skip enhancement
        # Check if backend name starts with "whisper" (case-insensitive) to handle variants:
        # - whisper, whisper-tiny, whisper-base, whisper-large-v3, whisper_vi, etc.
        if asr_backend and asr_backend.lower().startswith("whisper"):
            return False

        # Other backends (e.g., chunkformer) may need enhancement
        return True

    def generate_summary(
        self, transcript: str, template_id: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured summary with embedded tags using JSON schema validation.

        Args:
            transcript: Enhanced transcript text
            template_id: Template identifier (e.g., "meeting_notes_v1")
            **kwargs: Additional parameters for generation

        Returns:
            Dictionary containing structured summary and metadata

        Raises:
            FileNotFoundError: If template schema not found
            ValueError: If template schema is invalid
        """
        # Load model if not already loaded
        if not self._model_loaded:
            self._load_model(**kwargs)

        if self.model is None:
            return {
                "summary": None,
                "error": "LLM model not available",
                "model_info": self.model_info or {"model_name": "unavailable"},
            }

        # Load and validate template schema
        try:
            schema = load_template_schema(template_id, settings.templates_dir)
            self.current_template_id = template_id
            # Calculate schema hash for versioning
            self.current_schema_hash = str(hash(json.dumps(schema, sort_keys=True)))
        except Exception as e:
            logger.error(f"Failed to load template schema {template_id}: {e}")
            return {
                "summary": None,
                "error": f"Template schema error: {e}",
                "model_info": self.model_info or {"model_name": "unavailable"},
            }

        # Render summarization prompt
        try:
            prompt = self.prompt_renderer.render(
                template_id,
                transcript=transcript,
                schema=json.dumps(schema, indent=2),
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.error(f"Failed to render summarization prompt: {e}")
            return {
                "summary": None,
                "error": f"Prompt rendering error: {e}",
                "model_info": self.model_info or {"model_name": "unavailable"},
            }

        # Generate summary with retry logic
        max_retries = 1  # Changed from 2 to 1 (total 2 attempts per TDD)
        current_temperature = settings.llm_sum_temperature

        # Create guided decoding parameters for JSON schema constraint
        from vllm.sampling_params import GuidedDecodingParams

        guided_decoding = GuidedDecodingParams(json=schema)

        for retry_count in range(max_retries + 1):
            try:
                # Prepare prompt (add feedback on retry)
                if retry_count > 0 and "last_error" in locals():
                    last_error_str = str(locals().get("last_error"))
                    error_hint = (
                        "\n\nIMPORTANT: Previous attempt failed validation with error: "
                        f"{last_error_str}\nPlease ensure the output is valid JSON that strictly matches the provided schema."
                    )
                    prompt_for_attempt = prompt + error_hint
                else:
                    prompt_for_attempt = prompt

                # Prepare sampling parameters with current temperature and guided decoding
                sampling_kwargs = {
                    "temperature": current_temperature,
                    "top_p": settings.llm_sum_top_p,
                    "top_k": settings.llm_sum_top_k,
                    "max_tokens": settings.llm_sum_max_tokens,
                    "guided_decoding": guided_decoding,  # Add constrained decoding
                    **kwargs,
                }

                # Generate summary
                result = self.execute(
                    prompt_for_attempt, task="summarization", **sampling_kwargs
                )
                raw_output = result.text.strip()

                # Validate output against schema
                parsed_data, error_message = validate_llm_output(raw_output, schema)

                if parsed_data is not None:
                    # Success - return structured summary
                    logger.debug(
                        f"Summary generated successfully on attempt {retry_count + 1}"
                    )
                    return {
                        "summary": parsed_data,
                        "raw_output": raw_output,
                        "retry_count": retry_count,
                        "model_info": result.model_info,
                    }
                else:
                    # Save error for next attempt
                    last_error = error_message
                    logger.warning(
                        f"Validation failed (attempt {retry_count + 1}): {error_message}"
                    )

                    if retry_count < max_retries:
                        # Reduce temperature for retry
                        current_temperature = retry_with_lower_temperature(
                            settings.llm_sum_temperature, retry_count, max_retries
                        )
                        logger.info(f"Retrying with temperature {current_temperature}")
                    else:
                        # Final attempt failed
                        logger.error(
                            f"All retry attempts failed for template {template_id}"
                        )
                        return {
                            "summary": None,
                            "error": f"Schema validation failed after {max_retries + 1} attempts: {error_message}",
                            "raw_output": raw_output,
                            "retry_count": retry_count,
                            "model_info": result.model_info,
                        }

            except Exception as e:
                logger.error(
                    f"Summary generation failed (attempt {retry_count + 1}): {e}"
                )
                if retry_count >= max_retries:
                    return {
                        "summary": None,
                        "error": f"Generation failed after {max_retries + 1} attempts: {e}",
                        "model_info": self.model_info or {"model_name": "unavailable"},
                    }

    def unload(self) -> None:
        """
        Unload the LLM model and release GPU memory.

        This method ensures proper cleanup of vLLM resources and CUDA memory.
        """
        if self.model is not None:
            try:
                # Delete the model to free GPU memory
                del self.model
                logger.debug("LLM model unloaded successfully")
            except Exception as e:
                logger.warning(f"Error during model unload: {e}")
            finally:
                self.model = None
                self._model_loaded = False

        # Clear CUDA cache (allow tests to patch module-level torch)
        try:
            import torch as _torch
        except Exception:
            _torch = None
        try:
            torch_module = globals().get("torch", _torch)
            if (
                torch_module is not None
                and hasattr(torch_module, "cuda")
                and callable(getattr(torch_module.cuda, "is_available", None))
            ):
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")

        # Reset state
        self.tokenizer = None
        self.current_template_id = None
        self.current_schema_hash = None

    def get_version_info(self) -> Dict[str, Any]:
        """
        Get complete version information for the LLM processor (NFR-1).

        Returns:
            Dictionary containing comprehensive version metadata
        """
        return {
            "name": (
                self.model_info.get("model_name", settings.llm_enhance_model)
                if self.model_info
                else settings.llm_enhance_model
            ),
            "checkpoint_hash": self.checkpoint_hash or "unknown",
            "quantization": "awq-4bit",
            "chat_template": "qwen3_nonthinking",
            "thinking": False,
            "reasoning_parser": None,
            "structured_output": {
                "backend": "json_schema",
                "schema_id": self.current_template_id or "none",
                "schema_hash": self.current_schema_hash or "none",
            },
            "decoding_params": {
                "temperature": settings.llm_sum_temperature,
                "top_p": settings.llm_sum_top_p,
                "top_k": settings.llm_sum_top_k,
                "max_tokens": settings.llm_sum_max_tokens,
                "repetition_penalty": 1.05,
            },
        }


# Expose a module-level 'torch' symbol for tests to patch
try:  # pragma: no cover
    import torch as torch  # type: ignore
except Exception:  # Expose None if torch unavailable so patching still works
    torch = None  # type: ignore
