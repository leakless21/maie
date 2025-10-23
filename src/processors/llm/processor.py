"""
LLM processor module for MAIE.
Supports Qwen3-4B-Instruct AWQ-4bit model via direct vLLM integration.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from Levenshtein import distance as levenshtein_distance

from src.config import settings
from src.config.logging import get_module_logger
from src.processors.base import LLMBackend, LLMResult
from src.processors.llm.config import (
    GenerationConfig,
    build_generation_config,
    calculate_dynamic_max_tokens,
)
from src.processors.prompt.renderer import PromptRenderer
from src.processors.prompt.template_loader import TemplateLoader
from src.tooling.vllm_utils import (
    apply_overrides_to_sampling,
    calculate_checkpoint_hash,
    get_model_info,
)

from .schema_validator import (
    load_template_schema,
    retry_with_lower_temperature,
    validate_llm_output,
)

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)


class LLMProcessor(LLMBackend):
    """
    LLM processor for text enhancement and structured summary.
    Supports Qwen3-4B-Instruct AWQ-4bit model via direct vLLM integration.
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the LLM processor.

        Args:
            model_path: Path to the LLM model file (optional, uses config default)
            **kwargs: Additional backend-specific parameters
        """
        self.model_path = model_path or settings.llm_enhance.model
        self.model = None
        self.tokenizer = None
        self.checkpoint_hash = None
        self.model_info = None
        self.current_template_id = None
        self.current_schema_hash = None

        # Initialize prompt rendering system
        template_loader = TemplateLoader(settings.paths.templates_dir / "prompts")
        self.prompt_renderer = PromptRenderer(template_loader)

        # Build environment configs for different tasks
        self.env_config_enhancement = GenerationConfig(
            temperature=settings.llm_enhance.temperature,
            top_p=settings.llm_enhance.top_p,
            top_k=settings.llm_enhance.top_k,
            max_tokens=settings.llm_enhance.max_tokens,
        )

        self.env_config_summary = GenerationConfig(
            temperature=settings.llm_sum.temperature,
            top_p=settings.llm_sum.top_p,
            top_k=settings.llm_sum.top_k,
            max_tokens=settings.llm_sum.max_tokens,
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

        except (AttributeError, KeyError, TypeError, ValueError) as e:
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
            # Enforce GPU-only usage in all environments for vLLM
            try:
                import torch as _torch  # type: ignore

                if not _torch.cuda.is_available():
                    from os import getenv as _getenv

                    raise RuntimeError(
                        "CUDA is not available. GPU is required for vLLM. "
                        f"CUDA_VISIBLE_DEVICES={_getenv('CUDA_VISIBLE_DEVICES')!r}"
                    )
            except ImportError as _imp_err:
                raise RuntimeError(
                    "PyTorch is not installed. GPU is required for vLLM."
                ) from _imp_err

            # Disable vLLM telemetry before importing
            import os
            os.environ["VLLM_NO_USAGE_STATS"] = "1"
            os.environ["DO_NOT_TRACK"] = "1"

            from vllm import LLM
            import logging

            # Configure vLLM logging based on verbose setting
            vllm_logger = logging.getLogger("vLLM")
            if settings.verbose_components:
                vllm_logger.setLevel(logging.INFO)
                # Ensure logs are visible on console when verbose
                has_stream = any(
                    isinstance(h, logging.StreamHandler) for h in vllm_logger.handlers
                )
                if not has_stream:
                    stream_handler = logging.StreamHandler()
                    stream_handler.setLevel(logging.INFO)
                    vllm_logger.addHandler(stream_handler)
                    vllm_logger.propagate = False
                logger.info("Verbose mode enabled for vLLM")
            else:
                # Keep vLLM quiet (WARNING level)
                vllm_logger.setLevel(logging.WARNING)

            # Determine model name based on task or use default
            model_name = kwargs.get("model_name", self.model_path)

            logger.info(f"Loading LLM model: {model_name}")

            # Load model with configurable quantization
            quantization_method = (
                settings.llm_enhance.quantization
                or self._detect_quantization_method(model_name)
            )

            # Build LLM arguments
            llm_args = {
                "model": model_name,
                "gpu_memory_utilization": settings.llm_enhance.gpu_memory_utilization,
                "max_model_len": settings.llm_enhance.max_model_len,
                "trust_remote_code": True,
                "enforce_eager": False,  # Use CUDA graphs for performance
            }

            # Only add quantization if specified
            if quantization_method:
                llm_args["quantization"] = quantization_method

            if settings.llm_enhance.max_num_seqs is not None:
                llm_args["max_num_seqs"] = settings.llm_enhance.max_num_seqs
            if settings.llm_enhance.max_num_batched_tokens is not None:
                llm_args["max_num_batched_tokens"] = (
                    settings.llm_enhance.max_num_batched_tokens
                )
            if settings.llm_enhance.max_num_partial_prefills is not None:
                llm_args["max_num_partial_prefills"] = (
                    settings.llm_enhance.max_num_partial_prefills
                )

            logger.debug(f"Calling LLM() constructor with args: {llm_args}")
            self.model = LLM(**llm_args)
            logger.debug("LLM() constructor returned successfully")

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
            except (OSError, FileNotFoundError, RuntimeError) as e:
                logger.warning(f"Failed to calculate model hash: {e}")
                self.checkpoint_hash = f"unknown:{model_name}"
                self.model_info = {
                    "model_name": model_name,
                    "checkpoint_hash": self.checkpoint_hash,
                }

            self._model_loaded = True
            logger.info(f"LLM model loaded successfully: {model_name}")

            # NOTE: Tokenizer initialization is deferred to avoid blocking issues
            # with vLLM V1 engine's get_tokenizer() in multi-process mode.
            # The tokenizer will be initialized lazily when first needed.
            logger.debug(f"Tokenizer will be initialized on first use for {model_name}")

        except ImportError:
            logger.warning("vLLM not installed, LLM features will be unavailable")
            self.model = None
            self.checkpoint_hash = "vllm_not_installed"
            self.model_info = {
                "model_name": "unavailable",
                "reason": "vLLM not installed",
            }
        except (RuntimeError, OSError, MemoryError) as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
            self.checkpoint_hash = f"load_error:{str(e)[:50]}"
            self.model_info = {
                "model_name": "unavailable",
                "reason": f"Load error: {e}",
            }
            raise
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
            self.checkpoint_hash = f"load_error:{str(e)[:50]}"
            self.model_info = {
                "model_name": "unavailable",
                "reason": f"Load error: {e}",
            }
            raise

    def _ensure_tokenizer(self, model_name: str) -> None:
        """
        Ensure a tokenizer is available for token counting and prompt formatting.

        Preference order:
        1) vLLM-provided tokenizer via model.get_tokenizer()
        2) Hugging Face AutoTokenizer.from_pretrained(model_name)

        Note: This is called lazily (not during model load) to avoid blocking issues
        with vLLM V1 engine's get_tokenizer() in multi-process mode.
        """
        if self.tokenizer is not None:
            return

        logger.debug(f"Initializing tokenizer for {model_name}")

        # Try vLLM's tokenizer handle if exposed by the offline LLM instance
        # WARNING: In vLLM v0.11.0+ with V1 engine, get_tokenizer() may block
        # if called immediately after model initialization in multi-process mode.
        try:
            if self.model is not None and hasattr(self.model, "get_tokenizer"):
                logger.debug("Attempting to get tokenizer from vLLM model")
                tok = self.model.get_tokenizer()
                if tok is not None and hasattr(tok, "encode"):
                    self.tokenizer = tok
                    logger.debug("Successfully obtained tokenizer from vLLM model")
                    return
        except Exception as e:
            logger.debug(f"Unable to obtain tokenizer from vLLM model: {e}")

        # Fallback to Hugging Face tokenizer
        try:
            logger.debug(f"Falling back to Hugging Face tokenizer for {model_name}")
            from transformers import (
                AutoTokenizer,
            )  # local import to avoid hard dep when unused

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.debug(f"Successfully loaded Hugging Face tokenizer for {model_name}")
        except Exception as e:
            logger.warning(
                f"Failed to load Hugging Face tokenizer for {model_name}: {e}"
            )
            self.tokenizer = None

    def execute(self, text: str, **kwargs) -> LLMResult:
        """
        Execute LLM processing on text for enhancement and summary.

        Args:
            text: Input text to process
            **kwargs: Additional parameters including task type and optional retry_hint

        Returns:
            LLMResult with processed text and metadata
        """
        logger.info(
            f"execute() method called with text length: {len(text)}, task: {kwargs.get('task', 'general')}"
        )
        logger.info(f"First 200 chars of text: {text[:200]}")
        task = kwargs.get("task", "general")
        retry_hint = kwargs.pop("retry_hint", None)  # Extract retry hint if provided

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

        # Initialize variables for different code paths
        use_chat_api = False
        messages = None
        final_prompt = None
        tokens_used = None

        # Handle summary task with chat API and guided decoding
        if task == "summary":
            template_id = kwargs.get("template_id")
            if not template_id:
                logger.error("Summary task requires template_id parameter")
                return LLMResult(
                    text=text,
                    tokens_used=None,
                    model_info=self.model_info or {"model_name": "unknown"},
                    metadata={"task": task, "error": "Missing template_id"},
                )

            # Load schema for guided decoding
            logger.info(f"About to load schema for template {template_id}")
            try:
                schema = load_template_schema(template_id, settings.paths.templates_dir)
                logger.info(f"Schema loaded successfully for template {template_id}")
                logger.debug(f"Loaded schema for template {template_id}")
            except Exception as e:
                logger.error(f"Failed to load schema for template {template_id}: {e}")
                return LLMResult(
                    text=text,
                    tokens_used=None,
                    model_info=self.model_info or {"model_name": "unknown"},
                    metadata={"task": task, "error": f"Schema load failed: {e}"},
                )

            # Build OpenAI-format messages using chat API approach
            logger.info(f"About to render system prompt for template {template_id}")
            # Render system prompt (contains instructions + schema)
            try:
                system_prompt = self.prompt_renderer.render(
                    template_id, schema=json.dumps(schema, ensure_ascii=False, indent=2)
                )
                logger.info(
                    f"System prompt rendered successfully, length: {len(system_prompt)}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to render system prompt for template {template_id}: {e}"
                )
                return LLMResult(
                    text=text,
                    tokens_used=None,
                    model_info=self.model_info or {"model_name": "unknown"},
                    metadata={"task": task, "error": f"Prompt rendering error: {e}"},
                )

            # Append retry hint if provided
            if retry_hint:
                system_prompt += f"\n\n[RETRY FEEDBACK]\n{retry_hint}"
                logger.debug("Appended retry hint to system prompt")

            # Build user message with transcript
            logger.info(f"About to build user message, text length: {len(text)}")
            user_message_content = f"Transcript to analyze:\n{text}"
            logger.info(
                f"User message content built, length: {len(user_message_content)}"
            )

            # Build messages in OpenAI format
            logger.info("About to build messages array")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_content},
            ]
            logger.info(f"Messages array built, count: {len(messages)}")
            logger.info(
                f"User message (role={messages[1]['role']}, length={len(messages[1]['content'])})"
            )
            logger.debug(
                f"User message first 500 chars: {messages[1]['content'][:500]}"
            )
            logger.debug(f"Built messages for chat API with template {template_id}")

            use_chat_api = True

            # Set up guided decoding if not already provided
            if "guided_decoding" not in kwargs:
                try:
                    from vllm.sampling_params import GuidedDecodingParams

                    kwargs["guided_decoding"] = GuidedDecodingParams(
                        json=json.dumps(schema)
                    )
                    logger.debug("Set up guided JSON decoding")
                except Exception as e:
                    logger.warning(f"Failed to set up guided decoding: {e}")
        elif task == "enhancement":
            # Handle enhancement task with template rendering (chat format)
            try:
                final_prompt = self.prompt_renderer.render(
                    "text_enhancement_v1", text_input=text
                )
                logger.debug("Rendered enhancement prompt with chat template")
            except Exception as e:
                logger.error(f"Failed to render enhancement template: {e}")
                return LLMResult(
                    text=text,
                    tokens_used=None,
                    model_info=self.model_info or {"model_name": "unknown"},
                    metadata={"task": task, "error": f"Template render failed: {e}"},
                )
        else:
            # For other tasks, use text as prompt directly
            final_prompt = text

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

        # Calculate dynamic max_tokens if not explicitly provided
        if (
            "max_tokens" not in runtime_overrides_dict
            and "max_new_tokens" not in runtime_overrides_dict
        ):
            # Ensure tokenizer is ready
            if self.tokenizer is None:
                try:
                    self._ensure_tokenizer(self.model_path)
                except Exception as e:
                    logger.debug(f"Tokenizer initialization skipped/failed: {e}")

            # Determine input text for token calculation
            # For chat API: concatenate messages; for generate API: use final_prompt
            input_text_for_calc = None
            if use_chat_api and messages is not None:
                # Concatenate message contents for rough token estimate
                input_text_for_calc = "\n".join(
                    [msg.get("content", "") for msg in messages]
                )
            elif final_prompt is not None:
                input_text_for_calc = final_prompt

            if self.tokenizer is not None and input_text_for_calc is not None:
                try:
                    # Normalize task name for settings lookup (summary -> sum)
                    task_key = (
                        "sum"
                        if task == "summary"
                        else task.replace("enhancement", "enhance")
                    )

                    # Get model's max_model_len from settings
                    max_model_len = getattr(
                        settings, f"llm_{task_key}_max_model_len", 32768
                    )
                    if hasattr(settings, f"llm_{task_key}_max_model_len"):
                        max_model_len = getattr(
                            settings, f"llm_{task_key}_max_model_len"
                        )
                    elif hasattr(settings, "llm_enhance_max_model_len"):
                        max_model_len = getattr(settings, "llm_enhance_max_model_len")
                    else:
                        max_model_len = 32768  # fallback

                    # Calculate dynamic max_tokens using input text
                    dynamic_max_tokens = calculate_dynamic_max_tokens(
                        input_text=input_text_for_calc,
                        tokenizer=self.tokenizer,
                        task=task,
                        max_model_len=max_model_len,
                        user_override=runtime_overrides_dict.get("max_tokens"),
                    )
                    runtime_overrides_dict["max_tokens"] = dynamic_max_tokens
                    logger.debug(
                        f"Calculated dynamic max_tokens for {task}: {dynamic_max_tokens}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate dynamic max_tokens, using defaults: {e}"
                    )
            else:
                logger.debug(
                    "Tokenizer or input text not available, skipping dynamic max_tokens calculation"
                )

        # Safety fallback: ensure max_tokens is always set for summary tasks
        if task == "summary" and "max_tokens" not in runtime_overrides_dict:
            fallback_max_tokens = (
                8192  # Increased from 4096 to handle complete JSON generation
            )
            runtime_overrides_dict["max_tokens"] = fallback_max_tokens
            logger.warning(
                f"max_tokens not set for {task} task, using fallback: {fallback_max_tokens}. "
                "Consider setting --max-tokens explicitly for better control."
            )

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

        overrides = dict(sampling_params_dict)
        if guided_decoding is not None:
            overrides["guided_decoding"] = guided_decoding

        # Initialize output variables before try block
        generated_text = text  # Default fallback
        tokens_used = None
        vllm_outputs = None  # Store outputs for metadata extraction

        try:
            # Log GPU memory fragmentation before inference (NFR-10)
            if torch.cuda.is_available():
                try:
                    mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    mem_free = (
                        torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    ) / (1024**3)
                    fragmentation = (
                        mem_free / mem_reserved * 100 if mem_reserved > 0 else 0
                    )

                    logger.debug(
                        f"GPU memory before inference: allocated={mem_allocated:.2f}GB, "
                        f"reserved={mem_reserved:.2f}GB, free_in_reserved={mem_free:.2f}GB, "
                        f"fragmentation={fragmentation:.1f}%"
                    )

                    # Warn if fragmentation is high
                    if mem_free > 0.2 and fragmentation < 50:
                        logger.warning(
                            f"High memory fragmentation detected ({fragmentation:.1f}%). "
                            f"Consider setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                        )
                except Exception as mem_check_error:
                    logger.debug(f"Could not check GPU memory: {mem_check_error}")

            base_sampling = None
            if hasattr(self.model, "get_default_sampling_params"):
                try:
                    base_sampling = self.model.get_default_sampling_params()
                except Exception:
                    base_sampling = None

            if base_sampling is None:
                from vllm import SamplingParams

                sampling = SamplingParams(**overrides)
            else:
                sampling = apply_overrides_to_sampling(base_sampling, overrides)

            # Use chat() API for summary with messages, or generate() for others
            if task == "summary" and use_chat_api and messages is not None:
                logger.debug(
                    "Using chat() API for summary",
                    extra={
                        "messages_count": len(messages),
                        "sampling_params": str(sampling),
                        "model_type": type(self.model).__name__,
                    },
                )

                inference_start = time.time()
                logger.debug("About to call model.chat()")
                # Type ignore: our dict[str, str] format is compatible with vLLM's expected message format
                outputs = self.model.chat(messages=[messages], sampling_params=sampling)  # type: ignore
                vllm_outputs = outputs  # Store for metadata extraction
                inference_end = time.time()

                # Extract detailed output info for debugging
                output_info = {}
                if outputs and len(outputs) > 0 and outputs[0].outputs:
                    first_output = outputs[0].outputs[0]
                    output_info = {
                        "finish_reason": getattr(
                            first_output, "finish_reason", "unknown"
                        ),
                        "generated_tokens": len(first_output.token_ids)
                        if hasattr(first_output, "token_ids")
                        else "unknown",
                        "text_length": len(first_output.text)
                        if hasattr(first_output, "text")
                        else 0,
                    }

                logger.debug(
                    "model.chat() completed",
                    extra={
                        "inference_duration": inference_end - inference_start,
                        "outputs_count": len(outputs) if outputs else 0,
                        **output_info,
                    },
                )

                generated_text = outputs[0].outputs[0].text if outputs else ""
                tokens_used = (
                    len(outputs[0].prompt_token_ids)
                    if outputs
                    and hasattr(outputs[0], "prompt_token_ids")
                    and outputs[0].prompt_token_ids
                    else None
                )
            elif final_prompt is not None:
                logger.debug(
                    "Using generate() API",
                    extra={
                        "prompt_length": len(final_prompt),
                        "sampling_params": str(sampling),
                        "model_type": type(self.model).__name__,
                    },
                )

                inference_start = time.time()
                logger.debug("About to call model.generate()")
                # Use traditional generate() API
                outputs = self.model.generate([final_prompt], sampling)
                vllm_outputs = outputs  # Store for metadata extraction
                inference_end = time.time()

                # Extract detailed output info for debugging
                output_info = {}
                if outputs and len(outputs) > 0 and outputs[0].outputs:
                    first_output = outputs[0].outputs[0]
                    output_info = {
                        "finish_reason": getattr(
                            first_output, "finish_reason", "unknown"
                        ),
                        "generated_tokens": len(first_output.token_ids)
                        if hasattr(first_output, "token_ids")
                        else "unknown",
                        "text_length": len(first_output.text)
                        if hasattr(first_output, "text")
                        else 0,
                    }

                logger.debug(
                    "model.generate() completed",
                    extra={
                        "inference_duration": inference_end - inference_start,
                        "outputs_count": len(outputs) if outputs else 0,
                        **output_info,
                    },
                )

                generated_text = outputs[0].outputs[0].text if outputs else ""
                tokens_used = None
            else:
                # Fallback if neither path is available
                logger.error("Neither chat API messages nor final_prompt available")
                generated_text = text
                tokens_used = None

        except Exception as e:
            # Check if this is a CUDA OOM error and provide actionable guidance
            error_msg = str(e)
            if (
                "CUDA out of memory" in error_msg
                or "OutOfMemoryError" in type(e).__name__
            ):
                # Extract memory info if available
                if "reserved but unallocated" in error_msg:
                    logger.error(
                        f"LLM generation failed: {error_msg}\n"
                        "This is likely due to PyTorch memory fragmentation. "
                        "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to fix this."
                    )
                else:
                    logger.error(
                        f"LLM generation failed: {error_msg}\n"
                        "Try: (1) Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
                        "(2) Reduce APP_LLM_SUM__GPU_MEMORY_UTILIZATION, "
                        "(3) Reduce APP_LLM_SUM__MAX_MODEL_LEN"
                    )
            else:
                logger.error(f"LLM generation failed: {e}")
            generated_text = text

        # For summary tasks, validate and parse JSON output
        result_metadata = {"task": task, "config": sampling_params_dict}

        # Store vLLM output metadata if available
        if vllm_outputs and len(vllm_outputs) > 0 and vllm_outputs[0].outputs:
            first_output = vllm_outputs[0].outputs[0]
            # Handle token_ids that might be a Mock object in tests
            token_ids = getattr(first_output, "token_ids", None)
            generated_tokens = None
            if token_ids is not None and hasattr(token_ids, "__len__"):
                try:
                    generated_tokens = len(token_ids)
                except (TypeError, AttributeError):
                    # token_ids might be a Mock that doesn't support len()
                    generated_tokens = None
            result_metadata.update(
                {
                    "finish_reason": getattr(first_output, "finish_reason", "unknown"),
                    "generated_tokens": generated_tokens,
                    "output_length": len(generated_text),
                }
            )

        if task == "summary" and use_chat_api:
            result_metadata["method"] = "chat_api"
        if task == "summary":
            template_id = kwargs.get("template_id")
            try:
                # Parse JSON output
                structured_output = json.loads(generated_text)

                # Validate against schema
                if template_id:
                    schema = load_template_schema(
                        template_id, settings.paths.templates_dir
                    )
                    # validate_llm_output returns (validated_output, error_message)
                    validated_output, error_message = validate_llm_output(
                        json.dumps(structured_output), schema
                    )

                    if validated_output is not None:
                        result_metadata["structured_summary"] = validated_output
                        result_metadata["validation"] = "passed"
                        logger.info(f"Summary validation passed for {template_id}")
                    else:
                        result_metadata["validation"] = "failed"
                        result_metadata["validation_error"] = error_message
                        logger.warning(f"Summary validation failed: {error_message}")
                else:
                    result_metadata["structured_summary"] = structured_output
                    result_metadata["validation"] = "skipped"

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output: {e}")
                result_metadata["validation"] = "json_parse_error"
                result_metadata["parse_error"] = str(e)
            except Exception as e:
                logger.error(f"Summary validation error: {e}")
                result_metadata["validation"] = "error"
                result_metadata["error"] = str(e)

        return LLMResult(
            text=generated_text or text,
            tokens_used=tokens_used if "tokens_used" in locals() else None,
            model_info=self.model_info or {"model_name": "unknown"},
            metadata=result_metadata,
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

        # Generate enhanced text (execute() will render the chat template)
        result = self.execute(text, task="enhancement", **kwargs)
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
            schema = load_template_schema(template_id, settings.paths.templates_dir)
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

        # Generate summary with retry logic
        max_retries = 1  # Changed from 2 to 1 (total 2 attempts per TDD)
        current_temperature = settings.llm_sum.temperature
        last_error: Optional[str] = (
            None  # Track last validation error for retry feedback
        )

        # Create guided decoding parameters for JSON schema constraint
        from vllm.sampling_params import GuidedDecodingParams

        guided_decoding = GuidedDecodingParams(json=json.dumps(schema))

        # Log schema details for debugging
        logger.info(
            f"Starting summary generation for template {template_id}",
            extra={
                "template_id": template_id,
                "max_retries": max_retries + 1,
                "initial_temperature": current_temperature,
                "schema_summary": {
                    "type": schema.get("type"),
                    "properties": list(schema.get("properties", {}).keys()),
                    "required": schema.get("required", []),
                    "additional_properties": schema.get("additionalProperties", True),
                },
                "transcript_length": len(transcript),
                "guided_decoding_enabled": True,
            },
        )

        for retry_count in range(max_retries + 1):
            try:
                # Build retry hint from last error if retrying
                retry_hint = None
                if retry_count > 0 and last_error:
                    retry_hint = (
                        f"IMPORTANT: Previous attempt failed validation with error: {last_error}\n"
                        "Please ensure the output is valid JSON that strictly matches the provided schema."
                    )
                    logger.info(
                        f"Retrying with error feedback (attempt {retry_count + 1})"
                    )

                # Prepare sampling parameters with current temperature and guided decoding
                # NOTE: Do NOT pass max_tokens here - let execute() calculate it dynamically
                # based on input length to avoid truncation issues
                sampling_kwargs = {
                    "temperature": current_temperature,
                    "top_p": settings.llm_sum.top_p,
                    "top_k": settings.llm_sum.top_k,
                    "stop": [
                        "<|im_end|>"
                    ],  # Prevent chat template echo (BUGFIX_LLM_CHAT_TEMPLATE_ECHO.md)
                    "guided_decoding": guided_decoding,  # Add constrained decoding
                    **kwargs,
                }

                # Only pass max_tokens if explicitly provided by caller
                if "max_tokens" in kwargs:
                    sampling_kwargs["max_tokens"] = kwargs["max_tokens"]

                # Add retry_hint if available (will be appended to system prompt in execute())
                if retry_hint:
                    sampling_kwargs["retry_hint"] = retry_hint

                # Log attempt details
                logger.debug(
                    f"Generating summary (attempt {retry_count + 1}/{max_retries + 1})",
                    extra={
                        "attempt": retry_count + 1,
                        "max_attempts": max_retries + 1,
                        "temperature": current_temperature,
                        "sampling_params": {
                            "temperature": current_temperature,
                            "top_p": settings.llm_sum.top_p,
                            "top_k": settings.llm_sum.top_k,
                            "max_tokens": settings.llm_sum.max_tokens,
                        },
                    },
                )

                # Generate summary
                logger.debug(
                    f"About to call LLM inference (attempt {retry_count + 1})",
                    extra={
                        "attempt": retry_count + 1,
                        "transcript_length": len(transcript),
                        "sampling_params": {
                            "temperature": current_temperature,
                            "max_tokens": sampling_kwargs.get("max_tokens"),
                            "guided_decoding": sampling_kwargs.get("guided_decoding")
                            is not None,
                        },
                    },
                )

                inference_start_time = time.time()
                result = self.execute(
                    transcript,
                    task="summary",
                    template_id=template_id,
                    **sampling_kwargs,
                )
                inference_end_time = time.time()
                inference_duration = inference_end_time - inference_start_time

                logger.debug(
                    f"LLM inference completed (attempt {retry_count + 1})",
                    extra={
                        "attempt": retry_count + 1,
                        "inference_duration_seconds": inference_duration,
                        "inference_duration_minutes": inference_duration / 60,
                    },
                )

                # Check if execute() encountered an error (e.g., prompt rendering failure)
                if result.metadata and "error" in result.metadata:
                    error_in_execute = result.metadata["error"]
                    logger.error(
                        f"Execution error on attempt {retry_count + 1}: {error_in_execute}"
                    )
                    if retry_count >= max_retries:
                        logger.error(
                            f"All generation attempts failed for template {template_id}"
                        )
                        return {
                            "summary": None,
                            "error": f"Generation failed: {error_in_execute}",
                            "model_info": self.model_info
                            or {"model_name": "unavailable"},
                        }
                    else:
                        # Continue to retry
                        last_error = error_in_execute
                        current_temperature = retry_with_lower_temperature(
                            settings.llm_sum.temperature, retry_count, max_retries
                        )
                        continue

                raw_output = result.text.strip()

                # Extract finish_reason from metadata for debugging
                finish_reason = (
                    result.metadata.get("finish_reason", "unknown")
                    if result.metadata
                    else "unknown"
                )

                # Log the raw output for debugging
                output_preview = raw_output[:200] + (
                    "..." if len(raw_output) > 200 else ""
                )
                logger.debug(
                    f"LLM generated output (attempt {retry_count + 1})",
                    extra={
                        "attempt": retry_count + 1,
                        "raw_output": raw_output,
                        "output_preview": output_preview,
                        "output_length": len(raw_output),
                        "stop_reason": finish_reason,
                        "is_empty": len(raw_output.strip()) == 0,
                        "starts_with_brace": raw_output.strip().startswith("{"),
                        "ends_with_brace": raw_output.strip().endswith("}"),
                    },
                )

                # Check if output was truncated due to generation limit
                if finish_reason == "length":
                    max_output = settings.llm_sum.max_tokens
                    logger.warning(
                        f"LLM output was truncated on attempt {retry_count + 1}. "
                        f"The model hit the generation limit of {max_output} output units. "
                        f"Consider increasing the output limit in configuration."
                    )

                # Validate output against schema
                parsed_data, error_message = validate_llm_output(raw_output, schema)

                if parsed_data is not None:
                    # Post-process to reduce hallucinations for certain templates
                    try:
                        parsed_data = self._postprocess_summary(
                            template_id, transcript, parsed_data
                        )
                    except Exception:
                        pass

                    # Success - return structured summary
                    logger.info(
                        f"Summary generated successfully on attempt {retry_count + 1}",
                        extra={
                            "template_id": template_id,
                            "attempt": retry_count + 1,
                            "final_temperature": current_temperature,
                            "parsed_keys": list(parsed_data.keys()),
                            "validation_passed": True,
                        },
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
                        f"Validation failed (attempt {retry_count + 1}): {error_message}",
                        extra={
                            "template_id": template_id,
                            "attempt": retry_count + 1,
                            "error_message": error_message,
                            "raw_output": raw_output,
                            "output_preview": output_preview,
                            "will_retry": retry_count < max_retries,
                        },
                    )

                    if retry_count < max_retries:
                        # Reduce temperature for retry
                        current_temperature = retry_with_lower_temperature(
                            settings.llm_sum.temperature, retry_count, max_retries
                        )
                        logger.info(
                            f"Retrying with temperature {current_temperature}",
                            extra={
                                "template_id": template_id,
                                "attempt": retry_count + 1,
                                "new_temperature": current_temperature,
                                "previous_temperature": settings.llm_sum.temperature,
                            },
                        )
                    else:
                        # Final attempt failed - comprehensive error logging
                        logger.error(
                            f"All retry attempts failed for template {template_id}",
                            extra={
                                "template_id": template_id,
                                "total_attempts": max_retries + 1,
                                "final_error": error_message,
                                "final_raw_output": raw_output,
                                "final_output_preview": output_preview,
                                "temperature_history": [
                                    settings.llm_sum.temperature * (0.5**i)
                                    for i in range(max_retries + 1)
                                ],
                                "schema_summary": {
                                    "type": schema.get("type"),
                                    "properties": list(
                                        schema.get("properties", {}).keys()
                                    ),
                                    "required": schema.get("required", []),
                                },
                                "generation_failure_type": "validation_error",
                            },
                        )
                        return {
                            "summary": None,
                            "error": f"Schema validation failed after {max_retries + 1} attempts: {error_message}",
                            "raw_output": raw_output,
                            "retry_count": retry_count,
                            "model_info": result.model_info,
                        }

            except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as e:
                logger.error(
                    f"Summary generation failed (attempt {retry_count + 1}): {e}",
                    extra={
                        "template_id": template_id,
                        "attempt": retry_count + 1,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "will_retry": retry_count < max_retries,
                    },
                )
                if retry_count >= max_retries:
                    logger.error(
                        f"All generation attempts failed for template {template_id}",
                        extra={
                            "template_id": template_id,
                            "total_attempts": max_retries + 1,
                            "final_exception": str(e),
                            "generation_failure_type": "generation_error",
                        },
                    )
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
                self.model_info.get("model_name", settings.llm_enhance.model)
                if self.model_info
                else settings.llm_enhance.model
            ),
            "checkpoint_hash": self.checkpoint_hash or "unknown",
            "quantization": "awq-4bit",
            "thinking": False,
            "reasoning_parser": None,
            "structured_output": {
                "backend": "json_schema",
                "schema_id": self.current_template_id or "none",
                "schema_hash": self.current_schema_hash or "none",
            },
            "decoding_params": {
                "temperature": settings.llm_sum.temperature,
                "top_p": settings.llm_sum.top_p,
                "top_k": settings.llm_sum.top_k,
                "max_tokens": settings.llm_sum.max_tokens,
                "repetition_penalty": 1.05,
            },
        }

    # Internal helpers
    def _postprocess_summary(
        self, template_id: str, transcript: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Lightweight guardrails to curb hallucinated names/dates.

        - For meeting_notes_v1: ensure meeting_date appears in transcript; else null.
          Ensure participants are verbatim substrings of transcript; else drop.
        - For interview_transcript_v1: ensure interview_date appears in transcript; else null.
        """
        try:
            t_lower = (transcript or "").lower()
            if template_id == "meeting_notes_v1":
                md = data.get("meeting_date")
                if isinstance(md, str) and md and (md.lower() not in t_lower):
                    data["meeting_date"] = None

                participants = data.get("participants")
                if isinstance(participants, list):
                    filtered = []
                    for p in participants:
                        if isinstance(p, str) and p.strip() and (p.lower() in t_lower):
                            filtered.append(p)
                    data["participants"] = filtered

            elif template_id == "interview_transcript_v1":
                md = data.get("interview_date")
                if isinstance(md, str) and md and (md.lower() not in t_lower):
                    data["interview_date"] = None

        except Exception:
            # Best-effort; never fail the pipeline on guardrails
            return data

        return data


# Expose a module-level 'torch' symbol for tests to patch
try:  # pragma: no cover
    import torch as torch  # type: ignore
except Exception:  # Expose None if torch unavailable so patching still works
    torch = None  # type: ignore
