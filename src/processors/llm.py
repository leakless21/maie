"""
LLM processor module for MAIE.
Supports Qwen3-4B-Instruct AWQ-4bit model via direct vLLM integration.
"""

from typing import Any, Dict, Optional
from src.processors.base import LLMBackend, LLMResult
from src.processors.prompt.renderer import PromptRenderer
from src.processors.chat.template_manager import ChatTemplateManager
from src.tooling.vllm_utils import apply_overrides_to_sampling, normalize_overrides


class LLMProcessor(LLMBackend):
    """
    LLM processor for text enhancement and structured summarization.
    Supports Qwen3-4B-Instruct AWQ-4bit model via direct vLLM integration.
    """
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the LLM processor.
        
        Args:
            model_path: Path to the LLM model file
            **kwargs: Additional backend-specific parameters
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.chat_template_manager = ChatTemplateManager()
        self.prompt_renderer = PromptRenderer()
        self._load_model(**kwargs)
    
    def _load_model(self, **kwargs) -> None:
        """
        Load the LLM model.
        """
        # Implementation will be added later
        pass
    
    def execute(self, text: str, **kwargs) -> LLMResult:
        """
        Execute LLM processing on text for enhancement and summarization.
        """
        task = kwargs.get("task", "general")

        if self.model is None:
            return LLMResult(
                text=text,
                tokens_used=None,
                model_info={"model_name": "unavailable", "reason": "vLLM not installed"},
                metadata={"task": task, "fallback": True},
            )

        prompt_text = text

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
        request_overrides: Dict[str, Any] = {k: kwargs[k] for k in candidate_keys if k in kwargs}

        # Defaults from config per task
        if task == "enhancement":
            base_overrides: Dict[str, Any] = {
                "temperature": settings.llm_enhance_temperature,
                "top_p": settings.llm_enhance_top_p,
                "top_k": settings.llm_enhance_top_k,
                "max_tokens": settings.llm_enhance_max_tokens,
            }
        else:
            base_overrides = {
                "temperature": settings.llm_sum_temperature,
                "top_p": settings.llm_sum_top_p,
                "top_k": settings.llm_sum_top_k,
                "max_tokens": settings.llm_sum_max_tokens,
            }

        # Merge so that explicit request kwargs win over config defaults
        merged_overrides = {**base_overrides, **request_overrides}

        try:
            default_sampling = self.model.get_default_sampling_params()  # type: ignore[attr-defined]
        except Exception:
            from vllm import SamplingParams  # type: ignore
            default_sampling = SamplingParams()

        sampling = apply_overrides_to_sampling(default_sampling, merged_overrides)

        try:
            outputs = self.model.generate([prompt_text], sampling)  # type: ignore[arg-type]
            generated_text = outputs[0].outputs[0].text if outputs else ""
        except Exception:
            generated_text = text

        return LLMResult(
            text=generated_text or text,
            tokens_used=None,
            model_info={
                "model_name": getattr(self.model, "_model", getattr(self.model, "model", "unknown")),
                "generation_config": settings.llm_generation_config,
            },
            metadata={"task": task, "overrides": normalize_overrides(merged_overrides)},
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
        # For now, we'll pass the text directly to execute
        # The execute method will be updated to handle text enhancement
        result = self.execute(text, task="enhancement", **kwargs)
        
        # Calculate enhancement metrics
        original_tokens = len(text.split())
        enhanced_tokens = len(result.text.split())
        
        return {
            'enhanced_text': result.text,
            'original_text': text,
            'enhancement_applied': True,
            'tokens_added': max(0, enhanced_tokens - original_tokens),
            'model_info': result.model_info
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
        if asr_backend.lower() == "whisper":
            # In V1.0, whisper with erax-wow-turbo has native punctuation
            return False
        
        # Other backends (e.g., chunkformer) may need enhancement
        return True
    
    def unload(self) -> None:
        """
        Unload the LLM model and release resources.
        """
        # Implementation will be added later
        self.model = None
        self.tokenizer = None
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        Get version information for the LLM processor.
        
        Returns:
            Dictionary containing version information
        """
        return {
            "backend": "llm",
            "model_variant": "Qwen3-4B-Instruct AWQ-4bit",
            "version": "1.0.0"
        }
