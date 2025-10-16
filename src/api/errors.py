"""Standardized error responses and taxonomy for MAIE API."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ErrorResponse:
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details or {}}}


class ErrorCodes:
    PROCESSING_ERROR = "PROCESSING_ERROR"
    AUDIO_DECODE_ERROR = "AUDIO_DECODE_ERROR"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


__all__ = ["ErrorResponse", "ErrorCodes"]



