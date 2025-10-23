"""Data sanitization utilities for the MAIE project.

This module provides security-focused sanitization functions for user input,
file names, and other data to prevent common vulnerabilities.
"""

import re
import os
import unicodedata
from pathlib import Path
from typing import Any, List, Set, Optional
from .types import ALLOWED_FILE_EXTENSIONS, ALLOWED_MIME_TYPES, FORBIDDEN_PATH_CHARS, MAX_PATH_LENGTH


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename to sanitize
        
    Returns:
        Sanitized filename safe for use
        
    Examples:
        >>> sanitize_filename("../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("normal_file.txt")
        'normal_file.txt'
    """
    if not filename:
        return "unnamed_file"
    
    # Remove path traversal attempts
    filename = filename.replace("../", "").replace("..\\", "")
    
    # Remove forbidden characters
    for char in FORBIDDEN_PATH_CHARS:
        filename = filename.replace(char, "_")
    
    # Limit length
    if len(filename) > MAX_PATH_LENGTH:
        name, ext = os.path.splitext(filename)
        ext_len = len(ext)
        filename = name[:MAX_PATH_LENGTH - ext_len] + ext
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip().strip('.')
    
    # Ensure filename is not empty after sanitization
    if not filename:
        filename = "unnamed_file"
    
    return filename


def sanitize_metadata(metadata: Any, _seen: Optional[Set[int]] = None) -> Any:
    """Recursively sanitize metadata for serialization.
    
    Args:
        metadata: Metadata to sanitize (can be any type)
        _seen: Internal parameter to track circular references
        
    Returns:
        Sanitized metadata safe for serialization
        
    Examples:
        >>> sanitize_metadata({"key": "value", "nested": {"subkey": "subvalue"}})
        {'key': 'value', 'nested': {'subkey': 'subvalue'}}
        >>> sanitize_metadata({"key": None})
        {'key': None}
    """
    if _seen is None:
        _seen = set()
    
    # Handle circular references
    if id(metadata) in _seen:
        return "<Circular Reference>"
    
    if isinstance(metadata, dict):
        _seen.add(id(metadata))
        sanitized = {}
        for key, value in metadata.items():
            # Sanitize keys too
            clean_key = sanitize_text(str(key), preserve_newlines=False) if key is not None else key
            sanitized[clean_key] = sanitize_metadata(value, _seen)
        _seen.remove(id(metadata))
        return sanitized
    elif isinstance(metadata, list):
        _seen.add(id(metadata))
        sanitized = [sanitize_metadata(item, _seen) for item in metadata]
        _seen.remove(id(metadata))
        return sanitized
    elif isinstance(metadata, str):
        return sanitize_text(metadata)
    elif isinstance(metadata, (int, float, bool)) or metadata is None:
        return metadata
    else:
        # Handle Pydantic models and other objects with model_dump method
        if hasattr(metadata, 'model_dump') and callable(getattr(metadata, 'model_dump')):
            try:
                return sanitize_metadata(metadata.model_dump(), _seen)
            except Exception:
                pass
        # Convert other types to string and sanitize
        return sanitize_text(str(metadata))


def sanitize_text(text: str, *, preserve_newlines: bool = False) -> str:
    """Sanitize text content by removing/normalizing problematic characters.
    
    Args:
        text: Text to sanitize
        preserve_newlines: Whether to preserve newline characters
        
    Returns:
        Sanitized text
        
    Examples:
        >>> sanitize_text("Hello<script>alert('xss')</script>World")
        "Helloalert('xss')World"
        >>> sanitize_text("Line1\\nLine2", preserve_newlines=True)
        'Line1\\nLine2'
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Remove potentially dangerous HTML tags and JavaScript
    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove event handlers
    text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    # Remove javascript: URLs
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove or replace problematic characters
    if not preserve_newlines:
        # Replace newlines with spaces
        text = re.sub(r'[\r\n\t]+', ' ', text)
    else:
        # Only normalize newlines
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters (except newlines if preserving them)
    if preserve_newlines:
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    else:
        text = ''.join(c for c in text if ord(c) >= 32)
    
    return text


def validate_file_extension(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """Validate file extension against allowed list.
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: List of allowed extensions (if None, uses default)
        
    Returns:
        True if extension is allowed, False otherwise
        
    Examples:
        >>> validate_file_extension("test.mp3")
        True
        >>> validate_file_extension("test.exe")
        False
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_FILE_EXTENSIONS
    
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


def validate_mime_type(mime_type: str, allowed_mime_types: Optional[List[str]] = None) -> bool:
    """Validate MIME type against allowed list.
    
    Args:
        mime_type: MIME type to validate
        allowed_mime_types: List of allowed MIME types (if None, uses default)
        
    Returns:
        True if MIME type is allowed, False otherwise
        
    Examples:
        >>> validate_mime_type("audio/mpeg")
        True
        >>> validate_mime_type("application/exe")
        False
    """
    if allowed_mime_types is None:
        allowed_mime_types = ALLOWED_MIME_TYPES
    
    return mime_type in allowed_mime_types


def normalize_phone_number(phone: str) -> str:
    """Normalize phone number format.
    
    Args:
        phone: Phone number to normalize
        
    Returns:
        Normalized phone number
        
    Examples:
        >>> normalize_phone_number("(555) 123-4567")
        '5551234567'
        >>> normalize_phone_number("555.123.4567")
        '551234567'
    """
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Handle international numbers (starting with + or 00)
    if digits_only.startswith('+'):
        digits_only = digits_only[1:]
    elif digits_only.startswith('00'):
        digits_only = digits_only[2:]
    
    # For US numbers, remove leading '1' if it exists and there are 11 digits
    if len(digits_only) == 11 and digits_only.startswith('1'):
        digits_only = digits_only[1:]
    
    return digits_only


def sanitize_path(path: str) -> str:
    """Sanitize a file path to prevent directory traversal attacks.
    
    Args:
        path: Path to sanitize
        
    Returns:
        Sanitized path
        
    Examples:
        >>> sanitize_path("../../etc/passwd")
        'etc/passwd'
        >>> sanitize_path("/safe/path/file.txt")
        'safe/path/file.txt'
    """
    # Convert to Path object for normalization
    path_obj = Path(path)
    
    # Convert to string and remove parent directory references
    path_str = str(path_obj).replace('../', '').replace('..\\', '').replace('..', '')
    
    # Remove leading slashes to prevent absolute path issues
    path_str = path_str.lstrip('/\\')
    
    # Sanitize each component
    components = path_str.split('/')
    sanitized_components = [sanitize_filename(comp) for comp in components if comp]
    
    return '/'.join(sanitized_components)


def sanitize_url(url: str) -> str:
    """Sanitize URL to prevent malicious schemes or content.
    
    Args:
        url: URL to sanitize
        
    Returns:
        Sanitized URL
        
    Examples:
        >>> sanitize_url("javascript:alert('xss')")
        ''
        >>> sanitize_url("https://example.com")
        'https://example.com'
    """
    # Check for dangerous schemes
    dangerous_schemes = ['javascript:', 'data:', 'vbscript:', 'file:', 'ftp:']
    
    lower_url = url.lower().strip()
    
    for scheme in dangerous_schemes:
        if lower_url.startswith(scheme):
            return ""
    
    # Basic URL validation - allow http, https, and relative URLs
    if not (lower_url.startswith(('http://', 'https://', '/', '#', '?', '.')) or '://' not in lower_url):
        # If it's not a valid scheme, return empty
        return ""
    
    return url


def remove_control_characters(text: str, allowed_chars: Optional[List[str]] = None) -> str:
    """Remove control characters from text, preserving allowed ones.
    
    Args:
        text: Text to clean
        allowed_chars: List of allowed control characters (e.g., ['\n', '\t'])
        
    Returns:
        Text with control characters removed
        
    Examples:
        >>> remove_control_characters("Hello\x00World")
        'HelloWorld'
        >>> remove_control_characters("Line1\\nLine2", allowed_chars=['\\n'])
        'Line1\\nLine2'
    """
    if allowed_chars is None:
        allowed_chars = []
    
    # Convert allowed_chars to set of ordinals for faster lookup
    allowed_ordinals = {ord(c) for c in allowed_chars if len(c) == 1}
    
    result = []
    for char in text:
        if ord(char) < 32 and ord(char) not in allowed_ordinals:
            continue  # Skip control character
        result.append(char)
    
    return ''.join(result)


def validate_and_sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Validate and sanitize a filename with additional constraints.
    
    Args:
        filename: Filename to validate and sanitize
        max_length: Maximum allowed filename length
        
    Returns:
        Validated and sanitized filename
        
    Raises:
        ValueError: If filename doesn't have an allowed extension
    """
    # First sanitize the filename
    sanitized = sanitize_filename(filename)
    
    # Validate extension
    if not validate_file_extension(sanitized):
        raise ValueError(f"File extension not allowed: {Path(sanitized).suffix}")
    
    # Limit length
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        name = name[:max_length - len(ext)]
        sanitized = name + ext
    
    return sanitized