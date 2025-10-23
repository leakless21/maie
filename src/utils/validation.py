"""Data validation and coercion utilities for the MAIE project.

This module provides consistent data validation and coercion functions
used across the codebase to ensure data integrity and type safety.
"""

from typing import Any, List, Optional, Union
from .types import JSONDict


def coerce_optional_int(value: Any) -> Optional[int]:
    """Coerce value to optional int, handling empty strings as None.
    
    Args:
        value: Value to coerce to optional int
        
    Returns:
        Coerced integer value or None if input is None/empty
        
    Examples:
        >>> coerce_optional_int("5")
        5
        >>> coerce_optional_int("")
        None
        >>> coerce_optional_int(None)
        None
        >>> coerce_optional_int(10)
        10
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return int(stripped)
    return int(value)


def coerce_optional_str(value: Any) -> Optional[str]:
    """Coerce value to optional str, handling whitespace as None.
    
    Args:
        value: Value to coerce to optional string
        
    Returns:
        Coerced string value or None if input is None/whitespace-only
        
    Examples:
        >>> coerce_optional_str("hello")
        'hello'
        >>> coerce_optional_str("  ")
        None
        >>> coerce_optional_str(None)
        None
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return stripped
    return str(value)


def validate_range(
    value: Union[int, float], 
    min_val: Union[int, float], 
    max_val: Union[int, float], 
    field_name: str
) -> Union[int, float]:
    """Validate numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error message
        
    Returns:
        Validated value if within range
        
    Raises:
        ValueError: If value is outside the specified range
        
    Examples:
        >>> validate_range(5, 1, 10, "count")
        5
        >>> validate_range(15, 1, 10, "count")
        Traceback (most recent call last):
            ...
        ValueError: count must be between 1 and 10, got 15
    """
    if value < min_val or value > max_val:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}, got {value}")
    return value


def validate_positive(value: Union[int, float], field_name: str) -> Union[int, float]:
    """Validate value is positive.
    
    Args:
        value: Value to validate
        field_name: Name of field for error message
        
    Returns:
        Validated positive value
        
    Raises:
        ValueError: If value is not positive
        
    Examples:
        >>> validate_positive(5, "threads")
        5
        >>> validate_positive(-1, "threads")
        Traceback (most recent call last):
            ...
        ValueError: threads must be positive, got -1
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


def validate_port(port: Union[int, str]) -> int:
    """Validate port number is within valid range (1-65535).
    
    Args:
        port: Port number to validate
        
    Returns:
        Validated port number
        
    Raises:
        ValueError: If port is not in valid range
        
    Examples:
        >>> validate_port(8080)
        8080
        >>> validate_port("8080")
        8080
        >>> validate_port(70000)
        Traceback (most recent call last):
            ...
        ValueError: port must be between 1 and 65535, got 70000
    """
    port_int = int(port)
    validate_range(port_int, 1, 65535, "port")
    return port_int


def validate_choice(value: Any, choices: List[Any], field_name: str) -> Any:
    """Validate value is one of allowed choices.
    
    Args:
        value: Value to validate
        choices: List of allowed values
        field_name: Name of field for error message
        
    Returns:
        Validated value if it's in choices
        
    Raises:
        ValueError: If value is not in the allowed choices
        
    Examples:
        >>> validate_choice("http", ["http", "https"], "protocol")
        'http'
        >>> validate_choice("ftp", ["http", "https"], "protocol")
        Traceback (most recent call last):
            ...
        ValueError: protocol must be one of ['http', 'https'], got 'ftp'
    """
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}, got {repr(value)}")
    return value


def coerce_feature_list(value: Any) -> List[str]:
    """Coerce features from various input formats to List[str].
    
    Args:
        value: Features in various formats (string, list, etc.)
        
    Returns:
        List of feature strings
        
    Examples:
        >>> coerce_feature_list("a,b,c")
        ['a', 'b', 'c']
        >>> coerce_feature_list(["a", "b", "c"])
        ['a', 'b', 'c']
        >>> coerce_feature_list("")
        []
        >>> coerce_feature_list(None)
        []
    """
    if value is None:
        return []
    if isinstance(value, str):
        if value.strip() == "":
            return []
        # Split by comma and strip whitespace
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        # Convert all items to strings and filter out empty ones
        return [str(item).strip() for item in value if str(item).strip()]
    # For any other type, convert to string and split by comma
    return [str(value).strip()]


def validate_percentage(value: Union[int, float], field_name: str) -> Union[int, float]:
    """Validate value is a valid percentage (0-100).
    
    Args:
        value: Percentage value to validate
        field_name: Name of field for error message
        
    Returns:
        Validated percentage value
        
    Raises:
        ValueError: If value is not between 0 and 100
        
    Examples:
        >>> validate_percentage(50, "threshold")
        50
        >>> validate_percentage(150, "threshold")
        Traceback (most recent call last):
            ...
        ValueError: threshold must be between 0 and 100, got 150
    """
    return validate_range(value, 0, 100, field_name)


def blank_to_none(value: Any) -> Any:
    """Convert blank/empty values to None.
    
    Args:
        value: Value to check
        
    Returns:
        None if value is blank/empty, otherwise the original value
        
    Examples:
        >>> blank_to_none("")
        None
        >>> blank_to_none("  ")
        None
        >>> blank_to_none(None)
        None
        >>> blank_to_none("hello")
        'hello'
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def validate_non_empty(value: Any, field_name: str) -> Any:
    """Validate that value is not empty or None.
    
    Args:
        value: Value to validate
        field_name: Name of field for error message
        
    Returns:
        Validated non-empty value
        
    Raises:
        ValueError: If value is empty or None
        
    Examples:
        >>> validate_non_empty("hello", "name")
        'hello'
        >>> validate_non_empty("", "name")
        Traceback (most recent call last):
            ...
        ValueError: name cannot be empty
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be empty")
    if isinstance(value, str) and value.strip() == "":
        raise ValueError(f"{field_name} cannot be empty")
    if isinstance(value, (list, dict)) and len(value) == 0:
        raise ValueError(f"{field_name} cannot be empty")
    return value


def validate_list_length(
    value: List[Any], 
    min_length: Optional[int] = None, 
    max_length: Optional[int] = None, 
    field_name: str = "list"
) -> List[Any]:
    """Validate list length is within specified bounds.
    
    Args:
        value: List to validate
        min_length: Minimum allowed length (optional)
        max_length: Maximum allowed length (optional)
        field_name: Name of field for error message
        
    Returns:
        Validated list if length is within bounds
        
    Raises:
        ValueError: If list length is outside specified bounds
        
    Examples:
        >>> validate_list_length([1, 2, 3], min_length=2, max_length=5, field_name="items")
        [1, 2, 3]
        >>> validate_list_length([1], min_length=2, field_name="items")
        Traceback (most recent call last):
            ...
        ValueError: items length must be at least 2, got 1
    """
    length = len(value)
    
    if min_length is not None and length < min_length:
        raise ValueError(f"{field_name} length must be at least {min_length}, got {length}")
    
    if max_length is not None and length > max_length:
        raise ValueError(f"{field_name} length must be at most {max_length}, got {length}")
    
    return value