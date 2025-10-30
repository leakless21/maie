"""Tests for the validation module."""

import pytest
from src.utils.validation import (
    coerce_optional_int,
    coerce_optional_str,
    validate_range,
    validate_positive,
    validate_port,
    validate_choice,
    coerce_feature_list,
    validate_percentage,
    blank_to_none,
    validate_non_empty,
    validate_list_length,
)


class TestCoerceOptionalInt:
    """Tests for coerce_optional_int function."""

    def test_coerce_optional_int_with_int(self):
        """Test coerce_optional_int with integer input."""
        assert coerce_optional_int(42) == 42
        assert coerce_optional_int(0) == 0
        assert coerce_optional_int(-5) == -5

    def test_coerce_optional_int_with_string(self):
        """Test coerce_optional_int with string input."""
        assert coerce_optional_int("42") == 42
        assert coerce_optional_int("0") == 0
        assert coerce_optional_int("-5") == -5
        assert coerce_optional_int(" 42  ") == 42

    def test_coerce_optional_int_with_empty_string(self):
        """Test coerce_optional_int with empty string."""
        assert coerce_optional_int("") is None
        assert coerce_optional_int("   ") is None

    def test_coerce_optional_int_with_none(self):
        """Test coerce_optional_int with None."""
        assert coerce_optional_int(None) is None


class TestCoerceOptionalStr:
    """Tests for coerce_optional_str function."""

    def test_coerce_optional_str_with_string(self):
        """Test coerce_optional_str with string input."""
        assert coerce_optional_str("hello") == "hello"
        assert coerce_optional_str("  hello  ") == "hello"
        assert coerce_optional_str("0") == "0"

    def test_coerce_optional_str_with_empty_string(self):
        """Test coerce_optional_str with empty string."""
        assert coerce_optional_str("") is None
        assert coerce_optional_str("   ") is None

    def test_coerce_optional_str_with_none(self):
        """Test coerce_optional_str with None."""
        assert coerce_optional_str(None) is None

    def test_coerce_optional_str_with_non_string(self):
        """Test coerce_optional_str with non-string input."""
        assert coerce_optional_str(42) == "42"
        assert coerce_optional_str(0) == "0"


class TestValidateRange:
    """Tests for validate_range function."""

    def test_validate_range_valid(self):
        """Test validate_range with valid input."""
        assert validate_range(5, 1, 10, "test") == 5
        assert validate_range(1, 1, 10, "test") == 1
        assert validate_range(10, 1, 10, "test") == 10
        assert validate_range(5.5, 1.0, 10.0, "test") == 5.5

    def test_validate_range_invalid(self):
        """Test validate_range with invalid input."""
        with pytest.raises(ValueError, match="test must be between 1 and 10, got 15"):
            validate_range(15, 1, 10, "test")

        with pytest.raises(ValueError, match="test must be between 1 and 10, got 0"):
            validate_range(0, 1, 10, "test")


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_validate_positive_valid(self):
        """Test validate_positive with valid input."""
        assert validate_positive(1, "test") == 1
        assert validate_positive(5, "test") == 5
        assert validate_positive(0.1, "test") == 0.1

    def test_validate_positive_invalid(self):
        """Test validate_positive with invalid input."""
        with pytest.raises(ValueError, match="test must be positive, got 0"):
            validate_positive(0, "test")

        with pytest.raises(ValueError, match="test must be positive, got -1"):
            validate_positive(-1, "test")


class TestValidatePort:
    """Tests for validate_port function."""

    def test_validate_port_valid(self):
        """Test validate_port with valid input."""
        assert validate_port(80) == 80
        assert validate_port(8080) == 8080
        assert validate_port(1) == 1
        assert validate_port(65535) == 65535
        assert validate_port("8080") == 8080

    def test_validate_port_invalid(self):
        """Test validate_port with invalid input."""
        with pytest.raises(ValueError, match="port must be between 1 and 65535, got 0"):
            validate_port(0)

        with pytest.raises(
            ValueError, match="port must be between 1 and 65535, got 70000"
        ):
            validate_port(70000)


class TestValidateChoice:
    """Tests for validate_choice function."""

    def test_validate_choice_valid(self):
        """Test validate_choice with valid input."""
        assert validate_choice("http", ["http", "https"], "protocol") == "http"
        assert validate_choice("https", ["http", "https"], "protocol") == "https"
        assert validate_choice(1, [1, 2, 3], "number") == 1

    def test_validate_choice_invalid(self):
        """Test validate_choice with invalid input."""
        with pytest.raises(
            ValueError, match="protocol must be one of \\['http', 'https'\\], got 'ftp'"
        ):
            validate_choice("ftp", ["http", "https"], "protocol")


class TestCoerceFeatureList:
    """Tests for coerce_feature_list function."""

    def test_coerce_feature_list_with_string(self):
        """Test coerce_feature_list with string input."""
        assert coerce_feature_list("a,b,c") == ["a", "b", "c"]
        assert coerce_feature_list(" a , b , c ") == ["a", "b", "c"]
        assert coerce_feature_list("a,b,c,") == ["a", "b", "c"]

    def test_coerce_feature_list_with_list(self):
        """Test coerce_feature_list with list input."""
        assert coerce_feature_list(["a", "b", "c"]) == ["a", "b", "c"]
        assert coerce_feature_list([" a ", " b ", " c "]) == ["a", "b", "c"]
        assert coerce_feature_list(["a", "", "c"]) == ["a", "c"]

    def test_coerce_feature_list_empty(self):
        """Test coerce_feature_list with empty input."""
        assert coerce_feature_list("") == []
        assert coerce_feature_list(None) == []
        assert coerce_feature_list([]) == []


class TestValidatePercentage:
    """Tests for validate_percentage function."""

    def test_validate_percentage_valid(self):
        """Test validate_percentage with valid input."""
        assert validate_percentage(0, "test") == 0
        assert validate_percentage(50, "test") == 50
        assert validate_percentage(100, "test") == 100
        assert validate_percentage(75.5, "test") == 75.5

    def test_validate_percentage_invalid(self):
        """Test validate_percentage with invalid input."""
        with pytest.raises(ValueError, match="test must be between 0 and 100, got -1"):
            validate_percentage(-1, "test")

        with pytest.raises(ValueError, match="test must be between 0 and 100, got 101"):
            validate_percentage(101, "test")


class TestBlankToNone:
    """Tests for blank_to_none function."""

    def test_blank_to_none_with_blank(self):
        """Test blank_to_none with blank input."""
        assert blank_to_none("") is None
        assert blank_to_none("   ") is None
        assert blank_to_none(None) is None

    def test_blank_to_none_with_non_blank(self):
        """Test blank_to_none with non-blank input."""
        assert blank_to_none("hello") == "hello"
        assert blank_to_none("  hello  ") == "  hello  "


class TestValidateNonEmpty:
    """Tests for validate_non_empty function."""

    def test_validate_non_empty_valid(self):
        """Test validate_non_empty with valid input."""
        assert validate_non_empty("hello", "test") == "hello"
        assert validate_non_empty([1, 2, 3], "test") == [1, 2, 3]
        assert validate_non_empty({"key": "value"}, "test") == {"key": "value"}

    def test_validate_non_empty_invalid(self):
        """Test validate_non_empty with invalid input."""
        with pytest.raises(ValueError, match="test cannot be empty"):
            validate_non_empty("", "test")

        with pytest.raises(ValueError, match="test cannot be empty"):
            validate_non_empty([], "test")

        with pytest.raises(ValueError, match="test cannot be empty"):
            validate_non_empty({}, "test")


class TestValidateListLength:
    """Tests for validate_list_length function."""

    def test_validate_list_length_valid(self):
        """Test validate_list_length with valid input."""
        result = validate_list_length(
            [1, 2, 3], min_length=2, max_length=5, field_name="test"
        )
        assert result == [1, 2, 3]

        result = validate_list_length([1, 2], min_length=2, field_name="test")
        assert result == [1, 2]

        result = validate_list_length([1, 2, 3], max_length=5, field_name="test")
        assert result == [1, 2, 3]

    def test_validate_list_length_invalid(self):
        """Test validate_list_length with invalid input."""
        with pytest.raises(ValueError, match="test length must be at least 5, got 3"):
            validate_list_length([1, 2, 3], min_length=5, field_name="test")

        with pytest.raises(ValueError, match="test length must be at most 2, got 5"):
            validate_list_length([1, 2, 3, 4, 5], max_length=2, field_name="test")
