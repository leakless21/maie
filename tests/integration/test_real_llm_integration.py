"""
Các bài kiểm tra tích hợp LLM thực cho MAIE.

Các bài kiểm tra này thực hiện các cuộc gọi API thực tế đến các dịch vụ LLM và yêu cầu cấu hình phù hợp.
Chúng được đánh dấu bằng @pytest.mark.real_llm và nên được chạy riêng biệt với các bài kiểm tra đơn vị.

Cách sử dụng:
    # Chạy chỉ các bài kiểm tra LLM thực
    pytest -m real_llm tests/integration/test_real_llm_integration.py

    # Chạy với đầu ra chi tiết
    pytest -m real_llm -v tests/integration/test_real_llm_integration.py

    # Chạy với cấu hình mô hình cụ thể
    LLM_TEST_MODEL_PATH=/path/to/model pytest -m real_llm tests/integration/test_real_llm_integration.py

Biến môi trường:
    LLM_TEST_MODEL_PATH: Đường dẫn đến thư mục mô hình LLM cục bộ
    LLM_TEST_API_KEY: Khóa API cho các dịch vụ LLM dựa trên đám mây
    LLM_TEST_TEMPERATURE: Nhiệt độ cho việc tạo (mặc định: 0.1)
    LLM_TEST_MAX_TOKENS: Số token tối đa để tạo (mặc định: 100)
    LLM_TEST_TIMEOUT: Thời gian chờ cho các cuộc gọi API tính bằng giây (mặc định: 30)
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pytest

from src.processors.llm.processor import LLMProcessor
from src.processors.base import LLMResult


class TestRealLLMIntegration:
    """Các bài kiểm tra tích hợp LLM thực với các cuộc gọi API thực tế."""

    @pytest.mark.real_llm
    @pytest.mark.slow
    def test_real_text_enhancement(self, real_llm_config, skip_if_no_real_llm_config):
        """Kiểm tra cải thiện văn bản với mô hình LLM thực."""
        # Tạo thư mục template tạm thời
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "templates" / "prompts"
            templates_dir.mkdir(parents=True)

            # Tạo template cải thiện văn bản
            enhancement_template = templates_dir / "text_enhancement_v1.jinja"
            enhancement_template.write_text(
                """
Hãy cải thiện đoạn văn bản sau bằng cách thêm dấu câu và viết hoa đúng cách:

{{ text_input }}

Phiên bản đã cải thiện:
"""
            )

            # Khởi tạo bộ xử lý với cấu hình thực
            processor = LLMProcessor()

            # Ghi đè cài đặt cho kiểm tra
            if real_llm_config["model_path"]:
                processor.model_path = real_llm_config["model_path"]

            # Kiểm tra với văn bản tiếng Việt đơn giản
            test_text = "xin chào thế giới đây là một bài kiểm tra không có dấu câu"

            start_time = time.time()
            result = processor.enhance_text(test_text)
            end_time = time.time()

            # Xác minh kết quả
            assert result is not None
            assert "enhanced_text" in result
            assert "enhancement_applied" in result
            assert "edit_distance" in result
            assert "edit_rate" in result
            assert "model_info" in result

            # Kiểm tra rằng cải thiện đã được áp dụng
            assert result["enhancement_applied"] is True
            assert result["enhanced_text"] != test_text
            assert len(result["enhanced_text"]) > len(test_text)

            # Kiểm tra hiệu suất (nên hoàn thành trong thời gian chờ)
            assert (end_time - start_time) < real_llm_config["timeout"]

            # Log results for manual inspection
            print(f"\nVăn bản gốc: {test_text}")
            print(f"Văn bản đã cải thiện: {result['enhanced_text']}")
            print(f"Khoảng cách chỉnh sửa: {result['edit_distance']}")
            print(f"Tỷ lệ chỉnh sửa: {result['edit_rate']:.2%}")
            print(f"Thời gian xử lý: {end_time - start_time:.2f}s")

    @pytest.mark.real_llm
    @pytest.mark.slow
    def test_real_structured_summarization(
        self, real_llm_config, skip_if_no_real_llm_config
    ):
        """Kiểm tra tóm tắt có cấu trúc với mô hình LLM thực."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "templates"
            templates_dir.mkdir()

            # Create schema file for Vietnamese meeting notes
            schema_file = templates_dir / "meeting_notes_v1.json"
            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "maxLength": 200},
                    "abstract": {"type": "string", "maxLength": 1000},
                    "main_points": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 500},
                        "minItems": 1,
                        "maxItems": 10,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 50},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                },
                "required": ["title", "abstract", "main_points", "tags"],
                "additionalProperties": False,
            }
            schema_file.write_text(json.dumps(schema, indent=2))

            # Create prompt template for Vietnamese
            prompts_dir = templates_dir / "prompts"
            prompts_dir.mkdir()
            prompt_template = prompts_dir / "meeting_notes_v1.jinja"
            prompt_template.write_text(
                """
Hãy tóm tắt đoạn ghi âm cuộc họp sau theo định dạng JSON được chỉ định:

Bản ghi âm:
{{ transcript }}

Lược đồ:
{{ schema }}

Tạo tóm tắt có cấu trúc với các trường sau:
- title: Tiêu đề ngắn gọn, mô tả
- abstract: Tóm tắt 2-3 câu
- main_points: Các điểm thảo luận chính (mảng)
- tags: Thẻ phân loại (mảng, 1-5 mục)

Chỉ trả về JSON hợp lệ khớp với lược đồ:
"""
            )

            # Initialize processor
            processor = LLMProcessor()
            if real_llm_config["model_path"]:
                processor.model_path = real_llm_config["model_path"]

            # Test with realistic Vietnamese meeting transcript
            transcript = """
            Chào buổi sáng mọi người, chào mừng đến với cuộc họp nhóm hàng tuần của chúng ta.
            Hãy bắt đầu với báo cáo tiến độ dự án. Sarah, việc phát triển giao diện người dùng thế nào?
            Sarah: Chúng ta đang đúng tiến độ cho deadline quý 1. Giao diện người dùng đã hoàn thành 80%.
            John: Tuyệt vời! Còn việc tích hợp API backend thì sao?
            Sarah: Đó được lên lịch cho tuần tới. Chúng ta đang chờ hoàn thiện lược đồ cơ sở dữ liệu.
            John: Hoàn hảo. Mike, có cập nhật gì về công việc cơ sở dữ liệu không?
            Mike: Vâng, lược đồ đã sẵn sàng. Chúng ta đã tối ưu hóa các truy vấn và thêm chỉ mục phù hợp.
            John: Xuất sắc. Có vấn đề gì cản trở hoặc lo ngại không?
            Sarah: Chúng ta có thể cần thêm nguồn lực cho việc kiểm thử.
            Mike: Tôi đồng ý, giai đoạn kiểm thử có thể mất nhiều thời gian hơn dự kiến.
            John: Tôi sẽ xem xét việc có thêm hỗ trợ QA. Hãy kết thúc ở đây.
            """

            start_time = time.time()
            result = processor.generate_summary(transcript, "meeting_notes_v1")
            end_time = time.time()

            # Verify results
            assert result is not None
            assert "summary" in result
            assert "retry_count" in result
            assert "model_info" in result

            summary = result["summary"]
            assert isinstance(summary, dict)
            assert "title" in summary
            assert "abstract" in summary
            assert "main_points" in summary
            assert "tags" in summary

            # Validate content quality
            assert len(summary["title"]) > 0
            assert len(summary["abstract"]) > 0
            assert len(summary["main_points"]) > 0
            assert len(summary["tags"]) > 0

            # Check performance
            assert (end_time - start_time) < real_llm_config["timeout"]

            # Log results for manual inspection
            print(f"\nTóm tắt được tạo:")
            print(f"Tiêu đề: {summary['title']}")
            print(f"Tóm tắt: {summary['abstract']}")
            print(f"Điểm chính: {summary['main_points']}")
            print(f"Thẻ: {summary['tags']}")
            print(f"Số lần thử lại: {result['retry_count']}")
            print(f"Thời gian xử lý: {end_time - start_time:.2f}s")

    @pytest.mark.real_llm
    @pytest.mark.slow
    def test_model_loading_and_caching(
        self, real_llm_config, skip_if_no_real_llm_config
    ):
        """Kiểm tra hành vi tải, cache và dỡ mô hình."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "templates" / "prompts"
            templates_dir.mkdir(parents=True)

            enhancement_template = templates_dir / "text_enhancement_v1.jinja"
            enhancement_template.write_text("{{ text_input }}")

            processor = LLMProcessor()
            if real_llm_config["model_path"]:
                processor.model_path = real_llm_config["model_path"]

            # Test initial state
            assert processor.model is None
            assert not processor._model_loaded

            # First call should load the model
            start_time = time.time()
            result1 = processor.enhance_text("văn bản kiểm tra 1")
            load_time = time.time() - start_time

            assert processor.model is not None
            assert processor._model_loaded
            assert result1["enhanced_text"] is not None

            # Second call should use cached model (faster)
            start_time = time.time()
            result2 = processor.enhance_text("văn bản kiểm tra 2")
            cached_time = time.time() - start_time

            # Cached call should be significantly faster
            assert cached_time < load_time
            assert result2["enhanced_text"] is not None

            # Test unloading
            processor.unload()
            assert processor.model is None
            assert not processor._model_loaded

            print(f"\nThời gian tải mô hình: {load_time:.2f}s")
            print(f"Thời gian gọi từ cache: {cached_time:.2f}s")
            print(f"Tốc độ tăng: {load_time/cached_time:.1f}x")

    @pytest.mark.real_llm
    @pytest.mark.slow
    def test_error_handling_with_real_model(
        self, real_llm_config, skip_if_no_real_llm_config
    ):
        """Kiểm tra xử lý lỗi với mô hình thực."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "templates" / "prompts"
            templates_dir.mkdir(parents=True)

            enhancement_template = templates_dir / "text_enhancement_v1.jinja"
            enhancement_template.write_text("{{ text_input }}")

            processor = LLMProcessor()
            if real_llm_config["model_path"]:
                processor.model_path = real_llm_config["model_path"]

            # Test with empty input
            result = processor.enhance_text("")
            assert result["enhanced_text"] == ""
            assert result["enhancement_applied"] is False

            # Test with very long input (should handle gracefully)
            long_text = "kiểm tra " * 10000  # Very long Vietnamese text
            result = processor.enhance_text(long_text)
            assert result["enhanced_text"] is not None
            assert len(result["enhanced_text"]) > 0

    @pytest.mark.real_llm
    @pytest.mark.slow
    def test_performance_benchmark(self, real_llm_config, skip_if_no_real_llm_config):
        """Đánh giá hiệu suất LLM với các kích thước đầu vào khác nhau."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "templates" / "prompts"
            templates_dir.mkdir(parents=True)

            enhancement_template = templates_dir / "text_enhancement_v1.jinja"
            enhancement_template.write_text("{{ text_input }}")

            processor = LLMProcessor()
            if real_llm_config["model_path"]:
                processor.model_path = real_llm_config["model_path"]

            # Test different input sizes with Vietnamese text
            test_cases = [
                ("short", "xin chào thế giới"),
                (
                    "medium",
                    "đây là một đoạn văn bản có độ dài trung bình để kiểm tra khả năng xử lý của mô hình với các kích thước đầu vào hợp lý",
                ),
                (
                    "long",
                    "đây là một đoạn văn bản dài hơn nhiều chứa nhiều từ và câu để kiểm tra cách mô hình hoạt động với các đầu vào lớn hơn có thể đại diện cho các mẫu sử dụng thực tế",
                ),
            ]

            results = {}

            for size_name, text in test_cases:
                start_time = time.time()
                result = processor.enhance_text(text)
                end_time = time.time()

                processing_time = end_time - start_time
                results[size_name] = {
                    "input_length": len(text),
                    "output_length": len(result["enhanced_text"]),
                    "processing_time": processing_time,
                    "tokens_per_second": (
                        len(result["enhanced_text"]) / processing_time
                        if processing_time > 0
                        else 0
                    ),
                }

                print(f"\nVăn bản {size_name}:")
                print(f"  Độ dài đầu vào: {len(text)} ký tự")
                print(f"  Độ dài đầu ra: {len(result['enhanced_text'])} ký tự")
                print(f"  Thời gian xử lý: {processing_time:.2f}s")
                print(
                    f"  Token mỗi giây: {results[size_name]['tokens_per_second']:.1f}"
                )

            # Verify all tests completed successfully
            for size_name, metrics in results.items():
                assert metrics["processing_time"] < real_llm_config["timeout"]
                assert metrics["output_length"] > 0
