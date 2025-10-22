"""
Ví dụ về cách chạy các bài kiểm tra LLM thực với văn bản tiếng Việt.

Script này minh họa cách thiết lập và chạy các bài kiểm tra tích hợp LLM
với dữ liệu tiếng Việt thực tế.
"""

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import settings
from src.config.logging import get_module_logger

# Create module-bound logger for better debugging
logger = get_module_logger(__name__)


def check_pixi_availability():
    """Kiểm tra xem pixi có sẵn sàng không."""
    try:
        result = subprocess.run(
            ["pixi", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"✅ Pixi version: {version}")
            return True
        else:
            logger.info("❌ Pixi không hoạt động đúng cách")
            return False
    except FileNotFoundError:
        logger.exception("❌ Pixi không được cài đặt")
        return False
    except subprocess.TimeoutExpired:
        logger.exception("❌ Pixi không phản hồi")
        return False
    except Exception as e:
        logger.exception(f"❌ Lỗi khi kiểm tra Pixi: {e}")
        return False


def sync_dependencies():
    """Cài đặt dependencies với Pixi."""
    logger.info("🔄 Đồng bộ dependencies...")
    try:
        result = subprocess.run(
            ["pixi", "install"], cwd=Path(__file__).parent.parent, timeout=120
        )
        if result.returncode == 0:
            logger.info("✅ Dependencies đã được đồng bộ thành công")
            return True
        else:
            logger.info("❌ Lỗi khi đồng bộ dependencies")
            return False
    except subprocess.TimeoutExpired:
        logger.exception("❌ Timeout khi đồng bộ dependencies")
        return False
    except Exception as e:
        logger.exception(f"❌ Lỗi khi đồng bộ dependencies: {e}")
        return False


def check_dependencies():
    """Kiểm tra các dependencies cần thiết."""
    logger.info("🔍 Kiểm tra dependencies...")
    if not check_pixi_availability():
        logger.info("\n❌ Cần cài đặt Pixi:")
        logger.info("   curl -fsSL https://pixi.sh/install.sh | bash")
        return False
    try:
        result = subprocess.run(
            ["pixi", "run", "python", "-c", "import pytest; print('pytest available')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("✅ pytest có sẵn trong Pixi environment")
            return True
        else:
            logger.info("❌ pytest không có sẵn trong Pixi environment")
            logger.info("🔄 Thử đồng bộ dependencies...")
            if sync_dependencies():
                result = subprocess.run(
                    [
                        "pixi",
                        "run",
                        "python",
                        "-c",
                        "import pytest; print('pytest available')",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info("✅ pytest đã có sẵn sau khi đồng bộ")
                    return True
            logger.info("   Chạy thủ công: pixi install")
            return False
    except Exception as e:
        logger.exception(f"❌ Lỗi khi kiểm tra pytest: {e}")
        return False


def show_help():
    """Hiển thị hướng dẫn sử dụng script."""
    logger.info(
        "\n🚀 Vietnamese LLM Testing Script\n\nCách sử dụng:\n    # Sử dụng Pixi (khuyến nghị)\n    pixi run python examples/run_vietnamese_tests.py\n    \n    # Hoặc sử dụng Python trực tiếp\n    python examples/run_vietnamese_tests.py\n\nYêu cầu:\n    - Pixi package manager (https://pixi.sh)\n    - Dependencies đã được cài đặt: pixi install\n\nCấu hình:\n    Script này sử dụng cấu hình từ src/config/settings.py và environment variables.\n\n    Environment Variables (tùy chọn):\n    - LLM_TEST_MODEL_PATH: Đường dẫn đến mô hình LLM cục bộ\n    - LLM_TEST_API_KEY: API key cho dịch vụ LLM đám mây\n    - LLM_TEST_TEMPERATURE: Nhiệt độ cho generation (mặc định từ config)\n    - LLM_TEST_MAX_TOKENS: Số token tối đa (mặc định từ config)\n    - LLM_TEST_TIMEOUT: Thời gian chờ (mặc định: 60s)\n\n    Cấu hình từ config.py:\n    - llm_enhance_model: Mô hình cho text enhancement\n    - llm_sum_model: Mô hình cho summarization\n    - llm_enhance_temperature: Nhiệt độ cho enhancement\n    - llm_sum_temperature: Nhiệt độ cho summarization\n    - llm_enhance_max_tokens: Token tối đa cho enhancement\n    - llm_sum_max_tokens: Token tối đa cho summarization\n\nVí dụ:\n    # Sử dụng cấu hình mặc định từ config.py\n    pixi run python examples/run_vietnamese_tests.py\n\n    # Sử dụng model path tùy chỉnh\n    LLM_TEST_MODEL_PATH=/path/to/model pixi run python examples/run_vietnamese_tests.py\n\n    # Sử dụng API key\n    LLM_TEST_API_KEY=sk-123... pixi run python examples/run_vietnamese_tests.py\n\nCài đặt Pixi:\n    curl -fsSL https://pixi.sh/install.sh | bash\n\nCài đặt dependencies:\n    pixi install\n"
    )


def validate_configuration():
    """Kiểm tra và xác thực cấu hình cho các bài kiểm tra."""
    issues = []
    warnings = []
    enhance_model_path = os.getenv("LLM_TEST_MODEL_PATH") or settings.llm_enhance_model
    if enhance_model_path:
        if not Path(enhance_model_path).exists():
            issues.append(f"Enhancement model path không tồn tại: {enhance_model_path}")
        else:
            logger.info(f"✅ Enhancement model path hợp lệ: {enhance_model_path}")
    else:
        issues.append("Không có enhancement model path được cấu hình")
    sum_model_path = settings.llm_sum_model
    if sum_model_path:
        if "/" in sum_model_path and (not Path(sum_model_path).exists()):
            logger.info(f"✅ Summarization model (HuggingFace): {sum_model_path}")
        elif Path(sum_model_path).exists():
            logger.info(f"✅ Summarization model path hợp lệ: {sum_model_path}")
        else:
            warnings.append(f"Summarization model path không tồn tại: {sum_model_path}")
    api_key = os.getenv("LLM_TEST_API_KEY")
    if not enhance_model_path and (not api_key):
        issues.append("Cần cấu hình model path hoặc API key")
    if not settings.paths.templates_dir.exists():
        warnings.append(f"Templates directory không tồn tại: {settings.paths.templates_dir}")
    if not settings.paths.models_dir.exists():
        warnings.append(f"Models directory không tồn tại: {settings.paths.models_dir}")
    if settings.llm_enhance_temperature < 0 or settings.llm_enhance_temperature > 2:
        warnings.append(
            f"LLM enhancement temperature ngoài phạm vi hợp lệ: {settings.llm_enhance_temperature}"
        )
    if settings.llm_sum_temperature < 0 or settings.llm_sum_temperature > 2:
        warnings.append(
            f"LLM summarization temperature ngoài phạm vi hợp lệ: {settings.llm_sum_temperature}"
        )
    return (issues, warnings)


def main():
    """Chạy các bài kiểm tra LLM thực với cấu hình tiếng Việt."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        return 0
    logger.info("🚀 Chạy các bài kiểm tra LLM thực với văn bản tiếng Việt")
    logger.info("=" * 60)
    if not check_dependencies():
        logger.info("\n❌ Dependencies không đầy đủ. Vui lòng cài đặt trước khi chạy.")
        return 1
    logger.info("\n🔍 Kiểm tra cấu hình...")
    issues, warnings = validate_configuration()
    if warnings:
        logger.warning("\n⚠️  Cảnh báo:")
        for warning in warnings:
            logger.info(f"   - {warning}")
    if issues:
        logger.info("\n❌ Lỗi cấu hình:")
        for issue in issues:
            logger.info(f"   - {issue}")
        logger.info("\nCác cách cấu hình:")
        logger.info("  1. Đặt LLM_TEST_MODEL_PATH environment variable")
        logger.info("  2. Đặt LLM_TEST_API_KEY environment variable")
        logger.info("  3. Cấu hình llm_enhance_model trong config.py")
        logger.info("\nVí dụ:")
        logger.info("  export LLM_TEST_MODEL_PATH='/path/to/your/model'")
        logger.info("  # HOẶC")
        logger.info("  export LLM_TEST_API_KEY='sk-your-api-key-here'")
        logger.info("\nSau đó chạy lại script này.")
        return 1
    model_path = os.getenv("LLM_TEST_MODEL_PATH") or settings.llm_enhance_model
    api_key = os.getenv("LLM_TEST_API_KEY")
    logger.info("\n📋 Thông tin cấu hình:")
    if model_path:
        logger.info(f"   📁 Mô hình cục bộ: {model_path}")
        if Path(model_path).exists():
            logger.info("   ✅ Đường dẫn mô hình hợp lệ")
        else:
            logger.warning("   ⚠️  Đường dẫn mô hình không tồn tại")
    if api_key:
        logger.info(f"   🔑 API key: {api_key[:10]}...")
    logger.info("   🏗️  Cấu hình từ config.py:")
    logger.info(f"      - Environment: {settings.environment}")
    logger.debug(f"      - Debug mode: {settings.debug}")
    logger.info(f"      - Templates dir: {settings.paths.templates_dir}")
    logger.info(f"      - Models dir: {settings.paths.models_dir}")
    env = os.environ.copy()
    env.update(
        {
            "LLM_TEST_MODEL_PATH": model_path or "",
            "LLM_TEST_API_KEY": api_key or "",
            "LLM_TEST_TEMPERATURE": str(
                os.getenv("LLM_TEST_TEMPERATURE", settings.llm_enhance_temperature)
            ),
            "LLM_TEST_MAX_TOKENS": str(
                os.getenv("LLM_TEST_MAX_TOKENS", settings.llm_enhance_max_tokens or 200)
            ),
            "LLM_TEST_TIMEOUT": str(os.getenv("LLM_TEST_TIMEOUT", "60")),
            "PYTHONPATH": str(Path(__file__).parent.parent),
            "LLM_SUM_MODEL": settings.llm_sum_model,
            "LLM_SUM_TEMPERATURE": str(settings.llm_sum_temperature),
            "LLM_SUM_MAX_TOKENS": str(settings.llm_sum_max_tokens or 1000),
            "LLM_SUM_QUANTIZATION": str(settings.llm_sum_quantization or ""),
            "LLM_ENHANCE_QUANTIZATION": str(settings.llm_enhance_quantization or ""),
        }
    )
    logger.info("\n⚙️  Cấu hình LLM:")
    logger.info(f"   - Mô hình enhancement: {model_path}")
    logger.info(f"   - Mô hình summarization: {settings.llm_sum_model}")
    logger.info(f"   - Nhiệt độ enhancement: {env['LLM_TEST_TEMPERATURE']}")
    logger.info(f"   - Nhiệt độ summarization: {env['LLM_SUM_TEMPERATURE']}")
    logger.info(f"   - Token tối đa enhancement: {env['LLM_TEST_MAX_TOKENS']}")
    logger.info(f"   - Token tối đa summarization: {env['LLM_SUM_MAX_TOKENS']}")
    logger.info(f"   - Thời gian chờ: {env['LLM_TEST_TIMEOUT']}s")
    logger.info(
        f"   - GPU memory utilization: {settings.llm_enhance_gpu_memory_utilization}"
    )
    logger.info(
        f"   - Max model length enhancement: {settings.llm_enhance_max_model_len}"
    )
    logger.info(
        f"   - Max model length summarization: {settings.llm_sum_max_model_len}"
    )
    logger.info(
        f"   - Quantization enhancement: {settings.llm_enhance_quantization or 'auto-detect'}"
    )
    logger.info(
        f"   - Quantization summarization: {settings.llm_sum_quantization or 'auto-detect'}"
    )
    logger.info("\n🧪 Chạy các bài kiểm tra LLM thực...")
    cmd = [
        "pixi",
        "run",
        "pytest",
        "-m",
        "real_llm",
        "-v",
        "--tb=short",
        "tests/integration/test_real_llm_integration.py",
    ]
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            logger.info("\n✅ Tất cả các bài kiểm tra đã hoàn thành thành công!")
            logger.info("\n📊 Kết quả kiểm tra bao gồm:")
            logger.info("   - Cải thiện văn bản tiếng Việt")
            logger.info("   - Tóm tắt cuộc họp có cấu trúc")
            logger.info("   - Tải và cache mô hình")
            logger.info("   - Xử lý lỗi")
            logger.info("   - Đánh giá hiệu suất")
        else:
            logger.info(
                f"\n❌ Một số bài kiểm tra đã thất bại (mã lỗi: {result.returncode})"
            )
            return result.returncode
    except FileNotFoundError:
        logger.exception("❌ Lỗi: Không tìm thấy pytest hoặc pixi.")
        logger.exception("   Hãy cài đặt dependencies với:")
        logger.exception("   pixi install")
        return 1
    except KeyboardInterrupt:
        logger.exception("\n⏹️  Đã dừng bởi người dùng")
        return 1
    except Exception as e:
        logger.exception(f"❌ Lỗi không mong muốn: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
