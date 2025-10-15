#!/usr/bin/env python3
"""
Ví dụ về cách chạy các bài kiểm tra LLM thực với văn bản tiếng Việt.

Script này minh họa cách thiết lập và chạy các bài kiểm tra tích hợp LLM
với dữ liệu tiếng Việt thực tế.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


def check_pixi_availability():
    """Kiểm tra xem pixi có sẵn sàng không."""
    try:
        result = subprocess.run(
            ["pixi", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Pixi version: {version}")
            return True
        else:
            print("❌ Pixi không hoạt động đúng cách")
            return False
    except FileNotFoundError:
        print("❌ Pixi không được cài đặt")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Pixi không phản hồi")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra Pixi: {e}")
        return False


def sync_dependencies():
    """Cài đặt dependencies với Pixi."""
    print("🔄 Đồng bộ dependencies...")
    try:
        result = subprocess.run(
            ["pixi", "install"], cwd=Path(__file__).parent.parent, timeout=120
        )
        if result.returncode == 0:
            print("✅ Dependencies đã được đồng bộ thành công")
            return True
        else:
            print("❌ Lỗi khi đồng bộ dependencies")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Timeout khi đồng bộ dependencies")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi đồng bộ dependencies: {e}")
        return False


def check_dependencies():
    """Kiểm tra các dependencies cần thiết."""
    print("🔍 Kiểm tra dependencies...")

    # Kiểm tra pixi
    if not check_pixi_availability():
        print("\n❌ Cần cài đặt Pixi:")
        print("   curl -fsSL https://pixi.sh/install.sh | bash")
        return False

    # Kiểm tra pytest trong pixi environment
    try:
        result = subprocess.run(
            ["pixi", "run", "python", "-c", "import pytest; print('pytest available')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("✅ pytest có sẵn trong Pixi environment")
            return True
        else:
            print("❌ pytest không có sẵn trong Pixi environment")
            print("🔄 Thử đồng bộ dependencies...")
            if sync_dependencies():
                # Kiểm tra lại sau khi sync
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
                    print("✅ pytest đã có sẵn sau khi đồng bộ")
                    return True
            print("   Chạy thủ công: pixi install")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra pytest: {e}")
        return False


def show_help():
    """Hiển thị hướng dẫn sử dụng script."""
    print(
        """
🚀 Vietnamese LLM Testing Script

Cách sử dụng:
    # Sử dụng Pixi (khuyến nghị)
    pixi run python examples/run_vietnamese_tests.py
    
    # Hoặc sử dụng Python trực tiếp
    python examples/run_vietnamese_tests.py

Yêu cầu:
    - Pixi package manager (https://pixi.sh)
    - Dependencies đã được cài đặt: pixi install

Cấu hình:
    Script này sử dụng cấu hình từ src/config.py và environment variables.

    Environment Variables (tùy chọn):
    - LLM_TEST_MODEL_PATH: Đường dẫn đến mô hình LLM cục bộ
    - LLM_TEST_API_KEY: API key cho dịch vụ LLM đám mây
    - LLM_TEST_TEMPERATURE: Nhiệt độ cho generation (mặc định từ config)
    - LLM_TEST_MAX_TOKENS: Số token tối đa (mặc định từ config)
    - LLM_TEST_TIMEOUT: Thời gian chờ (mặc định: 60s)

    Cấu hình từ config.py:
    - llm_enhance_model: Mô hình cho text enhancement
    - llm_sum_model: Mô hình cho summarization
    - llm_enhance_temperature: Nhiệt độ cho enhancement
    - llm_sum_temperature: Nhiệt độ cho summarization
    - llm_enhance_max_tokens: Token tối đa cho enhancement
    - llm_sum_max_tokens: Token tối đa cho summarization

Ví dụ:
    # Sử dụng cấu hình mặc định từ config.py
    pixi run python examples/run_vietnamese_tests.py

    # Sử dụng model path tùy chỉnh
    LLM_TEST_MODEL_PATH=/path/to/model pixi run python examples/run_vietnamese_tests.py

    # Sử dụng API key
    LLM_TEST_API_KEY=sk-123... pixi run python examples/run_vietnamese_tests.py

Cài đặt Pixi:
    curl -fsSL https://pixi.sh/install.sh | bash

Cài đặt dependencies:
    pixi install
"""
    )


def validate_configuration():
    """Kiểm tra và xác thực cấu hình cho các bài kiểm tra."""
    issues = []
    warnings = []

    # Kiểm tra enhancement model path
    enhance_model_path = os.getenv("LLM_TEST_MODEL_PATH") or settings.llm_enhance_model
    if enhance_model_path:
        if not Path(enhance_model_path).exists():
            issues.append(f"Enhancement model path không tồn tại: {enhance_model_path}")
        else:
            print(f"✅ Enhancement model path hợp lệ: {enhance_model_path}")
    else:
        issues.append("Không có enhancement model path được cấu hình")

    # Kiểm tra summarization model path
    sum_model_path = settings.llm_sum_model
    if sum_model_path:
        # Kiểm tra xem có phải là HuggingFace model ID không (chứa "/")
        if "/" in sum_model_path and not Path(sum_model_path).exists():
            print(f"✅ Summarization model (HuggingFace): {sum_model_path}")
        elif Path(sum_model_path).exists():
            print(f"✅ Summarization model path hợp lệ: {sum_model_path}")
        else:
            warnings.append(f"Summarization model path không tồn tại: {sum_model_path}")

    # Kiểm tra API key
    api_key = os.getenv("LLM_TEST_API_KEY")
    if not enhance_model_path and not api_key:
        issues.append("Cần cấu hình model path hoặc API key")

    # Kiểm tra templates directory
    if not settings.templates_dir.exists():
        warnings.append(f"Templates directory không tồn tại: {settings.templates_dir}")

    # Kiểm tra models directory
    if not settings.models_dir.exists():
        warnings.append(f"Models directory không tồn tại: {settings.models_dir}")

    # Kiểm tra cấu hình LLM parameters
    if settings.llm_enhance_temperature < 0 or settings.llm_enhance_temperature > 2:
        warnings.append(
            f"LLM enhancement temperature ngoài phạm vi hợp lệ: {settings.llm_enhance_temperature}"
        )

    if settings.llm_sum_temperature < 0 or settings.llm_sum_temperature > 2:
        warnings.append(
            f"LLM summarization temperature ngoài phạm vi hợp lệ: {settings.llm_sum_temperature}"
        )

    return issues, warnings


def main():
    """Chạy các bài kiểm tra LLM thực với cấu hình tiếng Việt."""

    # Kiểm tra argument help
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        return 0

    print("🚀 Chạy các bài kiểm tra LLM thực với văn bản tiếng Việt")
    print("=" * 60)

    # Kiểm tra dependencies trước
    if not check_dependencies():
        print("\n❌ Dependencies không đầy đủ. Vui lòng cài đặt trước khi chạy.")
        return 1

    # Xác thực cấu hình
    print("\n🔍 Kiểm tra cấu hình...")
    issues, warnings = validate_configuration()

    # Hiển thị warnings
    if warnings:
        print("\n⚠️  Cảnh báo:")
        for warning in warnings:
            print(f"   - {warning}")

    # Kiểm tra issues
    if issues:
        print("\n❌ Lỗi cấu hình:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nCác cách cấu hình:")
        print("  1. Đặt LLM_TEST_MODEL_PATH environment variable")
        print("  2. Đặt LLM_TEST_API_KEY environment variable")
        print("  3. Cấu hình llm_enhance_model trong config.py")
        print("\nVí dụ:")
        print("  export LLM_TEST_MODEL_PATH='/path/to/your/model'")
        print("  # HOẶC")
        print("  export LLM_TEST_API_KEY='sk-your-api-key-here'")
        print("\nSau đó chạy lại script này.")
        return 1

    # Lấy cấu hình đã xác thực
    model_path = os.getenv("LLM_TEST_MODEL_PATH") or settings.llm_enhance_model
    api_key = os.getenv("LLM_TEST_API_KEY")

    # Hiển thị cấu hình
    print(f"\n📋 Thông tin cấu hình:")
    if model_path:
        print(f"   📁 Mô hình cục bộ: {model_path}")
        if Path(model_path).exists():
            print(f"   ✅ Đường dẫn mô hình hợp lệ")
        else:
            print(f"   ⚠️  Đường dẫn mô hình không tồn tại")
    if api_key:
        print(f"   🔑 API key: {api_key[:10]}...")

    # Hiển thị cấu hình từ settings
    print(f"   🏗️  Cấu hình từ config.py:")
    print(f"      - Environment: {settings.environment}")
    print(f"      - Debug mode: {settings.debug}")
    print(f"      - Templates dir: {settings.templates_dir}")
    print(f"      - Models dir: {settings.models_dir}")

    # Cấu hình môi trường cho tiếng Việt
    # Sử dụng cấu hình từ settings, với fallback cho test-specific values
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
            # Thêm cấu hình cho summarization model
            "LLM_SUM_MODEL": settings.llm_sum_model,
            "LLM_SUM_TEMPERATURE": str(settings.llm_sum_temperature),
            "LLM_SUM_MAX_TOKENS": str(settings.llm_sum_max_tokens or 1000),
            "LLM_SUM_QUANTIZATION": str(settings.llm_sum_quantization or ""),
            # Thêm cấu hình quantization cho enhancement model
            "LLM_ENHANCE_QUANTIZATION": str(settings.llm_enhance_quantization or ""),
        }
    )

    print(f"\n⚙️  Cấu hình LLM:")
    print(f"   - Mô hình enhancement: {model_path}")
    print(f"   - Mô hình summarization: {settings.llm_sum_model}")
    print(f"   - Nhiệt độ enhancement: {env['LLM_TEST_TEMPERATURE']}")
    print(f"   - Nhiệt độ summarization: {env['LLM_SUM_TEMPERATURE']}")
    print(f"   - Token tối đa enhancement: {env['LLM_TEST_MAX_TOKENS']}")
    print(f"   - Token tối đa summarization: {env['LLM_SUM_MAX_TOKENS']}")
    print(f"   - Thời gian chờ: {env['LLM_TEST_TIMEOUT']}s")
    print(f"   - GPU memory utilization: {settings.llm_enhance_gpu_memory_utilization}")
    print(f"   - Max model length enhancement: {settings.llm_enhance_max_model_len}")
    print(f"   - Max model length summarization: {settings.llm_sum_max_model_len}")
    print(
        f"   - Quantization enhancement: {settings.llm_enhance_quantization or 'auto-detect'}"
    )
    print(
        f"   - Quantization summarization: {settings.llm_sum_quantization or 'auto-detect'}"
    )

    # Chạy các bài kiểm tra
    print(f"\n🧪 Chạy các bài kiểm tra LLM thực...")

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
            print("\n✅ Tất cả các bài kiểm tra đã hoàn thành thành công!")
            print("\n📊 Kết quả kiểm tra bao gồm:")
            print("   - Cải thiện văn bản tiếng Việt")
            print("   - Tóm tắt cuộc họp có cấu trúc")
            print("   - Tải và cache mô hình")
            print("   - Xử lý lỗi")
            print("   - Đánh giá hiệu suất")
        else:
            print(f"\n❌ Một số bài kiểm tra đã thất bại (mã lỗi: {result.returncode})")
            return result.returncode

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy pytest hoặc pixi.")
        print("   Hãy cài đặt dependencies với:")
        print("   pixi install")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng bởi người dùng")
        return 1
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
