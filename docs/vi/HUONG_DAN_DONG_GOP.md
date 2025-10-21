# Hướng dẫn Đóng góp cho AI C500

## 📜 Tổng quan

Tài liệu này cung cấp hướng dẫn toàn diện cho các nhà phát triển muốn đóng góp vào dự án AI C500. Chúng tôi tuân theo phương pháp **Phát triển Dẫn dắt bởi Kiểm thử (TDD)** để đảm bảo chất lượng mã, độ tin cậy và khả năng bảo trì.

### Triết lý phát triển

- **Đỏ → Xanh → Tái cấu trúc**: Viết kiểm thử thất bại trước, triển khai giải pháp tối thiểu, sau đó tái cấu trúc.
- **Hành vi điều khiển bởi cấu hình**: Sử dụng biến môi trường để kiểm soát hành vi thời gian chạy.
- **Kiểm thử toàn diện**: Bao gồm kiểm thử đơn vị, tích hợp và E2E.

## 🚀 Bắt đầu

### Yêu cầu Hệ thống

#### Yêu cầu Phần cứng

Để phát triển và chạy AI C500 hiệu quả, môi trường phát triển của bạn nên đáp ứng các yêu cầu sau:

-   **CPU**: Tối thiểu 4 nhân (khuyến nghị 8+ nhân)
-   **RAM**: Tối thiểu 16GB (khuyến nghị 32GB+)
-   **Lưu trữ**: Ổ SSD tối thiểu 100GB để chứa các mô hình AI, phụ thuộc và dữ liệu kiểm thử.
-   **GPU**: Card đồ họa NVIDIA với ít nhất 16GB VRAM. Khuyến nghị 24GB+ VRAM để chạy các mô hình lớn hơn và các kiểm thử tích hợp.

#### Kiến trúc GPU được hỗ trợ

Hệ thống tương thích với các kiến trúc GPU NVIDIA sau:

-   Pascal (ví dụ: GTX 10-series, Tesla P100)
-   Turing (ví dụ: RTX 20-series, Tesla T4)
-   Ampere (ví dụ: RTX 30-series, A100)
-   Ada Lovelace (ví dụ: RTX 40-series)

### 1. Thiết lập môi trường

```bash
# Sao chép kho mã nguồn
git clone <repository-url>
cd maie

# Cài đặt tất cả phụ thuộc bằng Pixi
pixi install

# Kích hoạt môi trường phát triển
pixi shell

# Sao chép và cấu hình biến môi trường
cp .env.template .env
# Chỉnh sửa .env với cài đặt của bạn

# Tải các mô hình cần thiết
pixi run download-models
```

### 2. Chạy lần đầu

```bash
# Khởi động cả máy chủ API và worker
./scripts/dev.sh

# Xác minh cài đặt
curl http://localhost:8000/health

# Chạy kiểm thử khói
pixi run test -m "unit and not slow"
```

## 💻 Quy trình phát triển

### 1. Tạo nhánh

Tạo một nhánh mới từ `main` cho tính năng hoặc sửa lỗi của bạn.

```bash
git checkout -b feat/ten-tinh-nang
```

### 2. Phát triển Dẫn dắt bởi Kiểm thử (TDD)

Đây là cốt lõi của quy trình làm việc của chúng tôi.

**a. Viết một kiểm thử thất bại (Đỏ)**

Trước khi viết bất kỳ mã triển khai nào, hãy viết một kiểm thử xác định hành vi mong muốn. Kiểm thử này ban đầu sẽ thất bại.

```python
# tests/unit/test_new_feature.py
import pytest
from src.processors.new_feature import NewProcessor

def test_new_feature_basic_functionality():
    """Kiểm thử chức năng cơ bản của tính năng mới."""
    processor = NewProcessor()
    result = processor.process("test input")
    assert result["status"] == "success"
```

**b. Viết mã tối thiểu để vượt qua kiểm thử (Xanh)**

Triển khai lượng mã tối thiểu cần thiết để làm cho kiểm thử thành công.

```python
# src/processors/new_feature.py
class NewProcessor:
    def process(self, input_text: str) -> dict:
        return {
            "status": "success",
            "output": f"Đã xử lý: {input_text}"
        }
```

**c. Tái cấu trúc (Refactor)**

Bây giờ kiểm thử đã thành công, bạn có thể tái cấu trúc mã của mình để cải thiện thiết kế, hiệu suất hoặc khả năng đọc mà không thay đổi hành vi của nó. Chạy lại các kiểm thử để đảm bảo không có gì bị hỏng.

### 3. Kiểm tra chất lượng mã

Trước khi gửi, hãy đảm bảo mã của bạn đáp ứng các tiêu chuẩn của chúng tôi.

```bash
# Định dạng mã
pixi run format

# Kiểm tra lỗi
pixi run lint

# Chạy bộ kiểm thử đầy đủ
pixi run test
```

## 🧪 Chiến lược kiểm thử

Chúng tôi phân loại các kiểm thử để quản lý sự phức tạp và tốc độ thực thi.

### Danh mục kiểm thử

| Danh mục | Mục đích | Tốc độ | Phụ thuộc | Đánh dấu |
|---|---|---|---|---|
| **Đơn vị** | Kiểm thử các thành phần riêng lẻ cô lập | Nhanh | Mock/fake | `@pytest.mark.unit` |
| **Tích hợp** | Kiểm thử tương tác thành phần với thư viện thực | Trung bình | Thư viện thực | `@pytest.mark.integration` |
| **E2E** | Kiểm thử quy trình hoàn chỉnh từ đầu đến cuối | Chậm | Hệ thống đầy đủ | `@pytest.mark.e2e` |
| **GPU** | Yêu cầu phần cứng GPU | Chậm | GPU | `@pytest.mark.gpu` |

### Chạy kiểm thử

```bash
# Chạy tất cả kiểm thử
pytest

# Chỉ chạy kiểm thử đơn vị (nhanh)
pytest -m "unit"

# Chạy kiểm thử tích hợp
pytest -m "integration"

# Chạy một tệp kiểm thử cụ thể
pytest tests/unit/test_llm_processor.py
```

### Viết kiểm thử

- **Cấu trúc Sắp xếp-Hành động-Khẳng định (Arrange-Act-Assert)**: Giữ cho các kiểm thử của bạn có cấu trúc và dễ đọc.
- **Mock hiệu quả**: Sử dụng `pytest-mock` và các fixture để cô lập các thành phần khỏi các phụ thuộc bên ngoài (API, cơ sở dữ liệu, GPU).
- **Tên mô tả**: Tên kiểm thử phải mô tả rõ ràng những gì nó đang kiểm thử.

```python
# Ví dụ về cấu trúc kiểm thử
def test_descriptive_function_name():
    # Sắp xếp - Thiết lập dữ liệu kiểm thử và mock
    processor = AudioProcessor()
    mock_audio = "test_audio.wav"

    # Hành động - Thực thi hàm được kiểm thử
    result = processor.process_audio(mock_audio)

    # Khẳng định - Xác minh hành vi dự kiến
    assert result["status"] == "success"
```

##  Git Workflow

### Hướng dẫn Commit

Chúng tôi sử dụng **Commits Quy ước** để làm cho lịch sử Git có thể đọc được.

**Định dạng:** `type(scope): description`

- **types**: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`
- **scope**: Tên thành phần bị ảnh hưởng (ví dụ: `asr`, `api`, `worker`)

**Ví dụ:**

```
feat(asr): thêm Whisper backend với bộ lọc VAD
fix(api): xử lý lỗi tải lên tệp một cách duyên dáng
docs: cập nhật tài liệu quy trình phát triển
```

### Quy trình Yêu cầu Kéo (Pull Request)

1.  **Mô tả rõ ràng**: Giải thích những gì, tại sao và cách thức các thay đổi hoạt động.
2.  **Bao phủ kiểm thử**: Đảm bảo tất cả mã mới đều có kiểm thử.
3.  **Tài liệu**: Cập nhật bất kỳ tài liệu nào có liên quan.
4.  **CI/CD**: Đảm bảo tất cả các kiểm tra tự động đều thành công.
