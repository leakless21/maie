# Hướng dẫn sử dụng AI C500

**AI C500 (Hệ thống Phân tích Âm thanh Thông minh)** cung cấp API REST để xử lý tệp âm thanh với các tính năng nâng cao được hỗ trợ bởi AI bao gồm chuyển đổi văn bản, tóm tắt và tăng cường nội dung.

## ⚡ Khởi động nhanh

### Xử lý âm thanh cơ bản

**Tải lên và xử lý tệp âm thanh với cài đặt mặc định:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/your/audio.mp3'
```

**Phản hồi:**
```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PENDING"
}
```

**Kiểm tra trạng thái xử lý:**

```bash
curl -X GET 'http://localhost:8000/v1/status/c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b' \
  -H 'X-API-Key: your_api_key_here'
```

### Khởi động nhanh với Python

```python
import requests
import time

def process_audio_file(file_path, api_key):
    """Xử lý tệp âm thanh với API AI C500"""

    # Gửi để xử lý
    url = "http://localhost:8000/v1/process"
    headers = {"X-API-Key": api_key}

    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code != 202:
        print(f"Lỗi: {response.text}")
        return None

    task_id = response.json()["task_id"]
    print(f"Bắt đầu xử lý. Task ID: {task_id}")

    # Thăm dò để hoàn thành
    status_url = f"http://localhost:8000/v1/status/{task_id}"
    while True:
        response = requests.get(status_url, headers=headers)
        status_data = response.json()

        if status_data["status"] == "COMPLETE":
            return status_data["results"]
        elif status_data["status"] == "FAILED":
            print(f"Xử lý thất bại: {status_data}")
            return None

        print(f"Trạng thái: {status_data['status']}")
        time.sleep(5)  # Đợi 5 giây trước khi kiểm tra tiếp theo

# Cách sử dụng
results = process_audio_file("meeting_audio.wav", "your_api_key_here")
if results:
    print("Bản ghi:", results.get("clean_transcript"))
    print("Tóm tắt:", results.get("summary"))
```

## 🕵️ Xác thực

Tất cả các điểm cuối xử lý đều yêu cầu xác thực bằng khóa API trong tiêu đề yêu cầu:

```
X-API-Key: your_api_key_here
```

**Quy tắc xác thực:**

- Khóa API phải được cung cấp trong tiêu đề `X-API-Key` (không phân biệt chữ hoa thường)
- Độ dài khóa tối thiểu: 32 ký tự
- Sử dụng so sánh an toàn về thời gian để ngăn chặn tấn công thời gian
- Nhiều khóa dự phòng được hỗ trợ để luân chuyển khóa

**Ví dụ yêu cầu:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@audio.mp3'
```

## ⚙️ Các điểm cuối API

### POST /v1/process

Gửi tệp âm thanh để xử lý không đồng bộ với các tính năng và tham số có thể cấu hình.

#### Mô tả

Điểm cuối này chấp nhận dữ liệu biểu mẫu đa phần chứa tệp âm thanh và tham số xử lý. Yêu cầu được xếp hàng để xử lý không đồng bộ, trả về ngay lập tức với ID tác vụ để theo dõi trạng thái.

**Định dạng âm thanh được hỗ trợ:**

- WAV (`audio/wav`, `audio/wave`, `audio/x-wav`)
- MP3 (`audio/mpeg`)
- M4A (`audio/mp4`, `audio/x-m4a`)
- FLAC (`audio/flac`)

**Giới hạn kích thước tệp:**

- Kích thước tệp tối đa: 500MB (có thể cấu hình qua `max_file_size_mb`)
- Các tệp vượt quá giới hạn bị từ chối trong quá trình tải lên phát trực tuyến

#### Tham số

| Tham số     | Loại   | Bắt buộc    | Mô tả                                                          |
| ------------- | ------ | ----------- | -------------------------------------------------------------------- |
| `file`        | binary | Có         | Tệp âm thanh để xử lý                                                |
| `features`    | array  | Không          | Danh sách các đầu ra mong muốn (mặc định: `["clean_transcript", "summary"]`)
| `template_id` | string | Điều kiện | ID mẫu cho định dạng tóm tắt (bắt buộc nếu `summary` trong features)   |
| `asr_backend` | string | Không          | Lựa chọn backend ASR (mặc định: `"chunkformer"`)                         |

**Tùy chọn tính năng:**

- `raw_transcript` - Bản ghi ASR thô không làm sạch
- `clean_transcript` - Bản ghi đã xử lý với giảm nhiễu
- `summary` - Tóm tắt có cấu trúc sử dụng định dạng mẫu
- `enhancement_metrics` - Số liệu chất lượng âm thanh và xử lý

**Tùy chọn backend ASR:**

- `chunkformer` - Mô hình ASR ChunkFormer (mặc định)
- `whisper` - Mô hình Whisper của OpenAI

#### Ví dụ

**Xử lý cơ bản (Cài đặt mặc định):**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/audio.mp3' \
  -F 'features=clean_transcript' \
  -F 'features=summary' \
  -F 'template_id=meeting_notes_v1'
```

**Chỉ bản ghi thô:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/audio.wav' \
  -F 'features=raw_transcript'
```

**Tất cả tính năng với backend ChunkFormer:**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: your_api_key_here' \
  -F 'file=@/path/to/audio.m4a' \
  -F 'features=raw_transcript' \
  -F 'features=clean_transcript' \
  -F 'features=summary' \
  -F 'features=enhancement_metrics' \
  -F 'template_id=meeting_notes_v1' \
  -F 'asr_backend=chunkformer'
```

**Ví dụ máy khách Python:**

```python
import requests

def process_audio(file_path, api_key, features=None, template_id=None, asr_backend="whisper"):
    url = "http://localhost:8000/v1/process"
    headers = {"X-API-Key": api_key}

    # Đặt tính năng mặc định nếu không được cung cấp
    if features is None:
        features = ["clean_transcript", "summary"]

    # Chuẩn bị dữ liệu biểu mẫu đa phần
    # 'data' phải là một danh sách các tuple để hỗ trợ nhiều giá trị cho cùng một khóa 'features'
    form_data = [("features", f) for f in features]
    if template_id:
        form_data.append(("template_id", template_id))
    if asr_backend != "whisper":
        form_data.append(("asr_backend", asr_backend))

    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(url, headers=headers, files=files, data=form_data)
    
    return response.json()

# Cách sử dụng
result = process_audio(
    file_path="meeting_audio.wav",
    api_key="your_api_key_here",
    features=["clean_transcript", "summary"],
    template_id="meeting_notes_v1"
)
print(f"Task ID: {result['task_id']}")
```

#### Phản hồi

**202 Accepted**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "PENDING"
}
```

### GET /v1/status/{task_id}

Truy xuất trạng thái hiện tại và kết quả của tác vụ xử lý.

#### Mô tả

Kiểm tra trạng thái xử lý của tác vụ đã gửi trước đó. Trả về thông tin toàn diện bao gồm trạng thái xử lý, kết quả (khi hoàn thành) và siêu dữ liệu để tái tạo.

**Trạng thái tác vụ:**

- `PENDING` - Tác vụ được xếp hàng, đang chờ xử lý
- `PREPROCESSING` - Tiền xử lý âm thanh đang diễn ra
- `PROCESSING_ASR` - Chuyển đổi ASR đang chạy
- `PROCESSING_LLM` - Xử lý LLM (tóm tắt/tăng cường) đang chạy
- `COMPLETE` - Xử lý hoàn thành thành công
- `FAILED` - Xử lý thất bại với lỗi

#### Ví dụ

```bash
curl -X GET 'http://localhost:8000/v1/status/c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b' \
  -H 'X-API-Key: your_api_key_here'
```

**Ví dụ máy khách Python:**

```python
import requests

def get_task_status(task_id, api_key):
    url = f"http://localhost:8000/v1/status/{task_id}"
    headers = {"X-API-Key": api_key}

    response = requests.get(url, headers=headers)
    return response.json()

# Cách sử dụng
status = get_task_status(
    task_id="c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
    api_key="your_api_key_here"
)
print(f"Status: {status['status']}")
```

#### Phản hồi

**200 OK - Xử lý hoàn thành**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
  "submitted_at": "2025-10-20T07:18:17.637Z",
  "completed_at": "2025-10-20T07:20:45.123Z",
  "versions": {
    "pipeline_version": "1.0.0",
    "asr_backend": {
      "name": "whisper",
      "model_variant": "erax-wow-turbo",
      "model_path": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
      "checkpoint_hash": "a1b2c3d4e5f6...",
      "compute_type": "int8_float16",
      "decoding_params": {
        "beam_size": 5,
        "vad_filter": true
      }
    },
    "llm": {
      "name": "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
      "checkpoint_hash": "f6g7h8i9j0k1...",
      "quantization": "awq-4bit",
      "thinking": false,
      "reasoning_parser": null,
      "structured_output": {
        "backend": "json_schema",
        "schema_id": "meeting_notes_v1",
        "schema_hash": "sha256:..."
      },
      "decoding_params": {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.05
      }
    }
  },
  "metrics": {
    "input_duration_seconds": 2701.3,
    "processing_time_seconds": 162.8,
    "rtf": 0.06,
    "vad_coverage": 0.88,
    "asr_confidence_avg": 0.91,
    "edit_rate_cleaning": 0.15
  },
  "results": {
    "raw_transcript": "Cuộc họp ngày 4 tháng 10...",
    "clean_transcript": "Cuộc họp ngày 4 tháng 10 đã đề cập...",
    "summary": {
      "title": "Cuộc họp lập kế hoạch ngân sách Q4",
      "abstract": "Tổng kết đề xuất ngân sách quý 4, tập trung vào phân bổ cho marketing và R&D.",
      "main_points": [
        "Đã phê duyệt ngân sách cho các sáng kiến Q4",
        "Đã thảo luận kế hoạch tuyển dụng mới",
        "Đã đặt lịch cho việc ra mắt sản phẩm"
      ],
      "tags": ["Tài chính", "Ngân sách", "Lập kế hoạch"]
    }
  }
}
```

### GET /v1/models

Truy xuất thông tin về các mô hình xử lý âm thanh và backend có sẵn.

#### Ví dụ

```bash
curl -X GET 'http://localhost:8000/v1/models' \
  -H 'X-API-Key: your_api_key_here'
```

#### Phản hồi

```json
{
  "models": [
    {
      "id": "whisper",
      "name": "Whisper Backend",
      "description": "ASR backend sử dụng whisper",
      "type": "ASR",
      "version": "1.0",
      "supported_languages": ["en", "vi", "zh", "ja", "ko"]
    },
    {
      "id": "chunkformer",
      "name": "ChunkFormer Backend",
      "description": "ASR backend sử dụng chunkformer",
      "type": "ASR",
      "version": "1.0",
      "supported_languages": ["en", "vi", "zh", "ja", "ko"]
    }
  ]
}
```

### GET /v1/templates

Truy xuất thông tin về các mẫu xử lý có sẵn để định dạng đầu ra có cấu trúc.

#### Ví dụ

```bash
curl -X GET 'http://localhost:8000/v1/templates' \
  -H 'X-API-Key: your_api_key_here'
```

#### Phản hồi

```json
{
  "templates": [
    {
      "id": "meeting_notes_v1",
      "name": "Meeting Notes v1",
      "description": "Định dạng có cấu trúc cho bản ghi cuộc họp",
      "schema_url": "/v1/templates/meeting_notes_v1/schema",
      "parameters": {}
    },
    {
      "id": "interview_transcript_v1",
      "name": "Interview Transcript v1",
      "description": "Định dạng cho bản ghi phỏng vấn",
      "schema_url": "/v1/templates/interview_transcript_v1/schema",
      "parameters": {}
    }
  ]
}
```

## 📝 Hệ thống Mẫu

Hệ thống mẫu của AI C500 cung cấp cách linh hoạt để định dạng đầu ra xử lý âm thanh. Mẫu xác định cấu trúc và định dạng của bản tóm tắt được tạo, cho phép các định dạng đầu ra khác nhau cho các trường hợp sử dụng khác nhau.

### Ví dụ: Mẫu Ghi chú cuộc họp

```json
{
  "id": "meeting_notes_v1",
  "name": "Ghi chú cuộc họp v1",
  "description": "Định dạng có cấu trúc cho bản ghi cuộc họp",
  "version": "1.0",
  "schema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "Tiêu đề cuộc họp ngắn gọn"
      },
      "main_points": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Các điểm chính được thảo luận"
      },
      "decisions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Các quyết định được đưa ra"
      }
    },
    "required": ["title", "main_points"]
  }
}
```

##  libraries Thư viện máy khách

### Máy khách Python (đồng bộ)

```python
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class C500Client:
    """Máy khách API AI C500 cơ bản sử dụng requests"""

    base_url: str = "http://localhost:8000"
    api_key: str = ""

    def _get_headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    def process_audio(self, file_path: str, features: Optional[List[str]] = None,
                     template_id: Optional[str] = None,
                     asr_backend: str = "whisper") -> Dict[str, Any]:
        """Gửi tệp âm thanh để xử lý"""

        url = f"{self.base_url}/v1/process"
        headers = self._get_headers()

        # Chuẩn bị dữ liệu biểu mẫu multipart.
        # Phải sử dụng một danh sách các tuple để gửi nhiều giá trị cho khóa 'features'.
        form_data = [("asr_backend", asr_backend)]
        if features:
            for feature in features:
                form_data.append(("features", feature))
        if template_id:
            form_data.append(("template_id", template_id))

        with open(file_path, "rb") as audio_file:
            files = {"file": audio_file}
            response = requests.post(url, headers=headers, files=files, data=form_data)
            response.raise_for_status()

        return response.json()

    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Lấy trạng thái xử lý"""

        url = f"{self.base_url}/v1/status/{task_id}"
        headers = self._get_headers()

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def wait_for_completion(self, task_id: str, timeout: int = 300,
                          check_interval: int = 5) -> Dict[str, Any]:
        """Đợi xử lý hoàn thành"""

        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = self.get_status(task_id)
            status = status_response["status"]

            if status == "COMPLETE":
                return status_response
            elif status == "FAILED":
                raise Exception(f"Xử lý thất bại: {status_response}")

            time.sleep(check_interval)

        raise TimeoutError(f"Hết thời gian chờ xử lý sau {timeout} giây")

# Cách sử dụng
client = C500Client(api_key="your_api_key_here")
result = client.process_audio(
    "meeting.wav",
    features=["clean_transcript", "summary"],
    template_id="meeting_notes_v1"
)
task_id = result["task_id"]
final_result = client.wait_for_completion(task_id)
print(f"Tóm tắt: {final_result['results']['summary']}")
```

## 🚨 Xử lý lỗi

API sử dụng các mã trạng thái HTTP tiêu chuẩn và định dạng lỗi JSON nhất quán.

**Ví dụ: 413 Payload Too Large**

```json
{
  "detail": "File too large: 150.5MB (max 100MB)"
}
```

**Ví dụ: 429 Too Many Requests**

```json
{
  "detail": "Queue is full. Please try again later."
}
```

## 👍 Các phương pháp tốt nhất

- **Tối ưu hóa tải lên** : Sử dụng định dạng không mất dữ liệu (WAV, FLAC) để có chất lượng tốt nhất.
- **Xử lý lỗi** : Triển khai logic thử lại với backoff theo cấp số nhân cho lỗi `429`.
- **Giám sát tác vụ** : Thăm dò điểm cuối `/v1/status/{task_id}` định kỳ để nhận cập nhật.
- **Bảo mật** : Luân chuyển khóa API thường xuyên và lưu trữ chúng một cách an toàn.

## ❓ Khắc phục sự cố

- **"Hàng đợi đầy" (Lỗi 429)**: Đợi và thử lại yêu cầu của bạn. Giảm tốc độ gửi nếu lỗi vẫn tiếp diễn.
- **"Không tìm thấy tác vụ" (Lỗi 404)**: Xác minh `task_id` là chính xác. Các tác vụ có thể hết hạn sau một khoảng thời gian có thể cấu hình (mặc định là 24 giờ).
- **"Tệp quá lớn" (Lỗi 413)**: Kiểm tra kích thước tệp của bạn so với giới hạn đã định cấu hình. Cân nhắc nén âm thanh hoặc chia các bản ghi dài.
