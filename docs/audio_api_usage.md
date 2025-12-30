# Hướng dẫn nhanh API xử lý audio

Tài liệu này mô tả cách gọi API REST để nhận diện tiếng nói, tóm tắt, hiệu đính/
sửa lỗi văn bản và các tính năng liên quan.

## Thông tin chung

- Base URL: `http://localhost:8000`
- Phiên bản: mọi endpoint đều có tiền tố `/v1/`

## Xác thực

Gửi khóa API trong header cho các endpoint cần bảo vệ:

```
X-API-Key: <your_api_key>
```

## Luồng chuẩn

1. Gửi file audio qua `/v1/process` và nhận `task_id` (xử lý bất đồng bộ).
2. Gọi `/v1/status/{task_id}` cho tới khi trạng thái là `COMPLETE` hoặc `FAILED`.
3. Tùy chọn: xem danh sách model và template để chọn backend/định dạng tóm tắt.

## Endpoint chính

### POST /v1/process

Tải audio để xử lý bất đồng bộ.

- Headers: `Content-Type: multipart/form-data`, `X-API-Key: <key>`
- Trường form:
  - `file` (bắt buộc): WAV, MP3, M4A, FLAC. Tối đa 500MB (có thể cấu hình).
  - `features` (tùy chọn, lặp lại): `raw_transcript`, `clean_transcript`, `summary`,
    `enhancement_metrics`. Mặc định: `clean_transcript`, `summary`.
  - `template_id` (bắt buộc nếu có `summary`): ID template tóm tắt.
  - `asr_backend` (tùy chọn): `chunkformer` (mặc định) hoặc `whisper`.
  - `enable_diarization` (tùy chọn): `true` để gán nhãn người nói.
  - `enable_vad` (tùy chọn): bật Voice Activity Detection, mặc định theo server.
  - `vad_threshold` (tùy chọn): số thực 0.0–1.0, mặc định 0.5.

**cURL**

```bash
curl -X POST 'http://localhost:8000/v1/process' \
  -H 'X-API-Key: $API_KEY' \
  -F 'file=@/path/to/audio.mp3' \
  -F 'features=clean_transcript' \
  -F 'features=summary' \
  -F 'template_id=meeting_notes_v1' \
  -F 'asr_backend=chunkformer' \
  -F 'enable_diarization=true'
```

**Python**

```python
import requests

url = "http://localhost:8000/v1/process"
headers = {"X-API-Key": "<your_api_key>"}
form = [
    ("features", "clean_transcript"),
    ("features", "summary"),
    ("template_id", "meeting_notes_v1"),
]
with open("audio.mp3", "rb") as f:
    files = {"file": f}
    resp = requests.post(url, headers=headers, data=form, files=files, timeout=120)
print(resp.json())
```

### POST /v1/process_text (API sửa lỗi/hiệu đính văn bản)

Gửi văn bản thô (ví dụ transcript chưa sạch) để tóm tắt hoặc làm sạch/hiệu đính
không cần upload audio.

- Headers: `Content-Type: application/json`, `X-API-Key: <key>`
- Body mẫu:

```json
{
  "text": "Toan bo noi dung hop...",
  "features": ["clean_transcript", "summary"],
  "template_id": "meeting_notes_v1"
}
```

### GET /v1/status/{task_id}

Theo dõi tiến trình và lấy kết quả.

- Headers: `X-API-Key: <key>`
- Trạng thái: `PENDING`, `PREPROCESSING`, `PROCESSING_VAD`, `PROCESSING_ASR`, `PROCESSING_DIARIZATION`, `PROCESSING_LLM`,
  `COMPLETE`, `FAILED`.
- Nên polling mỗi 2–5 giây tới khi `COMPLETE` hoặc `FAILED`.

**Phản hồi mẫu (hoàn tất)**

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
  "metrics": {
    "input_duration_seconds": 2701.3,
    "processing_time_seconds": 162.8,
    "rtf": 0.06,
    "asr_confidence_avg": 0.91
  },
  "results": {
    "clean_transcript": "...",
    "summary": { "title": "...", "main_points": ["..."], "tags": ["..."] }
  }
}
```

### GET /v1/models

Liệt kê các backend ASR hiện có.

```bash
curl -H 'X-API-Key: $API_KEY' http://localhost:8000/v1/models
```

### Templates

- `GET /v1/templates`: Danh sách ID và mô tả template.
- `GET /v1/templates/{id}`: Lấy prompt và schema của template.
- `POST /v1/templates`: Tạo template (`id`, `schema_data`, `prompt_template`,
  tùy chọn `example`).
- `PUT /v1/templates/{id}`: Cập nhật template.
- `DELETE /v1/templates/{id}`: Xóa template.

Các thao tác ghi yêu cầu `X-API-Key` và `Content-Type: application/json`.

### Health Check

- `GET /health`: Trả về trạng thái dịch vụ, phiên bản, kết nối Redis, độ sâu hàng đợi.

## Xử lý lỗi

- HTTP ngay lập tức: `401` (thiếu/sai key), `413` (file quá lớn), `415` (sai định dạng),
  `422` (payload không hợp lệ), `429` (hàng đợi đầy hoặc rate limit), `500` (lỗi server).
- Lỗi trong tác vụ: xem `/v1/status/{task_id}` khi `status` = `FAILED` để đọc
  `error` và `error_code`.

## Khuyến nghị sử dụng

- Ưu tiên định dạng không nén (WAV/FLAC); cắt bỏ đoạn im lặng trước khi gửi.
- Dùng backoff lũy thừa khi gặp `429`; tránh gửi dồn dập.
- Poll trạng thái ở tần suất vừa phải (2–5 giây) và dừng khi đã xong/thất bại.
- Quản lý/luân phiên khóa API và lưu trong biến môi trường.
