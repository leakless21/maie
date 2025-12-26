## Tài liệu Yêu cầu Sản phẩm (PRD)

|                        |                                                  |
| :--------------------- | :----------------------------------------------- |
| **Tên dự án**          | AI C500 (Hệ thống Phân tích Âm thanh Thông minh) |
| **Phiên bản**          | 1.3 (V1.0 Sẵn sàng Sản xuất)                     |
| **Trạng thái**         | **Đã phê duyệt**                                 |
| **Tài liệu liên quan** | Project Brief V1.3, TDD V1.3                     |
| **Cập nhật lần cuối**  | 15 tháng 10, 2025                                |

#### 1. Giới thiệu

Tài liệu này xác định các yêu cầu sản phẩm cho phiên bản V1.0 của AI C500. Nó chi tiết hóa các tính năng và yêu cầu phi chức năng của dự án, đóng vai trò là hướng dẫn nền tảng cho việc thiết kế, phát triển và thử nghiệm.

Tổng quan kiến trúc: Hệ thống sử dụng kiến trúc ba tầng với máy chủ API, hàng đợi/lưu trữ Redis và worker GPU chạy suy luận ASR và LLM tuần tự trên một GPU duy nhất. Thiết kế này ưu tiên sự đơn giản và độ tin cậy trên các môi trường có tài nguyên hạn chế. Việc tạo prompt dựa trên định dạng ChatML mặc định đi kèm với mô hình Qwen3 cộng với các mẫu prompt Jinja cho mỗi `template_id` để kết xuất các `messages` kiểu OpenAI.

#### 2. Tính năng & Yêu cầu Chức năng V1.0 (FR)

- FR-1: Tiếp nhận Âm thanh — Hệ thống PHẢI chấp nhận các định dạng âm thanh phổ biến bao gồm `.wav`, `.mp3`, `.m4a`, và `.flac` thông qua tải lên `multipart/form-data`.

- FR-2: Backend ASR — Hệ thống cung cấp hai backend ASR cho V1.0:

  - chunkformer: Backend mặc định cho V1.0, được tối ưu hóa cho việc phiên âm âm thanh dạng dài và độ trễ của một yêu cầu. Biến thể mô hình mặc định là `khanhld/chunkformer-rnnt-large-vie`, cung cấp xử lý theo từng đoạn với các cửa sổ ngữ cảnh có thể cấu hình.

  - whisper: Backend thay thế sử dụng các mô hình dựa trên Whisper thông qua runtime CTranslate2. Biến thể mô hình mặc định là EraX-WoW-Turbo V1.1 (`erax-wow-turbo`), cung cấp dấu câu và viết hoa tự nhiên, được tối ưu hóa cho thông lượng với bộ lọc VAD.

  **Phạm vi tính năng ASR V1.0:**

  - ✅ Phiên âm cấp độ phân đoạn (văn bản + dấu thời gian bắt đầu/kết thúc cho mỗi phân đoạn)
  - ✅ Lọc VAD (Phát hiện Hoạt động Giọng nói) để cải thiện tốc độ/độ chính xác
  - ✅ Phát hiện ngôn ngữ
  - ✅ Điểm tin cậy cấp độ phân đoạn (cho số liệu FR-5)
  - ✅ Xử lý tuần tự (mẫu tải → thực thi → dỡ tải)
  - ✅ Hỗ trợ mô hình Distil-Whisper (các mô hình thay thế nhanh hơn)
  - ✅ Hỗ trợ mô hình ChunkFormer (tối ưu hóa cho âm thanh dạng dài)

  **Hoãn lại sau V1.0:**

  - ❌ Dấu thời gian cấp độ từ → V1.1+ (không yêu cầu bởi số liệu FR-5; cho các tính năng dòng thời gian/phụ đề)
  - ❌ Suy luận hàng loạt → V1.2+ (mâu thuẫn với kiến trúc tuần tự; yêu cầu tải trước mô hình)
  - ✅ Phân chia người nói — Đã triển khai; xem `docs/archive/diarization/DIARIZATION_FINAL_STATUS.md` (tích hợp pyannote.audio và logic căn chỉnh đã được áp dụng)
  - ❌ Phiên âm trực tuyến → V1.3+ (yêu cầu API WebSocket và thay đổi kiến trúc)

- FR-3: Nâng cao Văn bản — Nâng cao văn bản là một bước tùy chọn trong quy trình. Nó PHẢI được bỏ qua khi backend ASR được chọn cung cấp đủ dấu câu và cách viết hoa (ví dụ: `whisper` với biến thể `erax-wow-turbo`). Khi được yêu cầu (với các backend thiếu dấu câu; không áp dụng cho mặc định V1.0), quy trình sẽ sử dụng LLM để sửa dấu câu và viết hoa. Lưu ý: ChunkFormer có thể yêu cầu nâng cao văn bản tùy thuộc vào cấu hình mô hình.

- FR-4: Tóm tắt có cấu trúc — Hệ thống PHẢI tạo ra một bản tóm tắt từ bản phiên âm đã được nâng cao bằng cách sử dụng Qwen3-4B-Instruct LLM với lượng tử hóa AWQ 4-bit. Các bản tóm tắt PHẢI xác thực dựa trên một JSON Schema đã được phiên bản hóa tương ứng với `template_id` đã chọn. Việc tạo LLM sẽ sử dụng các kỹ thuật giải mã có ràng buộc hoặc đầu ra có cấu trúc để đảm bảo tuân thủ (vLLM `response_format` với JSON Schema). Việc tạo prompt PHẢI sử dụng các mẫu Jinja (hệ thống + người dùng) được kết xuất thành `messages`, dựa trên định dạng trò chuyện mặc định của mô hình. Bản thân các prompt có thể cấu hình và được quản lý dưới dạng các mẫu Jinja2 trong thư mục `templates/prompts`, cho phép sửa đổi và phiên bản hóa dễ dàng.

- FR-5: Số liệu Thời gian chạy — Kết quả PHẢI bao gồm các số liệu tự báo cáo để giúp người dùng đánh giá chất lượng và hiệu suất của quá trình xử lý. Chúng bao gồm: `rtf` (Hệ số Thời gian thực), `asr_confidence_avg`, `vad_coverage`, và `edit_rate_cleaning`.

- FR-6: Phân loại Tự động (Nhúng trong Tóm tắt) — Hệ thống PHẢI tạo ra các thẻ danh mục có liên quan (1-10 thẻ) cho âm thanh đầu vào. Điều này được thực hiện bằng cách bao gồm một trường `tags` trong lược đồ mẫu tóm tắt (FR-4). Các thẻ được tạo ra trong cùng một lượt suy luận LLM với bản tóm tắt, đảm bảo sự mạch lạc về ngữ nghĩa và giảm thời gian xử lý. Thẻ KHÔNG phải là một tính năng riêng biệt mà được nhúng trong cấu trúc tóm tắt.

- FR-7: Điểm cuối & Hợp đồng API

  - FR-7.1: Điểm cuối Xử lý Bất đồng bộ — `POST /v1/process`
    - Yêu cầu: Xem Phụ lục A. Bao gồm các `features` và `template_id` tùy chọn.
    - Phản hồi: `202 Accepted` với `{ "task_id": "..." }`.
  - FR-7.2: Điểm cuối Truy xuất Trạng thái & Kết quả — `GET /v1/status/{task_id}`
    - Phản hồi: Xem Phụ lục B để biết hợp đồng dữ liệu đầy đủ, chi tiết.

- FR-8: Điểm cuối Khám phá — API PHẢI hiển thị các điểm cuối khám phá để cho phép các nhà phát triển truy vấn các tài nguyên có sẵn.
  - `GET /v1/models`: Trả về danh sách các backend ASR và các biến thể mô hình có sẵn. Trong V1.0, điều này trả về cả backend Whisper (mặc định) và backend ChunkFormer với các biến thể mô hình và siêu dữ liệu tương ứng của chúng.
  - `GET /v1/templates`: Trả về danh sách các mẫu tóm tắt có sẵn và liên kết đến các JSON Schema của chúng. (Tương lai: có thể bao gồm phiên bản siêu dữ liệu prompt.)

#### 3. Yêu cầu Phi chức năng (NFR)

- NFR-1: Khả năng tái tạo — Tất cả các kết quả do API tạo ra PHẢI bao gồm một khối `versions`, chứa thông tin chi tiết về `pipeline_version`, tên mô hình, mã hash checkpoint, lượng tử hóa, các tham số giải mã chính và biến thể mẫu prompt để đảm bảo mọi kết quả đều có thể tái tạo và kiểm toán được.

- NFR-2: Trải nghiệm Nhà phát triển — Hệ thống PHẢI hiển thị một đặc tả OpenAPI 3.1 hợp lệ.

- NFR-3: Triển khai — Toàn bộ hệ thống PHẢI được đóng gói (Docker) và có thể triển khai tại chỗ thông qua tệp `docker-compose.yml`. Kiến trúc được tối ưu hóa cho các triển khai một GPU (16-24GB VRAM) thông qua thực thi mô hình tuần tự chỉ trong V1.0.

- NFR-4: Khả năng cấu hình — Sử dụng các giá trị mặc định có chủ ý với một bộ biến môi trường tối thiểu (khoảng 5–6) cho người vận hành (ví dụ: URL Redis, khóa API, độ sâu hàng đợi). Khả năng cấu hình bổ sung có thể được giới thiệu dần dần.

- NFR-5: Độ tin cậy & Chống áp lực — Hệ thống PHẢI triển khai kiểm tra độ sâu hàng đợi và trả về `429 Too Many Requests` khi hàng đợi đầy để ngăn chặn quá tải. Mục tiêu 6+ yêu cầu đồng thời mà không mất ổn định trên phần cứng tham chiếu.

- NFR-6: Bảo mật — Hệ thống PHẢI triển khai các thực hành xử lý tệp an toàn:
  - Xác thực kích thước tệp TRƯỚC KHI tải nội dung để ngăn chặn các cuộc tấn công từ chối dịch vụ (DoS) do kiệt sức bộ nhớ
  - Xác thực cả loại MIME và phần mở rộng tệp cho các tệp âm thanh được tải lên
  - Làm sạch tên tệp để ngăn chặn các cuộc tấn công duyệt đường dẫn
  - Sử dụng so sánh an toàn về thời gian để xác thực khóa API
  - Truyền trực tuyến các tệp lớn vào đĩa thay vì đệm trong bộ nhớ
  - Lưu trữ tệp trong các thư mục dành riêng cho tác vụ (`data/audio/{task-id}/raw.{ext}` và `preprocessed.wav`)

---

#### Phụ lục A: Hợp đồng Yêu cầu cho `POST /v1/process`

Nội dung Yêu cầu (`multipart/form-data`)

- `file` (tệp): Tệp âm thanh. (Bắt buộc)
- `features` (list[str]): Các đầu ra mong muốn. (Tùy chọn)
  - Giá trị: `raw_transcript`, `clean_transcript`, `summary`, `enhancement_metrics`.
  - Mặc định: `["clean_transcript", "summary"]`.
  - Lưu ý: `tags` KHÔNG CÒN là một tính năng riêng biệt. Các thẻ được nhúng trong đầu ra `summary` thông qua lược đồ mẫu.
- `asr_backend`: Lựa chọn backend giữa `"chunkformer"` (mặc định) và `"whisper"` cho các trường hợp sử dụng khác nhau.
- `template_id` (str): Định dạng tóm tắt. (Bắt buộc nếu `summary` có trong `features`)
  - Các mẫu nên bao gồm một trường `tags` (mảng từ 1-10 chuỗi) để phân loại tự động.

---

#### Phụ lục B: Hợp đồng Dữ liệu API Cuối cùng cho `GET /v1/status/{task_id}`

Phản hồi Cuối cùng Thành công (Nội dung JSON)

**Lưu ý:** Ví dụ dưới đây cho thấy phản hồi của backend Whisper. Đối với backend ChunkFormer, phần `asr_backend` sẽ bao gồm `name: "chunkformer"`, `model_variant: "rnnt-large-vie"`, `model_path: "khanhld/chunkformer-rnnt-large-vie"`, và các tham số kiến trúc như `chunk_size`, `left_context_size`, `right_context_size`, `total_batch_duration`, `return_timestamps`.

```json
{
  "task_id": "c4b3a216-3e7f-4d2a-8f9a-1b9c8d7e6a5b",
  "status": "COMPLETE",
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
  "chỉ_số": {
    "thời_lượng_đầu_vào_giây": 2701.3,
    "thời_gian_xử_lý_giây": 162.8,
    "hệ_số_thời_gian_thực": 0.06,
    "độ_phủ_vad": 0.88,
    "độ_tin_cậy_asr_trung_bình": 0.91,
    "tỷ_lệ_chỉnh_sửa_làm_sạch": 0.05
  },
  "kết_quả": {
    "bản_ghi_thô": "the meeting on oct 4 focused on q4 budgets...",
    "bản_ghi_sạch": "The meeting on October 4th focused on Q4 budgets.",
    "tóm_tắt": {
      "tiêu_đề": "Q4 Budget Planning Meeting",
      "abstract": "A review of the fourth-quarter budget proposal, focusing on marketing and R&D allocations.",
      "chủ_đề_chính": [
        "Marketing budget approved with a 5% increase.",
        "R&D budget for 'Project Phoenix' is pending final review."
      ],
      "thẻ": [
        "Finance",
        "Budget Planning",
        "Marketing Spend",
        "R&D Allocation"
      ]
    }
  }
}
```

**Phạm vi Đầu ra ASR V1.0:**

Phiên âm V1.0 chỉ cung cấp dữ liệu **cấp độ phân đoạn**:

- ✅ Văn bản phân đoạn với dấu thời gian bắt đầu/kết thúc
- ✅ Điểm tin cậy cấp độ phân đoạn
- ✅ Vùng giọng nói được lọc bởi VAD
- ✅ Siêu dữ liệu phát hiện ngôn ngữ

**Hoãn lại cho các bản phát hành trong tương lai:**

- ❌ Dấu thời gian cấp độ từ → V1.1+ (cho các tính năng dòng thời gian/phụ đề)
- ✅ Nhãn người nói — Đã triển khai; xem `docs/archive/diarization/DIARIZATION_FINAL_STATUS.md`
- ❌ Độ tin cậy cấp độ từ → V1.1+ (yêu cầu dấu thời gian cấp độ từ)

Hợp đồng phản hồi trên là hoàn chỉnh cho các yêu cầu V1.0. Các cấu trúc dữ liệu cấp độ từ sẽ được thêm vào trong V1.1 khi các tính năng dòng thời gian và phụ đề được triển khai.

---

Lưu ý quan trọng về Thẻ:

- Thẻ (FR-6) được nhúng trong đối tượng `summary`, không phải là một trường cấp cao riêng biệt
- Điều này đảm bảo các thẻ được tạo ra trong một lượt suy luận LLM duy nhất cùng với bản tóm tắt
- Các mẫu phải bao gồm một trường `tags` trong lược đồ JSON của chúng
- Nếu `summary` không được yêu cầu, các thẻ sẽ không được tạo
