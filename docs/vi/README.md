# AI C500 — Hệ thống Phân tích Âm thanh Thông minh

[![Phiên bản](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.12+-green.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)]()
[![Bài kiểm tra](https://img.shields.io/badge/tests-70%2F70-green.svg)]()

**AI C500 (Hệ thống Phân tích Âm thanh Thông minh)** là một giải pháp toàn diện, sẵn sàng cho doanh nghiệp, được thiết kế để chuyển đổi các tệp âm thanh phi cấu trúc thành dữ liệu có cấu trúc, thông minh và có thể hành động. Trong bối cảnh ngày càng nhiều thông tin được trao đổi qua âm thanh—từ các cuộc họp, phỏng vấn, đến các cuộc gọi hỗ trợ khách hàng—AI C500 cung cấp các công cụ mạnh mẽ để tự động hóa việc trích xuất thông tin chi tiết có giá trị.

Hệ thống kết hợp các mô hình Nhận dạng Giọng nói Tự động (ASR) và Mô hình Ngôn ngữ Lớn (LLM) tiên tiến để cung cấp các bản ghi chính xác, tóm tắt thông minh và khả năng tăng cường nội dung, tất cả đều thông qua một API dễ tích hợp và có thể triển khai tại chỗ để đảm bảo an toàn dữ liệu.

### Các trường hợp sử dụng chính

- **Phân tích cuộc họp**: Tự động tạo biên bản cuộc họp, tóm tắt các quyết định và các mục hành động.
- **Ghi âm phỏng vấn**: Chuyển các cuộc phỏng vấn thành văn bản có cấu trúc để phân tích và lưu trữ dễ dàng.
- **Phân tích cuộc gọi hỗ trợ**: Phân tích các cuộc gọi của trung tâm liên lạc để đánh giá chất lượng và trích xuất các vấn đề phổ biến của khách hàng.
- **Chuyển đổi nội dung media**: Chuyển đổi podcast, bài giảng và các nội dung âm thanh khác thành văn bản để tạo phụ đề, bài viết blog, hoặc tài liệu.

### Khả năng chính

- **Hỗ trợ đa định dạng âm thanh**: Xử lý WAV, MP3, M4A, FLAC.
- **Hai lựa chọn ASR**: Whisper và ChunkFormer cho độ chính xác tối ưu.
- **Tóm tắt thông minh**: Tóm tắt nội dung dựa trên mẫu.
- **Tăng tốc GPU**: Tối ưu cho việc triển khai trên VRAM 16-24GB.
- **Sẵn sàng cho doanh nghiệp**: Giám sát toàn diện, ghi log và kiểm tra sức khỏe.

### 🏗️ Kiến trúc hệ thống

AI C500 triển khai kiến trúc cổ điển ba tầng được tối ưu hóa cho các khối lượng công việc xử lý âm thanh chuyên sâu GPU:

```
┌─────────────────────────────────────────────────────────────────┐
│                           Lớp API                               │
│                    (Khung web Litestar)                         │
├─────────────────────────────────────────────────────────────────┤
│                      Lớp hàng đợi                               │
│                (Redis + RQ Task Queue)                          │
├─────────────────────────────────────────────────────────────────┤
│                     Lớp Worker                                  │
│              (Worker GPU với xử lý tuần tự)                     │
└─────────────────────────────────────────────────────────────────┘
```

## ⚡ Khởi động nhanh

### Sử dụng Docker Compose (Khuyến nghị)

1.  **Sao chép và cấu hình:**

    ```bash
    git clone <repository-url>
    cd maie
    cp .env.template .env
    # Chỉnh sửa .env với cấu hình của bạn
    ```

2.  **Khởi động hệ thống:**

    ```bash
    docker-compose up -d
    ```

3.  **Xác minh triển khai:**

    ```bash
    curl -f http://localhost:8000/health
    ```

4.  **Xử lý tệp âm thanh đầu tiên của bạn:**

    ```bash
    curl -X POST "http://localhost:8000/v1/process" \
      -H "X-API-Key: <your-api-key>" \
      -F "file=@/path/to/your/audio.wav" \
      -F "features=clean_transcript" \
      -F "features=summary" \
      -F "template_id=meeting_notes_v1"
    ```

## 📖 Tài liệu

Để tìm hiểu sâu hơn, vui lòng tham khảo các tài liệu chi tiết sau:

- **[README.md](README.md)**: Tổng quan và hướng dẫn cài đặt nhanh.
- **[HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md)**: Hướng dẫn chi tiết cách sử dụng API, bao gồm các ví dụ và trường hợp sử dụng thực tế.
- **[HUONG_DAN_VAN_HANH.md](HUONG_DAN_VAN_HANH.md)**: Tài liệu về kiến trúc hệ thống, cách cấu hình và triển khai AI C500.
- **[HUONG_DAN_DONG_GOP.md](HUONG_DAN_DONG_GOP.md)**: Hướng dẫn dành cho nhà phát triển muốn đóng góp vào dự án, bao gồm quy trình phát triển và kiểm thử.
