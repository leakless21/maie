# Structured Analysis Template (v1)

## Tổng quan

Template **Phân tích nội dung có cấu trúc** được thiết kế để phân tích và cấu trúc hóa nội dung cuộc họp, buổi thảo luận hoặc bài thuyết trình theo 5 phần chính:

1. **Mở đầu** - Phần giới thiệu, người tham gia, bối cảnh
2. **Báo cáo** - Thông tin, dữ liệu, cập nhật được trình bày
3. **Thảo luận** - Trao đổi ý kiến, các quan điểm
4. **Kết luận** - Quyết định, thỏa thuận đạt được
5. **Giao việc** - Phân công nhiệm vụ, deadline

## Mục đích sử dụng

- **Chính:** Phân tích cuộc họp/buổi thuyết trình thành các phần có cấu trúc
- **Phụ:**
  - Tạo biên bản họp chi tiết
  - Theo dõi phân công công việc
  - Phân tích nội dung bài thuyết trình

## Cấu trúc Output

### 1. Mở đầu (mở_đầu)

- **summary**: Tóm tắt phần mở đầu
- **participants**: Danh sách người tham gia
- **context**: Bối cảnh, mục đích

### 2. Báo cáo (báo_cáo)

- **summary**: Tóm tắt nội dung báo cáo
- **điểm_chính**: Các điểm chính (max 20)
- **số_liệu_đề_cập**: Số liệu được đề cập (max 30)

### 3. Thảo luận (thảo_luận)

- **summary**: Tóm tắt thảo luận
- **chủ_đề**: Các chủ đề được thảo luận (max 15)
- **ý_kiến**: Ý kiến của từng người (max 30)
  - người_nói: Người nói
  - nội_dung: Nội dung ý kiến

### 4. Kết luận (kết_luận)

- **tóm_tắt**: Tóm tắt kết luận
- **quyết_định**: Các quyết định (max 20)
- **thỏa_thuận**: Các thỏa thuận (max 20)

### 5. Giao việc (giao_việc)

- **summary**: Tóm tắt phần giao việc
- **công_việc**: Danh sách công việc (max 50)
  - mô_tả: Mô tả công việc
  - người_phụ_trách: Người được giao
  - hạn_chót: Hạn hoàn thành (YYYY-MM-DD hoặc text)
  - mức_độ_ưu_tiên: cao | trung bình | thấp | không xác định

### 6. Tags

- 1-10 thẻ phân loại nội dung

## Điểm khác biệt

So với `meeting_notes_v2`, template này:

- ✅ **Phân tích chi tiết hơn** với 5 phần rõ ràng
- ✅ **Theo dõi ý kiến** từng người trong thảo luận
- ✅ **Phân tích số liệu** được đề cập trong báo cáo
- ✅ **Ánh xạ workflow** của cuộc họp (mở đầu → báo cáo → thảo luận → kết luận → giao việc)
- ✅ **Phù hợp với văn hóa họp hành Việt Nam**

## Sử dụng qua API

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@meeting.wav" \
  -F "template_id=structured_analysis_v1"
```

## Khi nào nên dùng

✅ **Tốt cho:**

- Cuộc họp có cấu trúc rõ ràng
- Buổi thuyết trình có phần Q&A
- Họp review/báo cáo có giao việc
- Cần phân tích chi tiết từng phần

❌ **Không phù hợp cho:**

- Trao đổi không chính thức
- Nội dung không có cấu trúc
- Chỉ cần summary tổng quan (dùng `generic_summary_v2`)
- Chỉ cần meeting minutes đơn giản (dùng `meeting_notes_v2`)

## Xử lý trường hợp đặc biệt

**Nếu thiếu một phần:**

- Vẫn phải có object với summary giải thích
- Các mảng có thể để rỗng `[]`
- Không bỏ qua phần nào

**Nội dung không rõ ràng:**

- Summary mô tả ngắn gọn những gì có
- Đánh dấu thông tin không xác định
- Không bịa đặt thông tin

## Ví dụ

Xem [example.json](example.json) để biết output mẫu cho một cuộc họp triển khai dự án đầy đủ.

---

**Version**: 1.0.0  
**Created**: 2025-12-26  
**Language**: Vietnamese
