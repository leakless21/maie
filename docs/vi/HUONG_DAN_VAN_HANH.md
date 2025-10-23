# Hướng dẫn Vận hành và Kiến trúc AI C500

## 🏗️ Kiến trúc hệ thống

AI C500 (Hệ thống Phân tích Âm thanh Thông minh) là hệ thống phân tán ba tầng sẵn sàng sản xuất được thiết kế để triển khai tại chỗ các khối lượng công việc xử lý âm thanh. Kiến trúc cho phép xử lý hiệu quả các tệp âm thanh thông qua quy trình tinh vi bao gồm tiền xử lý âm thanh, nhận dạng giọng nói tự động (ASR) và tăng cường mô hình ngôn ngữ lớn (LLM).

### Nguyên tắc thiết kế

- **Xử lý GPU tuần tự**: Được thiết kế cho các ràng buộc VRAM 16-24GB với quản lý vòng đời mô hình rõ ràng.
- **Kiến trúc mô-đun**: Hỗ trợ nhiều backend ASR (Whisper, ChunkFormer) thông qua mẫu thiết kế Factory.
- **Xử lý dựa trên mẫu**: Tạo đầu ra có cấu trúc bằng mẫu JSON.
- **Cấu hình an toàn**: Cấu hình dựa trên Pydantic với hỗ trợ biến môi trường.
- **Xử lý lỗi toàn diện**: Xử lý lỗi có cấu trúc với ghi log chi tiết.

### Kiến trúc ba tầng

AI C500 triển khai kiến trúc ba tầng được tối ưu hóa cho các khối lượng công việc xử lý âm thanh chuyên sâu GPU:

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

### Phân tích thành phần

#### 1. Lớp API (Tầng trình bày)

- **Công nghệ**: Khung web [Litestar](https://litestar.dev/)
- **Trách nhiệm**: Cung cấp các điểm cuối REST API không trạng thái, xử lý tải lên và xác thực tệp, quản lý xác thực và ủy quyền, tạo tác vụ và cho phép thăm dò trạng thái.
- **Cấu hình chính**:
  - `API_HOST`, `API_PORT`: Địa chỉ và cổng máy chủ.
  - `SECRET_API_KEY`: Khóa xác thực.
  - `MAX_FILE_SIZE_MB`: Giới hạn kích thước tệp tải lên.

#### 2. Lớp hàng đợi (Tầng logic)

- **Công nghệ**: Redis với RQ (Redis Queue)
- **Trách nhiệm**: Phân phối và cân bằng tải tác vụ, xử lý công việc nền, và giám sát độ sâu hàng đợi.
- **Cấu hình chính**:
  - `REDIS_URL`: URL kết nối Redis.
  - `REDIS_RESULTS_DB`: Cơ sở dữ liệu Redis để lưu trữ kết quả.
  - `MAX_QUEUE_DEPTH`: Độ sâu hàng đợi tối đa để chống áp lực.

#### 3. Lớp Worker (Tầng dữ liệu)

- **Công nghệ**: Worker Python RQ với tăng tốc GPU.
- **Trách nhiệm**: Thực thi quy trình xử lý tuần tự, quản lý vòng đời mô hình (tải → thực thi → dỡ tải), và tối ưu hóa bộ nhớ GPU.
- **Cấu hình chính**:
  - `WORKER_CONCURRENCY`: Số lượng quy trình worker đồng thời.
  - `JOB_TIMEOUT`: Thời gian chờ tối đa cho một công việc.
  - Cấu hình mô hình chi tiết (ASR, LLM) để quản lý tài nguyên GPU.

### Luồng dữ liệu

Quy trình xử lý từ đầu đến cuối của một yêu cầu trong AI C500:

```mermaid
graph TD
    A[Tải lên tệp] --> B[Xác thực API]
    B --> C[Tạo tác vụ]
    C --> D[Hàng đợi Redis]
    D --> E[Worker GPU]
    E --> F[Tiền xử lý âm thanh]
    F --> G[Xử lý ASR]
    G --> H[Tăng cường LLM]
    H --> I[Lưu trữ kết quả]
    I --> J[Cập nhật trạng thái]
```

### Mẫu thiết kế

- **Factory Pattern**: Được sử dụng trong `ASRFactory` để tạo các backend ASR khác nhau (`Whisper`, `ChunkFormer`) một cách linh hoạt, cho phép dễ dàng mở rộng với các mô hình mới.
- **Template Method Pattern**: Quy trình xử lý trong `pipeline.py` tuân theo một chuỗi các bước cố định (tiền xử lý, ASR, LLM), đảm bảo thực thi tuần tự và quản lý tài nguyên nhất quán.
- **Observer Pattern**: Trạng thái tác vụ trong Redis được cập nhật ở mỗi giai đoạn, cho phép máy khách "quan sát" tiến trình bằng cách thăm dò điểm cuối trạng thái.

### Ngăn xếp công nghệ

| Thành phần      | Công nghệ         | Mục đích                                       |
| --------------- | ----------------- | ---------------------------------------------- |
| Khung web       | Litestar          | REST API hiệu suất cao với OpenAPI.            |
| Hàng đợi tác vụ | Redis + RQ        | Xử lý công việc nền không đồng bộ.             |
| Cấu hình        | Pydantic Settings | Quản lý cấu hình an toàn, dựa trên môi trường. |
| Xử lý âm thanh  | PyTorch           | Khung học sâu cho các hoạt động GPU.           |
| Suy luận LLM    | vLLM              | Phục vụ LLM hiệu suất cao.                     |
| Backend ASR     | OpenAI Whisper    | Nhận dạng giọng nói chất lượng cao.            |
| Backend ASR     | ChunkFormer       | ASR phát trực tuyến cho tiếng Việt.            |

### Cân nhắc hiệu suất

- **Quản lý VRAM**: Kiến trúc được thiết kế để tải và dỡ tải các mô hình (ASR, LLM) một cách tuần tự, cho phép hoạt động trong môi trường có VRAM hạn chế (16-24GB). Bộ nhớ đệm CUDA được xóa một cách rõ ràng sau mỗi bước để giải phóng tài nguyên.
- **Hệ số thời gian thực (RTF)**: Hệ thống tính toán RTF để đo lường hiệu suất xử lý, cho biết tốc độ xử lý nhanh hơn hay chậm hơn thời gian thực.
- **Lượng tử hóa**: Hỗ trợ các loại tính toán lượng tử hóa (ví dụ: `int8_float16`) để giảm mức sử dụng bộ nhớ và tăng tốc độ suy luận với tác động tối thiểu đến độ chính xác.

## ⚙️ Hướng dẫn cấu hình

AI C500 sử dụng hệ thống cấu hình dựa trên Pydantic, cho phép thiết lập thông qua các biến môi trường.

### Yêu cầu Hệ thống

#### Yêu cầu Phần cứng

Để chạy AI C500 hiệu quả, đặc biệt là với các tác vụ xử lý GPU, hệ thống của bạn nên đáp ứng các yêu cầu sau:

- **CPU**: Tối thiểu 4 nhân (khuyến nghị 8+ nhân)
- **RAM**: Tối thiểu 16GB (khuyến nghị 32GB+)
- **Lưu trữ**: Ổ SSD tối thiểu 100GB để chứa các mô hình AI và dữ liệu âm thanh.
- **GPU**: Card đồ họa NVIDIA với ít nhất 16GB VRAM. Khuyến nghị 24GB+ VRAM để có hiệu suất tối ưu và khả năng xử lý các mô hình lớn hơn.

#### Kiến trúc GPU được hỗ trợ

Hệ thống tương thích với các kiến trúc GPU NVIDIA sau:

- Pascal (ví dụ: GTX 10-series, Tesla P100)
- Turing (ví dụ: RTX 20-series, Tesla T4)
- Ampere (ví dụ: RTX 30-series, A100)
- Ada Lovelace (ví dụ: RTX 40-series)

### Thiết lập môi trường

1.  **Sao chép kho mã nguồn:**

    ```bash
    git clone <repository-url>
    cd maie
    ```

2.  **Tạo tệp `.env`:**

    ```bash
    cp .env.template .env
    ```

    Sau đó, chỉnh sửa tệp `.env` với các giá trị cấu hình của bạn.

3.  **Tải mô hình:**
    ```bash
    pixi run download-models
    ```

### Biến môi trường chính

#### Cấu hình API

| Biến               | Mô tả                              | Mặc định                   |
| ------------------ | ---------------------------------- | -------------------------- |
| `API_HOST`         | Địa chỉ liên kết máy chủ API       | `0.0.0.0`                  |
| `API_PORT`         | Cổng máy chủ API                   | `8000`                     |
| `SECRET_API_KEY`   | Khóa xác thực API                  | `your_secret_api_key_here` |
| `MAX_FILE_SIZE_MB` | Kích thước tệp tải lên tối đa (MB) | `500.0`                    |

#### Cấu hình Redis

| Biến               | Mô tả                      | Mặc định                   |
| ------------------ | -------------------------- | -------------------------- |
| `REDIS_URL`        | URL kết nối Redis          | `redis://localhost:6379/0` |
| `REDIS_RESULTS_DB` | Số DB Redis cho kết quả    | `1`                        |
| `MAX_QUEUE_DEPTH`  | Kích thước hàng đợi tối đa | `50`                       |

#### Cấu hình Mô hình ASR (ChunkFormer - Mặc định)

| Biến                     | Mô tả                   | Mặc định                             |
| ------------------------ | ----------------------- | ------------------------------------ |
| `CHUNKFORMER_MODEL_NAME` | Tên mô hình ChunkFormer | `khanhld/chunkformer-rnnt-large-vie` |

#### Cấu hình Mô hình ASR (Whisper - Thay thế)

| Biến                    | Mô tả                                  | Mặc định         |
| ----------------------- | -------------------------------------- | ---------------- |
| `WHISPER_MODEL_VARIANT` | Biến thể mô hình Whisper               | `erax-wow-turbo` |
| `WHISPER_DEVICE`        | Thiết bị tính toán (`cuda` hoặc `cpu`) | `cuda`           |
| `WHISPER_COMPUTE_TYPE`  | Loại lượng tử hóa                      | `float16`        |

#### Cấu hình Mô hình LLM

| Biến                                 | Mô tả                        | Mặc định                                  |
| ------------------------------------ | ---------------------------- | ----------------------------------------- |
| `LLM_ENHANCE_MODEL`                  | Đường dẫn mô hình tăng cường | `data/models/qwen3-4b-instruct-2507-awq`  |
| `LLM_SUM_MODEL`                      | Đường dẫn mô hình tóm tắt    | `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit` |
| `LLM_ENHANCE_GPU_MEMORY_UTILIZATION` | Tỷ lệ sử dụng bộ nhớ GPU     | `0.9`                                     |

#### Cấu hình Worker

| Biến                 | Mô tả                            | Mặc định      |
| -------------------- | -------------------------------- | ------------- |
| `WORKER_NAME`        | Định danh worker                 | `maie-worker` |
| `JOB_TIMEOUT`        | Thời gian chờ công việc (giây)   | `600`         |
| `RESULT_TTL`         | Thời gian lưu giữ kết quả (giây) | `86400`       |
| `WORKER_CONCURRENCY` | Số quy trình worker              | `2`           |

### Cấu hình Docker

Tệp `docker-compose.yml` cung cấp một cách dễ dàng để triển khai AI C500 trong môi trường container.

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - SECRET_API_KEY=${SECRET_API_KEY}

  worker:
    build:
      context: .
      target: production
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

### Khả năng mở rộng và Triển khai

- **Mở rộng quy mô Worker**: Tăng số lượng `worker` trong `docker-compose.yml` để xử lý nhiều tác vụ hơn đồng thời.
- **Cân bằng tải**: Sử dụng một bộ cân bằng tải (ví dụ: Nginx) phía trước các dịch vụ `api` để phân phối lưu lượng truy cập.
- **Cụm Redis**: Đối với các thiết lập quy mô lớn, hãy sử dụng một cụm Redis chuyên dụng.

### Bảo mật

- **Khóa API**: Sử dụng khóa API mạnh và luân chuyển chúng định kỳ.
- **Mạng**: Chạy Redis và các worker trên một mạng nội bộ không thể truy cập từ bên ngoài.
- **Bảo mật tệp tải lên**: Đảm bảo xác thực kích thước và loại tệp được bật.

### Bảo trì và Dọn dẹp

AI C500 bao gồm các công cụ bảo trì tự động để quản lý dung lượng ổ cứng và bộ nhớ cache. Các script dọn dẹp giúp hệ thống hoạt động hiệu quả trong thời gian dài.

#### Các Script Dọn dẹp Chính

**1. clean-logs.sh - Dọn dẹp Log Files**

```bash
# Dọn dẹp log files cũ hơn 7 ngày (mặc định)
./scripts/clean-logs.sh

# Dọn dẹp log files cũ hơn 30 ngày
DAYS_TO_KEEP=30 ./scripts/clean-logs.sh

# Chạy thử nghiệm (không thực sự xóa)
DRY_RUN=true ./scripts/clean-logs.sh
```

**2. clean-audio.sh - Dọn dẹp Audio Files đã xử lý**

```bash
# Dọn dẹp audio files cho các tác vụ hoàn thành/thất bại
./scripts/clean-audio.sh

# Giữ audio files trong 14 ngày
RETENTION_DAYS=14 ./scripts/clean-audio.sh

# Chạy thử nghiệm
DRY_RUN=true ./scripts/clean-audio.sh
```

**3. clean-cache.sh - Dọn dẹp Redis Cache**

```bash
# Dọn dẹp cache Redis và queue entries
./scripts/clean-cache.sh

# Chạy thử nghiệm
DRY_RUN=true ./scripts/clean-cache.sh
```

**4. clean-all.sh - Dọn dẹp Toàn diện**

```bash
# Chạy tất cả script dọn dẹp theo thứ tự
./scripts/clean-all.sh

# Chạy thử nghiệm tất cả
DRY_RUN=true ./scripts/clean-all.sh

# Bỏ qua dọn dẹp cache
SKIP_CACHE=true ./scripts/clean-all.sh
```

**5. disk-monitor.sh - Giám sát Dung lượng Ổ cứng**

```bash
# Kiểm tra dung lượng ổ cứng hiện tại
./scripts/disk-monitor.sh

# Cảnh báo khi sử dụng > 80% (mặc định)
DISK_THRESHOLD=80 ./scripts/disk-monitor.sh

# Tự động dọn dẹp khi vượt ngưỡng
EMERGENCY_CLEANUP=true ./scripts/disk-monitor.sh
```

#### Cấu hình Biến Môi trường

| Biến             | Mô tả                          | Mặc định                   |
| ---------------- | ------------------------------ | -------------------------- |
| `LOG_DIR`        | Thư mục chứa log files         | `logs`                     |
| `DAYS_TO_KEEP`   | Số ngày giữ log files          | `7`                        |
| `AUDIO_DIR`      | Thư mục chứa audio files       | `data/audio`               |
| `RETENTION_DAYS` | Số ngày giữ audio files        | `7`                        |
| `REDIS_URL`      | URL kết nối Redis              | `redis://localhost:6379/1` |
| `DISK_THRESHOLD` | Ngưỡng cảnh báo dung lượng (%) | `80`                       |
| `DRY_RUN`        | Chế độ chạy thử nghiệm         | `false`                    |

#### Tự động hóa cấp Máy chủ

**Phân biệt các loại Cron Job:**

Đối với việc tự động hóa máy chủ MAIE, cần phân biệt rõ giữa **crontab cấp người dùng** và **crontab cấp hệ thống**:

- **Crontab cấp người dùng** (`crontab -e`): Chạy với quyền của người dùng hiện tại, không cần quyền root, cô lập với các tiến trình MAIE
- **Crontab cấp hệ thống** (`sudo crontab -e` hoặc `/etc/crontab`): Chạy với quyền root, ảnh hưởng toàn bộ hệ thống, cần quyền quản trị

Đối với việc triển khai máy chủ MAIE, **khuyến nghị sử dụng crontab cấp người dùng** vì:

- ✅ Không cần quyền root
- ✅ Cô lập với môi trường người dùng MAIE
- ✅ Giảm rủi ro bảo mật
- ✅ Dễ quản lý và khắc phục sự cố hơn

**Thiết lập Crontab cấp Người dùng:**

**Đối với tài khoản người dùng MAIE:**

```bash
# Chuyển sang người dùng MAIE (khuyến nghị)
sudo su - maie

# Chỉnh sửa crontab của người dùng (không cần sudo khi đã là người dùng maie)
crontab -e

# Thêm các dòng sau để bảo trì thường xuyên:
# Dọn dẹp toàn diện hàng tuần (Chủ nhật 2 giờ sáng)
0 2 * * 0 ./scripts/clean-all.sh

# Dọn dẹp audio hàng ngày (6 giờ sáng)
0 6 * * * ./scripts/clean-audio.sh

# Dọn dẹp cache hàng ngày (7 giờ sáng)
0 7 * * * ./scripts/clean-cache.sh

# Giám sát dung lượng mỗi 6 giờ
0 */6 * * * ./scripts/disk-monitor.sh
```

**Đối với các tài khoản người dùng khác nhau:**

```bash
# Với quyền root (không khuyến nghị cho MAIE)
sudo crontab -e

# Với người dùng maie chuyên dụng (khuyến nghị)
sudo su - maie
crontab -e

# Với người dùng triển khai (thay thế)
sudo su - deploy
crontab -e
```

**Ví dụ Cron Job với Logging:**

**Thiết lập cơ bản với logging:**

```bash
# Tạo thư mục log cho cron jobs
mkdir -p logs/cron

# Chỉnh sửa crontab với quyền người dùng MAIE
crontab -e

# Thêm các dòng sau:
# Dọn dẹp hàng ngày với logging (3 giờ sáng)
0 3 * * * ./scripts/clean-all.sh >> logs/cron/cleanup.log 2>&1

# Giám sát dung lượng mỗi giờ (chế độ không khẩn cấp)
0 * * * * ./scripts/disk-monitor.sh >> logs/cron/monitor.log 2>&1

# Giám sát khẩn cấp trong giờ làm việc (9 giờ sáng - 5 giờ chiều, các ngày trong tuần)
0 9-17 * * 1-5 EMERGENCY_CLEANUP=true ./scripts/disk-monitor.sh >> logs/cron/emergency.log 2>&1
```

**Lịch trình máy chủ Production:**

```bash
# Chỉnh sửa crontab với quyền người dùng MAIE
crontab -e

# Thêm các dòng sau cho môi trường production:
# Lịch trình dọn dẹp bảo thủ
# Dọn dẹp toàn diện hàng ngày lúc 3 giờ sáng (ít tích cực hơn)
0 3 * * * ./scripts/clean-all.sh >> logs/cron/cleanup.log 2>&1

# Dọn dẹp audio hai lần mỗi ngày (6 giờ sáng, 6 giờ chiều)
0 6,18 * * * ./scripts/clean-audio.sh >> logs/cron/audio-cleanup.log 2>&1

# Dọn dẹp cache ba lần mỗi ngày (7 giờ sáng, 2 giờ chiều, 10 giờ tối)
0 7,14,22 * * * ./scripts/clean-cache.sh >> logs/cron/cache-cleanup.log 2>&1

# Giám sát dung lượng mỗi 4 giờ
0 */4 * * * ./scripts/disk-monitor.sh >> logs/cron/disk-monitor.log 2>&1

# Giám sát khẩn cấp trong giờ cao điểm (8 giờ sáng - 8 giờ tối)
0 8-20 * * * EMERGENCY_CLEANUP=true ./scripts/disk-monitor.sh >> logs/cron/emergency.log 2>&1
```

**Kiểm tra Cron Job:**

**Kiểm tra các Cron Job đang hoạt động:**

```bash
# Liệt kê cron jobs của người dùng hiện tại
crontab -l

# Kiểm tra xem dịch vụ cron có đang chạy không
systemctl status cron

# Thử nghiệm cron job thủ công
./scripts/clean-all.sh >> logs/cron/manual-test.log 2>&1
```

**Giám sát thực thi Cron Job:**

```bash
# Theo dõi log cron theo thời gian thực
tail -f logs/cron/*.log

# Kiểm tra thời gian thực thi cuối cùng
ls -la logs/cron/

# Xác minh dung lượng ổ đĩa sau khi dọn dẹp
df -h . && echo "--- Hoạt động dọn dẹp gần đây ---" && tail -3 logs/cron/cleanup.log
```

**Khi nào sử dụng System vs User Cron:**

**Sử dụng User-Level Cron (Khuyến nghị cho MAIE):**

- ✅ Các tác vụ bảo trì máy chủ MAIE
- ✅ Không cần quyền root
- ✅ Cô lập với các tiến trình MAIE
- ✅ Dễ khắc phục sự cố hơn
- ✅ Rủi ro bảo mật thấp hơn

**Sử dụng System-Level Cron (Khi thực sự cần thiết):**

- ❌ Các tác vụ bảo trì toàn hệ thống
- ❌ Cần quyền root
- ❌ Ảnh hưởng toàn bộ máy chủ
- ❌ Rủi ro bảo mật cao hơn
- ❌ Phức tạp hơn để khắc phục sự cố

**Môi trường Phát triển:**

```bash
# Dọn dẹp thường xuyên hơn
export RETENTION_DAYS=3
export DAYS_TO_KEEP=3
export DISK_THRESHOLD=70

# Lên lịch hàng tuần với quyền người dùng MAIE
0 2 * * 0 ./scripts/clean-all.sh >> logs/cron/cleanup.log 2>&1
```

**Môi trường Production:**

```bash
# Giữ dữ liệu lâu hơn để debug
export RETENTION_DAYS=14
export DAYS_TO_KEEP=30
export DISK_THRESHOLD=85

# Lên lịch hàng ngày với quyền người dùng MAIE
0 3 * * * ./scripts/clean-all.sh >> logs/cron/cleanup.log 2>&1
```

#### Tích hợp Docker

Thêm cleanup scripts vào `docker-compose.yml`:

```yaml
services:
  # ... các service hiện tại ...

  cleanup:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - REDIS_URL=redis://redis:6379/1
      - RETENTION_DAYS=7
    command: ["./scripts/clean-all.sh"]
    profiles:
      - cleanup
```

**Sử dụng:**

```bash
# Chạy dọn dẹp trong Docker
docker-compose --profile cleanup up

# Chạy giám sát trong Docker
docker-compose --profile monitor up
```

#### Các Tính năng An toàn

- **Chế độ Dry-run**: Tất cả scripts hỗ trợ `DRY_RUN=true` để kiểm tra an toàn
- **Xác thực Trạng thái**: Chỉ xóa audio files cho tác vụ hoàn thành/thất bại
- **Redis An toàn**: Sử dụng TTL, không xóa dữ liệu đang hoạt động
- **Xử lý Lỗi**: Scripts sử dụng `set -euo pipefail` cho xử lý lỗi mạnh mẽ

#### Khắc phục sự cố

**Lỗi "Permission denied":**

```bash
# Sửa quyền
chmod +x scripts/clean-*.sh

# Hoặc chạy với sudo
sudo ./scripts/clean-logs.sh
```

**Lỗi kết nối Redis:**

```bash
# Kiểm tra kết nối
redis-cli ping

# Xác thực URL format
echo $REDIS_URL  # Should be: redis://host:port/db
```

**Dung lượng không được giải phóng:**

```bash
# Kiểm tra files lớn còn lại
find . -type f -size +100M -exec ls -lh {} \;

# Xác thực cleanup đã chạy
DRY_RUN=false ./scripts/clean-all.sh
```

### Khắc phục sự cố

- **Vấn đề kết nối Redis**: Đảm bảo dịch vụ Redis đang chạy và có thể truy cập được từ các container `api` và `worker`.
- **Lỗi bộ nhớ GPU**: Nếu bạn gặp lỗi hết bộ nhớ CUDA, hãy thử giảm `LLM_..._GPU_MEMORY_UTILIZATION` hoặc sử dụng các mô hình được lượng tử hóa nhỏ hơn (`WHISPER_COMPUTE_TYPE=int8`).
- **Vấn đề tải mô hình**: Kiểm tra xem các đường dẫn mô hình trong `.env` là chính xác và các tệp mô hình tồn tại.
- **Vấn đề cleanup**: Kiểm tra logs trong `logs/cron/cleanup.log` để xem chi tiết lỗi cleanup.
