# Báo Cáo Phân Tích & Kiến Trúc Hệ Thống: Market Sentiment Demo

## 1. Giới thiệu tổng quan
Dự án **Market Sentiment Demo** là một hệ thống phân tích tâm lý thị trường (market sentiment) end-to-end dành cho thị trường tài chính Việt Nam. 

Hệ thống tự động thu thập tin tức từ CafeF, lưu trữ, lọc bài viết liên quan đến tài chính, trích xuất thực thể (các mã cổ phiếu), phân loại khía cạnh (aspect) của tin tức, và gọi một mô hình AI (qua FastAPI) để đánh giá tâm lý (tích cực, tiêu cực, trung lập). Cuối cùng, hệ thống tổng hợp chỉ số tâm lý theo ngày/mã cổ phiếu và đối chiếu với dữ liệu giá thị trường thực tế để kiểm định mức độ tương quan, cung cấp dashboard trực quan.

## 2. Kiến trúc Hệ thống

Hệ thống được thiết kế theo kiến trúc Data Pipeline (ETL & ML Inference) gồm các thành phần chính:

*   **Crawler (Data Ingestion):** Thu thập dữ liệu từ các chuyên mục của CafeF.
*   **Database (PostgreSQL):** Lưu trữ toàn bộ dữ liệu từ bài viết thô, kết quả xử lý NLP, đến dữ liệu thị trường và báo cáo.
*   **Preprocessing & NLP Pipeline:**
    *   Lọc mức độ liên quan (Relevance Filter).
    *   Trích xuất và liên kết thực thể (Entity Linking).
    *   Trích xuất khía cạnh (Aspect Extraction).
*   **ML Inference (Mô hình AI):** Giao tiếp với một mô hình FastAPI bên ngoài để phân tích cảm xúc (Sentiment Analysis).
*   **Data Aggregation & Validation:** Tổng hợp điểm số cảm xúc, thu thập dữ liệu giá từ VNStock, và kiểm định tương quan.
*   **Dashboard:** Hiển thị báo cáo và dữ liệu dưới dạng Streamlit app.

### Sơ đồ luồng dữ liệu (Data Flow)

1. **Tin tức CafeF** -> `crawlers` -> **Raw Data (`articles`)**
2. **Raw Data** -> `relevance_filter` -> **Filtered Data (`article_relevance`)**
3. **Filtered Data** -> `entity_linker` -> **Entities (`article_entities`)**
4. **Entities** -> `aspect_extractor` -> **Aspects (`entity_aspects`)**
5. **Aspects** -> `model_client` (FastAPI) -> **Sentiment Scores (`entity_sentiments`)**
6. **Sentiment Scores** -> `build_sentiment_aggregates` -> **Daily Index (`daily_sentiment_index`, `article_ticker_sentiment`)**
7. **VNStock Market Data** -> `fetch_market_data` -> **Market Features (`market_prices`, `market_features`)**
8. **Daily Index & Market Features** -> `correlation_report` -> **Validation Views & CSV Reports**
9. **Database Views** -> `Streamlit Dashboard` -> **User Interface**

---

## 3. Phân tích chi tiết các thành phần (Logic cốt lõi)

### 3.1. Master Data (Dữ liệu gốc)
- Hệ thống sử dụng một danh sách các mã cổ phiếu (`ticker_master`) và các tên gọi khác (alias) của cổ phiếu (`ticker_aliases`) để phục vụ cho việc nhận diện trong văn bản.

### 3.2. Data Collection (Cào dữ liệu - src/crawlers)
- **Crawler:** Sử dụng `requests` và `BeautifulSoup` (hoặc `trafilatura` làm fallback) để cào tin tức từ CafeF (các mục: thị trường chứng khoán, doanh nghiệp, ngân hàng, báo cáo phân tích).
- **Tính năng:** Có cơ chế retry, delay, timeout và crawl theo trang (timeline API của CafeF) để lấy dữ liệu lịch sử (backfill) hoặc dữ liệu hàng ngày (daily).
- Dữ liệu thô lưu vào bảng `articles` dựa trên hàm băm (hash) của nội dung để tránh trùng lặp.

### 3.3. Preprocessing (Tiền xử lý & Lọc - src/preprocessing)
- **Relevance Filter (`relevance_filter.py`):** Đánh giá xem một bài viết có thực sự liên quan đến tài chính hay không dựa trên danh sách từ khóa tài chính (như "cổ phiếu", "lợi nhuận", "VN-Index"...) và trừ điểm nếu có từ khóa nhiễu (showbiz, đời sống). 
- **Phân loại:** Bài viết được chấm điểm và gán nhãn: `process_sentiment` (xử lý tiếp), `review_later`, hoặc `skip_sentiment` (bỏ qua).

### 3.4. NLP Engine (src/nlp)
- **Entity Linking (`entity_linker.py`):** Trích xuất các mã cổ phiếu được nhắc đến trong văn bản. Hệ thống ánh xạ dựa trên bảng alias. Có xử lý nhiễu với các mã cổ phiếu dễ nhầm lẫn như CEO, GAS, POW (chỉ nhận diện nếu viết hoa toàn bộ và đứng độc lập).
- **Aspect Extraction (`aspect_extractor.py`):** Dựa vào context xung quanh thực thể, hệ thống phân loại câu đó thuộc khía cạnh nào trong 5 khía cạnh chính: `business_financials`, `leadership_insider`, `macro_policy`, `capital_structure`, `legal_esg`. Việc này được thực hiện qua mapping từ khóa.

### 3.5. Inference (Suy luận AI - src/inference)
- `model_client.py`: Gửi text và aspect đã trích xuất tới một endpoint FastAPI bên ngoài.
- FastAPI trả về kết quả Sentiment (positive, negative, neutral) cùng các chỉ số xác suất (prob_positive, prob_negative, ...).
- Kết quả được lưu vào bảng `entity_sentiments`.

### 3.6. Aggregation & Validation (Tổng hợp và Kiểm định)
- **Aggregates (`build_sentiment_aggregates.py`):** 
  - Tính điểm Sentiment trung bình cho từng bài báo/mã cổ phiếu (`article_ticker_sentiment`).
  - Gộp thành Sentiment Index hàng ngày cho từng mã (`daily_sentiment_index`). Nếu ngày đăng báo là ngày nghỉ, điểm sentiment được dồn sang ngày giao dịch kế tiếp.
- **Market Data (`market_data/`):** Tải dữ liệu giá OHLCV lịch sử từ VNStock và lưu vào cơ sở dữ liệu, tính toán các chỉ số phái sinh như tỷ suất sinh lời (return_1d, return_3d...), biến động khối lượng.
- **Validation (`validation/correlation_report.py`):** Xây dựng các View trong PostgreSQL để ghép dữ liệu Sentiment và Market. Tính toán độ tương quan (Correlation) và Tỷ lệ chính xác (Hit Rate) để xem liệu tin tốt có thực sự đẩy giá lên và tin xấu đẩy giá xuống hay không.

### 3.7. Orchestration (Điều phối - src/jobs/run_full_pipeline.py)
- Toàn bộ các bước trên được kết nối thành một quy trình tự động hóa hoàn chỉnh qua script `run_full_pipeline.py`.
- Cho phép chạy toàn bộ hoặc bỏ qua một số bước bằng các tham số dòng lệnh (`--skip-crawl`, `--skip-inference`, v.v.).

---

## 4. Công nghệ sử dụng
- **Ngôn ngữ:** Python 3
- **Database:** PostgreSQL (với Docker & pgAdmin).
- **Thư viện Crawl & Parse:** `requests`, `beautifulsoup4`, `trafilatura`.
- **Thư viện Data & Math:** `pandas`, `SQLAlchemy`.
- **Giao diện:** `streamlit`, `plotly` (cho Dashboard).
- **Dữ liệu thị trường:** `vnstock`.

---

## 5. Điểm mạnh của kiến trúc
1. **Khả năng mở rộng (Scalability):** Tách biệt rõ ràng giữa Crawler, NLP rules và AI Model. Mô hình AI chạy qua FastAPI giúp dễ dàng scale GPU server độc lập với pipeline data.
2. **Quản lý trạng thái tốt:** Thiết kế Database chặt chẽ với quan hệ rõ ràng (`articles` -> `article_relevance` -> `article_entities` -> `entity_aspects` -> `entity_sentiments`). Các thao tác insert sử dụng cơ chế `ON CONFLICT DO UPDATE` (Upsert), giúp pipeline chạy lại (retry) nhiều lần không bị lỗi trùng lặp dữ liệu.
3. **Giảm thiểu nhiễu tín hiệu (Noise reduction):** Bộ lọc Relevance và Entity Linker có luật xử lý nhiễu rõ ràng (ví dụ: né mã CEO khi là danh từ chung), kết hợp Aspect Extraction giúp bối cảnh hoá câu văn trước khi đưa vào mô hình.
4. **Kiểm định chặt chẽ (Validation):** Không chỉ xuất ra sentiment, hệ thống tích hợp sâu với dữ liệu giá để chứng minh mức độ hiệu quả (Correlation & Hit rate).

## 6. Đề xuất cải thiện
- **Message Broker/Queue:** Hiện tại `run_full_pipeline` chạy đồng bộ theo từng batch lớn. Khi scale lên nhiều báo, nên dùng Celery/RabbitMQ hoặc Kafka để xử lý luồng stream (ví dụ: báo mới về -> push queue -> NLP -> push queue -> FastAPI).
- **NLP linh hoạt hơn:** Hiện tại Aspect Extraction dùng Rule-based (từ khóa). Có thể cải tiến dùng Zero-shot Classification của LLM hoặc embedding similarity để gán nhãn aspect chính xác hơn khi từ khóa không có trong từ điển.
- **Dynamic Crawler:** CafeF có thể thay đổi cấu trúc trang. Cần thiết lập hệ thống cảnh báo tự động khi Crawler trả về lỗi (lỗi CSS Selector, chặn IP).
