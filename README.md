# Market Sentiment Demo - End-to-End

Project demo xây hệ thống Market Sentiment cho thị trường tài chính Việt Nam.

```text
CafeF news
→ crawl bài viết (category feed hoặc ticker-specific events)
→ lưu PostgreSQL
→ relevance filter (rule-based scoring)
→ entity linking theo mã cổ phiếu
→ aspect extraction
→ sentiment inference (local PhoBERT hoặc FastAPI)
→ aggregate sentiment theo mã/ngày
→ lấy dữ liệu giá thị trường
→ kiểm định tương quan sentiment với biến thị trường
→ dashboard
```

---

## 1. Cài môi trường

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Cấu hình môi trường

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
copy .env.example .env
```

Chỉnh `.env` theo môi trường:

```env
DB_HOST=127.0.0.1
DB_PORT=5434
DB_NAME=market_sentiment
DB_USER=postgres
DB_PASSWORD=postgres

# Local model (không cần API server)
SENTIMENT_MODEL_PATH=../stock_news/models/finetuned_last_layer_vn30_content
```

---

## 3. Chạy PostgreSQL

```bash
docker compose up -d
```

> **Windows:** nếu port 5432/5433 đã bị local PostgreSQL chiếm, đổi mapping trong `docker-compose.yml` và `DB_PORT` trong `.env`.

---

## 4. Khởi tạo DB và master data

```bash
python -m src.jobs.init_db
python -m src.master_data.build_ticker_master
python -m src.master_data.aliases_builder
```

---

## 5. Crawl CafeF

### 5a. Category feed (tin tức theo chuyên mục)

Crawl tất cả chuyên mục, chỉ lấy bài từ 2024 trở đi:

```bash
python -m src.jobs.crawl_backfill --max-per-category 200 --timeline-pages 10
```

Tùy chọn:

```bash
# Chỉ crawl một chuyên mục
python -m src.jobs.crawl_backfill --category doanh_nghiep --max-per-category 200

# Crawl song song với nhiều thread (mặc định 10, khuyến nghị 30–50 khi backfill lớn)
python -m src.jobs.crawl_backfill --category doanh_nghiep --max-per-category 999999 \
  --timeline-pages 2000 --workers 50

# Mở rộng phạm vi năm
python -m src.jobs.crawl_backfill --min-year 2023 --timeline-pages 20

# Không giới hạn năm
python -m src.jobs.crawl_backfill --min-year 0
```

Các chuyên mục có sẵn: `thi_truong_chung_khoan`, `doanh_nghiep`, `ngan_hang`, `bao_cao_phan_tich`.

### 5b. Corporate events theo mã cổ phiếu

Crawl tin tức sự kiện doanh nghiệp từ CafeF Events API, `detected_tickers` được gán sẵn:

```bash
# VN30 (mặc định)
python -m src.jobs.crawl_corporate_events

# Danh sách mã chỉ định
python -m src.jobs.crawl_corporate_events --tickers FPT,VCB,HPG --min-year 2024

# Toàn bộ mã từ ticker_master
python -m src.jobs.crawl_corporate_events --from-ticker-master --max-pages 5
```

---

## 6. Relevance filter

Phân loại bài thành `process_sentiment`, `review_later`, `skip_sentiment` bằng rule-based scoring:

```bash
python -m src.jobs.run_relevance_filter
```

---

## 7. Entity linking & aspect extraction

```bash
python -m src.jobs.build_article_entities
python -m src.jobs.build_entity_aspects
```

---

## 8. Sentiment inference

### Local model (khuyến nghị — không cần server)

Cần đặt `SENTIMENT_MODEL_PATH` trong `.env`, hoặc truyền trực tiếp:

```bash
python -m src.jobs.run_model_inference \
  --local-model-path ../stock_news/models/finetuned_last_layer_vn30_content
```

Model là PhoBERT finetuned (`wonrax/phobert-base-vietnamese-sentiment`), output gồm `prob_negative`, `prob_neutral`, `prob_positive` cho từng entity-aspect.

### FastAPI server (tùy chọn)

Nếu có API server, đặt `SENTIMENT_MODEL_API_URL` trong `.env` và để `SENTIMENT_MODEL_PATH` trống:

```bash
python -m src.jobs.run_model_inference --model-version finetuned-v1
```

---

## 9. Aggregate sentiment

```bash
python -m src.jobs.build_sentiment_aggregates
```

---

## 10. Market data

```bash
# Fetch giá lịch sử
python -m src.jobs.fetch_market_data --start-date 2025-01-01 --tickers FPT,VCB,HPG

# Tính biến thị trường (return, volatility, ...)
python -m src.jobs.build_market_features

# Kiểm định tương quan sentiment vs giá
python -m src.jobs.run_market_validation_pipeline \
  --start-date 2025-01-01 --tickers FPT,VCB,HPG
```

---

## 11. Export daily panel CSV

Xuất CSV phẳng từ PostgreSQL để dùng cho các analysis script:

```bash
# Toàn bộ dữ liệu
python -m src.jobs.export_daily_panel

# Lọc theo ngày và mã
python -m src.jobs.export_daily_panel --start-date 2025-01-01 --tickers FPT,VCB,HPG

# Chỉ định đường dẫn output
python -m src.jobs.export_daily_panel --output data/my_panel.csv
```

Output CSV có các cột: `date`, `ticker`, `sector`, `sentiment_score`, `news_count`, `log_return`, `volume_growth`, `clv`, `return_1d`, `forward_return_1d`, `target_up`, ... (PostgreSQL là nguồn lưu trữ chính, CSV chỉ là artifact tạm thời cho analysis).

---

## 12. Analysis

Sau khi export panel CSV, chạy các analysis script trong `src/analysis/`. Tất cả đều nhận `--panel` (mặc định `data/processed/validation/daily_panel.csv`) và ghi kết quả vào `data/results/`.

### Cross-correlations (sentiment vs trading metrics)

```bash
python -m src.analysis.cross_correlations
python -m src.analysis.cross_correlations --news-days-only --ccf-output data/results/ccf_news_days.csv
```

### Granger causality

```bash
# OLS Granger F-test
python -m src.analysis.granger_sentiment

# VAR-based Granger
python -m src.analysis.var_granger_sentiment
```

### Binary confusion: sentiment direction vs stock direction

```bash
python -m src.analysis.sentiment_target_confusion --lags 0 1 2 3 4 5
```

### Event study (abnormal returns around news events)

```bash
python -m src.analysis.event_study_sentiment --horizons 1 3 5 10 --threshold 0.0
```

### ML model comparison

```bash
# Rolling-window feature ML (Logistic, RF, GB)
python -m src.analysis.compare_ml_models

# Granger-style restricted vs sentiment-augmented (regression + classification)
python -m src.analysis.compare_granger_style_models
```

### Visualization

```bash
python -m src.analysis.visualize_sentiment_price --tickers FPT VCB HPG --lag 1
# → data/results/plots/FPT_sentiment_price_timeseries.png, ...
```

Tất cả script đều hỗ trợ `--news-days-only` để chỉ dùng ngày có tin tức.

---

## 13. Full pipeline

```bash
# Chạy toàn bộ pipeline backfill
python -m src.jobs.run_full_pipeline \
  --mode backfill \
  --max-per-category 200 \
  --timeline-pages 10 \
  --reset-stage2

# Kèm kiểm định thị trường
python -m src.jobs.run_full_pipeline \
  --mode backfill \
  --max-per-category 3500 \
  --timeline-pages 100 \
  --reset-stage2 \
  --run-market-validation \
  --market-start-date 2025-01-01 \
  --market-tickers FPT,VCB,HPG,VIC,MBB,TCB
```

---

## 14. Dashboard

```bash
streamlit run src/dashboard/app_sentiment_dashboard.py
```

---

## QA report

```bash
python -m src.jobs.qa_report
```

Ví dụ output:

```text
===== QA REPORT =====
Tổng số bài: 300
Số bài theo category:
  doanh_nghiep              100
  thi_truong_chung_khoan    100
  ngan_hang                  80
  bao_cao_phan_tich          20

Số bài theo decision:
  process_sentiment    160
  review_later          60
  skip_sentiment         80
```

---

## Ghi chú kỹ thuật

- Crawler có retry, timeout, user-agent và delay (configurable qua `.env`).
- Parser ưu tiên BeautifulSoup selector, fallback sang trafilatura nếu content < 300 ký tự.
- Upsert theo URL — chạy lại không bị trùng bài.
- Relevance filter: rule-based keyword + ticker scoring, không dùng LLM.
- `crawl_backfill` dùng `--min-year` (default 2024) để dừng sớm khi gặp bài cũ.
- `crawl_backfill --workers N` fetch N bài song song bằng `ThreadPoolExecutor`; flush DB sau mỗi 100 bài nên Ctrl+C không mất data.
- `crawl_corporate_events` gán `detected_tickers` sẵn từ Events API, bỏ qua bước entity linking cho những mã đó.
- Local inference dùng model HuggingFace từ thư mục local, không cần API server.
