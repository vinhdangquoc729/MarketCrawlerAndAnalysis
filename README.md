# Market Sentiment — End-to-End (VN30)

Hệ thống thu thập tin tức tài chính Việt Nam, phân tích sentiment bằng PhoBERT finetuned, và kiểm định mối quan hệ giữa sentiment với biến động giá cổ phiếu VN30.

```
CafeF news / corporate events
  → crawl & parse bài viết
  → PostgreSQL
  → relevance filter (rule-based)
  → entity linking (mã cổ phiếu)
  → aspect extraction
  → sentiment inference (PhoBERT finetuned, local)
  → aggregate sentiment theo mã / ngày
  → fetch giá thị trường (vnstock)
  → build market features (log_return, volume_growth, ...)
  → export daily panel CSV
  → analysis (cross-correlation, Granger, VAR, event study, ML)
  → dashboard (Streamlit)
```

---

## Cấu trúc thư mục

```
src/
├── crawlers/          # CafeF HTML crawler + parser + events crawler
├── jobs/              # Entry-point scripts cho từng bước pipeline
├── nlp/               # Entity linker, aspect extractor, sentiment input builder
├── inference/         # LocalSentimentModel (PhoBERT) + HTTP client
├── market_data/       # vnstock provider, market feature builder
├── master_data/       # Ticker master, alias builder
├── storage/           # DB layer (SQLAlchemy + psycopg2)
├── preprocessing/     # Relevance filter, text cleaner
├── analysis/          # Cross-corr, Granger, VAR, event study, ML, analog forecast
├── dashboard/         # Streamlit app
├── validation/        # QA queries
└── label/             # PhoBERT finetuning scripts
models/                # Finetuned PhoBERT checkpoints
data/                  # Panel CSV, results
docker-compose.yml     # PostgreSQL container
requirements.txt
.env.example
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

## 2. Cấu hình `.env`

```bash
# Windows
copy .env.example .env
# macOS/Linux
cp .env.example .env
```

Chỉnh `.env`:

```env
DB_HOST=127.0.0.1
DB_PORT=5434
DB_NAME=market_sentiment
DB_USER=postgres
DB_PASSWORD=postgres
DB_TIMEZONE=Asia/Ho_Chi_Minh

# Đường dẫn đến thư mục model finetuneds
SENTIMENT_MODEL_PATH=models/finetuned_last_layer_vn30_content
```

---

## 3. Chạy PostgreSQL

```bash
docker compose up -d
```

> **Windows:** nếu port 5432/5433 đã bị local PostgreSQL chiếm, đổi port mapping trong `docker-compose.yml` và `DB_PORT` trong `.env`.

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

```bash
# Backfill từ 2024
python -m src.jobs.crawl_backfill --max-per-category 200 --timeline-pages 10

# Song song, không giới hạn
python -m src.jobs.crawl_backfill --min-year 2024 --timeline-pages 2000 --workers 50
```

Chuyên mục có sẵn: `thi_truong_chung_khoan`, `doanh_nghiep`, `ngan_hang`, `bao_cao_phan_tich`.

### 5b. Corporate events theo mã cổ phiếu

```bash
# VN30 (mặc định)
python -m src.jobs.crawl_corporate_events

# Mã chỉ định
python -m src.jobs.crawl_corporate_events --tickers FPT,VCB,HPG --min-year 2024
```

---

## 6. Relevance filter

```bash
python -m src.jobs.run_relevance_filter
```

Phân loại bài thành `process_sentiment`, `review_later`, `skip_sentiment` bằng rule-based keyword scoring.

---

## 7. Entity linking & aspect extraction

```bash
# Chỉ VN30, 4 workers song song
python -m src.jobs.build_article_entities --vn30-only --workers 4

python -m src.jobs.build_entity_aspects
```

---

## 8. Sentiment inference

Model: **PhoBERT finetuned** (`wonrax/phobert-base-vietnamese-sentiment`, finetuned last encoder layer + classifier head trên dữ liệu VN30 với format `[STOCK]...[/STOCK][TEXT]...[/TEXT]`).

```bash
python -m src.jobs.run_model_inference \
  --local-model-path models/finetuned_last_layer_vn30_content
```

Output: `prob_negative`, `prob_neutral`, `prob_positive` cho từng entity-aspect pair.

---

## 9. Aggregate sentiment

```bash
python -m src.jobs.build_sentiment_aggregates
```

---

## 10. Market data

```bash
# Fetch giá lịch sử (delay 4s/request để tránh rate limit vnstock)
python -m src.jobs.fetch_market_data --start-date 2024-01-01 --delay-seconds 4

# Tính market features (log_return, volume_growth, CLV, ...)
python -m src.jobs.build_market_features
```

---

## 11. Export daily panel

```bash
# Xuất CSV phẳng từ PostgreSQL
python -m src.jobs.export_daily_panel

# Lọc theo ngày / mã
python -m src.jobs.export_daily_panel --start-date 2024-01-01 --tickers FPT,VCB,HPG
```

Cột output: `date`, `ticker`, `sector`, `sentiment_score`, `news_count`, `log_return`, `volume_growth`, `clv`, `target_up`, ...

---

## 12. Analysis

Tất cả script nhận `--panel` (default `data/processed/validation/daily_panel.csv`) và ghi kết quả vào `data/results/`.

```bash
# Cross-correlations (sentiment vs return/volume theo lag)
python -m src.analysis.cross_correlations

# Granger causality OLS F-test
python -m src.analysis.granger_sentiment

# VAR-based Granger
python -m src.analysis.var_granger_sentiment

# Binary confusion: sentiment direction vs stock direction
python -m src.analysis.sentiment_target_confusion --lags -5 -4 -3 -2 -1 0 1 2 3 4 5

# Event study: abnormal returns sau ngày có tin POS/NEG/NEU
python -m src.analysis.event_study_sentiment --horizons 1 3 5 10

# ML model comparison (with vs without sentiment)
python -m src.analysis.compare_ml_models
python -m src.analysis.compare_granger_style_models

# Analog forecasting: tìm ngày lịch sử tương đồng
python -m src.analysis.analog_forecast --window 5 --top-k 15 --horizon 5
```

Flag `--news-days-only` có thể thêm vào hầu hết các script để chỉ dùng ngày có tin tức.

---

## 13. Dashboard

```bash
streamlit run src/dashboard/app_sentiment_dashboard.py
```

### Các tab

| Tab | Nội dung |
|-----|----------|
| 📊 Overview | Bảng sentiment tổng hợp theo mã / ngày |
| 🔎 Ticker Detail | Sentiment + giá chi tiết theo mã |
| 📈 Market | Biểu đồ market index, volatility |
| 📰 News | Danh sách bài viết + sentiment |
| 🧾 Explain Evidence | Giải thích aspect-level sentiment |
| 🧪 Market Validation | Kiểm định tương quan tổng hợp |
| 🛠 Data Quality | QA queries, phát hiện data issues |
| 🔬 Research | Cross-corr, Granger, VAR, Event Study, Confusion Matrix, ML models |
| 📡 Tín hiệu hôm nay | Analog forecasting: tìm ngày lịch sử tương đồng → dự báo chiều giá |

### Tab "Tín hiệu hôm nay"

Dựa trên dữ liệu từ 2024. Với mỗi mã VN30:
1. Lấy vector đặc trưng 5 ngày gần nhất: `(log_return, volume_growth, sentiment_score)`
2. Chuẩn hóa within-window (z-score) để tránh temporal proximity bias
3. Tìm top-K ngày lịch sử có pattern gần nhất (Euclidean distance)
4. Tính **win rate** (tỷ lệ ngày tương đồng → giá tăng sau H ngày)
5. Hiển thị signal: 🟢 BULLISH / 🔴 BEARISH / ⚪ NEUTRAL

---

## 14. Full pipeline (one-shot)

```bash
python -m src.jobs.run_full_pipeline \
  --mode backfill \
  --max-per-category 3500 \
  --timeline-pages 100 \
  --reset-stage2 \
  --run-market-validation \
  --market-start-date 2024-01-01
```

---

## 15. Model finetuning

```bash
# Chuẩn bị labels
python -m src.label.prepare_labels --csv data/labels/raw_labels.csv

# Finetune PhoBERT (last encoder layer + classifier head)
python -m src.label.finetune \
  --csv data/labels/context_labels.csv \
  --epochs 15 --lr 1e-5
```

Base model: `wonrax/phobert-base-vietnamese-sentiment`
Input format: `[STOCK] {ticker} [/STOCK] [TEXT] {title}. {content} [/TEXT]`

---

## Ghi chú kỹ thuật

- Crawler: retry, timeout, configurable delay, user-agent rotation.
- Parser: BeautifulSoup selector, fallback sang trafilatura nếu content < 300 ký tự.
- Upsert theo URL — chạy lại không bị trùng bài.
- `crawl_backfill --workers N`: `ThreadPoolExecutor`, flush DB sau mỗi 100 bài.
- `crawl_corporate_events`: gán `detected_tickers` từ Events API, bỏ qua entity linking.
- vnstock rate limit: guest tier ~20 req/min → dùng `--delay-seconds 4`.
- Subprocess output trong dashboard dùng `PYTHONIOENCODING=utf-8` để xử lý tiếng Việt trên Windows.

Nguồn data và model sau khi đã fine-tune: https://drive.google.com/drive/folders/1CqV7UHSDvs4jAW4hwor6PqBEqYmK8Ry4?usp=sharing