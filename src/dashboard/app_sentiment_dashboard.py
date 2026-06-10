"""Streamlit dashboard for Market Sentiment explainability and validation."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Market Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# DB CONNECTION
# =========================================================

@st.cache_resource
def get_engine() -> Engine:
    """Create SQLAlchemy engine from .env."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    name = os.getenv("DB_NAME", "market_sentiment")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    timezone = os.getenv("DB_TIMEZONE", "Asia/Ho_Chi_Minh")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"

    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"options": f"-c timezone={timezone}"},
    )


def read_sql(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read SQL safely."""
    engine = get_engine()
    return pd.read_sql(text(sql), engine, params=params)


def table_exists(table_or_view: str) -> bool:
    """Check if table or view exists."""
    sql = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = :name

        UNION ALL

        SELECT 1
        FROM information_schema.views
        WHERE table_schema = 'public'
          AND table_name = :name
    ) AS exists
    """
    try:
        df = read_sql(sql, {"name": table_or_view})
        return bool(df["exists"].iloc[0])
    except Exception:
        return False


# =========================================================
# DATA LOADERS
# =========================================================

@st.cache_data(ttl=120)
def load_daily_sentiment() -> pd.DataFrame:
    if not table_exists("daily_sentiment_index"):
        return pd.DataFrame()

    return read_sql(
        """
        SELECT
            date,
            ticker,
            sector,
            sentiment_index,
            final_sentiment,
            article_count,
            positive_count,
            neutral_count,
            negative_count,
            avg_confidence,
            trust_level,
            created_at
        FROM daily_sentiment_index
        ORDER BY date ASC, ticker ASC
        """
    )


@st.cache_data(ttl=120)
def load_evidence(ticker: str | None = None, limit: int = 200) -> pd.DataFrame:
    if not table_exists("sentiment_evidence_view"):
        return pd.DataFrame()

    if ticker and ticker != "ALL":
        return read_sql(
            """
            SELECT
                title,
                url,
                category,
                published_at_vn,
                ticker,
                company_name,
                sector,
                entity_text,
                aspect,
                aspect_keywords,
                aspect_score,
                sentiment_label,
                sentiment_score,
                confidence,
                prob_negative,
                prob_neutral,
                prob_positive,
                context,
                model_version,
                created_at
            FROM sentiment_evidence_view
            WHERE ticker = :ticker
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"ticker": ticker, "limit": limit},
        )

    return read_sql(
        """
        SELECT
            title,
            url,
            category,
            published_at_vn,
            ticker,
            company_name,
            sector,
            entity_text,
            aspect,
            aspect_keywords,
            aspect_score,
            sentiment_label,
            sentiment_score,
            confidence,
            prob_negative,
            prob_neutral,
            prob_positive,
            context,
            model_version,
            created_at
        FROM sentiment_evidence_view
        ORDER BY created_at DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


@st.cache_data(ttl=120)
def load_market_validation() -> pd.DataFrame:
    if not table_exists("sentiment_market_forward_dataset"):
        return pd.DataFrame()

    return read_sql(
        """
        SELECT *
        FROM sentiment_market_forward_dataset
        ORDER BY date ASC, ticker ASC
        """
    )


@st.cache_data(ttl=120)
def load_aspect_market_validation() -> pd.DataFrame:
    if not table_exists("sentiment_aspect_market_dataset"):
        return pd.DataFrame()

    return read_sql(
        """
        SELECT *
        FROM sentiment_aspect_market_dataset
        ORDER BY date ASC, ticker ASC, aspect ASC
        """
    )


@st.cache_data(ttl=120)
def load_market_prices(ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
    if not table_exists("market_prices"):
        return pd.DataFrame()
    filters = ["ticker = :ticker"]
    params: dict = {"ticker": ticker}
    if start_date:
        filters.append("date >= :start_date")
        params["start_date"] = str(start_date)
    if end_date:
        filters.append("date <= :end_date")
        params["end_date"] = str(end_date)
    where = "WHERE " + " AND ".join(filters)
    return read_sql(
        f"SELECT date, open, high, low, close, volume FROM market_prices {where} ORDER BY date ASC",
        params,
    )


@st.cache_data(ttl=120)
def load_market_prices_summary() -> pd.DataFrame:
    if not table_exists("market_prices"):
        return pd.DataFrame()

    return read_sql(
        """
        SELECT
            ticker,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(*) AS row_count
        FROM market_prices
        GROUP BY ticker
        ORDER BY ticker
        """
    )


@st.cache_data(ttl=120)
def load_news(
    ticker: str | None = None,
    sentiment: str | None = None,
    start_date=None,
    end_date=None,
    page: int = 0,
    page_size: int = 20,
) -> tuple[pd.DataFrame, int]:
    """Load paginated articles with their dominant sentiment."""
    filters = []
    params: dict = {"limit": page_size, "offset": page * page_size}

    if ticker and ticker != "ALL":
        filters.append("ae.ticker = :ticker")
        params["ticker"] = ticker

    if sentiment and sentiment != "ALL":
        filters.append("es.sentiment_label = :sentiment")
        params["sentiment"] = sentiment

    if start_date:
        filters.append("a.published_at::date >= :start_date")
        params["start_date"] = str(start_date)

    if end_date:
        filters.append("a.published_at::date <= :end_date")
        params["end_date"] = str(end_date)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    base_sql = f"""
        SELECT * FROM (
            SELECT DISTINCT ON (a.article_id)
                a.article_id,
                a.title,
                a.url,
                a.category,
                a.published_at AT TIME ZONE 'Asia/Ho_Chi_Minh' AS published_at_vn,
                ae.ticker,
                es.sentiment_label,
                ROUND(es.sentiment_score::numeric, 3) AS sentiment_score,
                ROUND(es.confidence::numeric, 3) AS confidence,
                ROUND(es.prob_positive::numeric, 3) AS prob_positive,
                ROUND(es.prob_neutral::numeric, 3) AS prob_neutral,
                ROUND(es.prob_negative::numeric, 3) AS prob_negative
            FROM articles a
            JOIN article_entities ae ON a.article_id = ae.article_id
            JOIN entity_aspects ea ON ae.id = ea.entity_id
            JOIN entity_sentiments es ON ea.id = es.entity_aspect_id
            {where}
            ORDER BY a.article_id, es.confidence DESC
        ) sub
        ORDER BY published_at_vn DESC
    """

    count_sql = f"""
        SELECT COUNT(DISTINCT a.article_id)
        FROM articles a
        JOIN article_entities ae ON a.article_id = ae.article_id
        JOIN entity_aspects ea ON ae.id = ea.entity_id
        JOIN entity_sentiments es ON ea.id = es.entity_aspect_id
        {where}
    """

    try:
        total = read_sql(count_sql, params).iloc[0, 0]
        paged_sql = base_sql + " LIMIT :limit OFFSET :offset"
        df = read_sql(paged_sql, params)
        return df, int(total)
    except Exception:
        return pd.DataFrame(), 0


@st.cache_data(ttl=120)
def load_future_date_articles() -> pd.DataFrame:
    """Detect articles whose published_at is suspiciously in the future."""
    if not table_exists("articles"):
        return pd.DataFrame()

    try:
        return read_sql(
            """
            SELECT
                title,
                published_at,
                crawl_at,
                url
            FROM articles
            WHERE published_at > CURRENT_DATE + INTERVAL '7 days'
            ORDER BY published_at DESC
            LIMIT 100
            """
        )
    except Exception:
        return pd.DataFrame()


# =========================================================
# UPDATE PIPELINE
# =========================================================

import threading
import datetime as _dt

# _PIPELINE_STATE is a plain dict the background thread writes to directly
# (background threads have no Streamlit run context so st.session_state is
# inaccessible from them). We store the dict inside st.session_state so it
# survives across reruns — Streamlit does not deep-copy session state objects,
# so the background thread and the main thread always share the exact same
# Python dict object. The assignment below runs on every rerun but only
# creates the dict once; after that it just re-binds the name to the same object.
if "_pipeline_state" not in st.session_state:
    st.session_state["_pipeline_state"] = {
        "active": False,
        "running": False,
        "done": False,
        "success": False,
        "steps": [],
    }
_PIPELINE_STATE: dict = st.session_state["_pipeline_state"]


@st.cache_data(ttl=30)
def get_last_update_info() -> dict:
    info: dict = {"last_crawl": None, "article_count": 0, "last_market_date": None}
    try:
        df = read_sql(
            "SELECT MAX(crawl_at) AT TIME ZONE 'Asia/Ho_Chi_Minh' AS last_crawl, COUNT(*) AS total FROM articles"
        )
        if not df.empty and df["last_crawl"].iloc[0] is not None:
            info["last_crawl"] = pd.to_datetime(df["last_crawl"].iloc[0])
            info["article_count"] = int(df["total"].iloc[0])
    except Exception:
        pass
    try:
        if table_exists("market_prices"):
            df2 = read_sql("SELECT MAX(date) AS last_date FROM market_prices")
            if not df2.empty and df2["last_date"].iloc[0] is not None:
                info["last_market_date"] = pd.to_datetime(df2["last_date"].iloc[0]).date()
    except Exception:
        pass
    return info


def _pipeline_worker(steps: list, project_root: str) -> None:
    """Background thread: writes progress to _PIPELINE_STATE (no st.* calls)."""
    for i, (_, cmd) in enumerate(steps):
        _PIPELINE_STATE["steps"][i]["status"] = "running"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-m"] + cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=project_root,
            env=env,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            _PIPELINE_STATE["steps"][i]["status"] = "error"
            _PIPELINE_STATE["steps"][i]["stderr"] = combined[-3000:]
            _PIPELINE_STATE["running"] = False
            _PIPELINE_STATE["done"] = True
            return
        _PIPELINE_STATE["steps"][i]["status"] = "done"
        _PIPELINE_STATE["steps"][i]["stderr"] = combined[-3000:]

    _PIPELINE_STATE["running"] = False
    _PIPELINE_STATE["done"] = True
    _PIPELINE_STATE["success"] = True


def start_update_pipeline(last_market_date) -> None:
    """Initialise _PIPELINE_STATE and kick off the background thread."""
    market_start = (
        str(last_market_date)
        if last_market_date is not None
        else str(_dt.date.today() - _dt.timedelta(days=30))
    )

    steps: list[tuple[str, list[str]]] = [
        ("Crawl tin tức mới",        ["src.jobs.crawl_backfill", "--timeline-pages", "6", "--max-per-category", "100"]),
        ("Gắn thực thể bài báo",     ["src.jobs.build_article_entities", "--since-date", market_start, "--vn30-only", "--workers", "10"]),
        ("Phân tích khía cạnh",      ["src.jobs.build_entity_aspects"]),
        ("Lọc bài liên quan",        ["src.jobs.run_relevance_filter"]),
        ("Chạy inference",           ["src.jobs.run_model_inference"]
                                     + (["--local-model-path", os.getenv("SENTIMENT_MODEL_PATH")]
                                        if os.getenv("SENTIMENT_MODEL_PATH") else [])),
        ("Tổng hợp sentiment",       ["src.jobs.build_sentiment_aggregates", "--start-date", market_start]),
        ("Cập nhật dữ liệu giá",     ["src.jobs.fetch_market_data", "--start-date", market_start, "--allow-empty", "--delay-seconds", "4"]),
        ("Xây dựng market features", ["src.jobs.build_market_features", "--start-date", market_start]),
    ]

    _PIPELINE_STATE["steps"] = [
        {"label": label, "status": "pending", "stderr": ""} for label, _ in steps
    ]
    _PIPELINE_STATE["active"] = True
    _PIPELINE_STATE["running"] = True
    _PIPELINE_STATE["done"] = False
    _PIPELINE_STATE["success"] = False

    threading.Thread(
        target=_pipeline_worker,
        args=(steps, str(PROJECT_ROOT)),
        daemon=True,
    ).start()


def render_update_progress() -> None:
    """Render live pipeline progress in the sidebar (reads from _PIPELINE_STATE)."""
    steps = _PIPELINE_STATE["steps"]
    running = _PIPELINE_STATE["running"]
    done = _PIPELINE_STATE["done"]
    success = _PIPELINE_STATE["success"]

    icons = {"pending": "⏸", "running": "⏳", "done": "✅", "error": "❌"}

    if running:
        st.sidebar.info("Đang cập nhật...", icon="🔄")
    elif done and success:
        st.sidebar.success("Cập nhật hoàn tất!")
    elif done:
        st.sidebar.error("Cập nhật thất bại!")

    for step in steps:
        icon = icons.get(step["status"], "?")
        st.sidebar.write(f"{icon} {step['label']}")
        if step.get("stderr"):
            label = "Chi tiết lỗi" if step["status"] == "error" else "Chi tiết"
            with st.sidebar.expander(label):
                st.code(step["stderr"], language="text")

    if done:
        if st.sidebar.button("Đóng", key="_upd_close_btn"):
            _PIPELINE_STATE["active"] = False
            _PIPELINE_STATE["steps"] = []
            if success:
                st.cache_data.clear()
            st.rerun()


# =========================================================
# UI HELPERS
# =========================================================

def format_pct(x: float | int | None) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x) * 100:.2f}%"


def format_score(x: float | int | None) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.3f}"


def format_date_value(value) -> str:
    if pd.isna(value):
        return "-"
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def sentiment_badge(label: str) -> str:
    if label == "positive":
        return "🟢 positive"
    if label == "negative":
        return "🔴 negative"
    return "⚪ neutral"


def apply_common_filters(
    df: pd.DataFrame,
    start_date,
    end_date,
    selected_ticker: str,
    selected_sector: str,
) -> pd.DataFrame:
    """Apply sidebar filters to any dataframe with date/ticker/sector."""
    if df.empty:
        return df

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])

    result = result[
        (result["date"].dt.date >= start_date)
        & (result["date"].dt.date <= end_date)
    ]

    if selected_ticker != "ALL" and "ticker" in result.columns:
        result = result[result["ticker"] == selected_ticker]

    if selected_sector != "ALL" and "sector" in result.columns:
        result = result[result["sector"] == selected_sector]

    return result


def calculate_directional_hit_rate(df: pd.DataFrame, target_col: str) -> tuple[float | None, int]:
    """Positive sentiment expects positive return, negative sentiment expects negative return."""
    if df.empty or target_col not in df.columns:
        return None, 0

    directional_df = df[df["final_sentiment"].isin(["positive", "negative"])].copy()
    directional_df = directional_df.dropna(subset=["final_sentiment", target_col])

    if directional_df.empty:
        return None, 0

    directional_df["hit"] = (
        ((directional_df["final_sentiment"] == "positive") & (directional_df[target_col] > 0))
        | ((directional_df["final_sentiment"] == "negative") & (directional_df[target_col] < 0))
    )

    return float(directional_df["hit"].mean()), len(directional_df)


# =========================================================
# MAIN APP
# =========================================================

st.title("📈 Market Sentiment Dashboard")
st.caption(
    "Dashboard giải thích output model sentiment theo mã cổ phiếu, aspect, evidence bài báo "
    "và kiểm định với biến thị trường."
)

with st.sidebar:
    st.header("⚙️ Bộ lọc")

    _update_info = get_last_update_info()
    if _update_info["last_crawl"] is not None:
        st.caption(f"📰 Tin tức: {_update_info['last_crawl'].strftime('%d/%m/%Y %H:%M')}  ({_update_info['article_count']:,} bài)")
    if _update_info["last_market_date"] is not None:
        st.caption(f"📈 Giá: {_update_info['last_market_date'].strftime('%d/%m/%Y')}")

    _do_update = st.button("🔄 Cập nhật thông tin mới nhất", width="stretch")

    if st.button("♻️ Refresh cache", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    try:
        db_check = read_sql("SELECT NOW() AS db_now, current_setting('TimeZone') AS timezone")
        st.success("DB connected")
        st.caption(f"Timezone: {db_check['timezone'].iloc[0]}")
    except Exception as exc:
        st.error("Không kết nối được DB")
        st.exception(exc)
        st.stop()


daily_df = load_daily_sentiment()
pipeline_ready = not daily_df.empty

if not pipeline_ready:
    with st.expander("ℹ️ Pipeline chưa hoàn tất — Overview / Ticker / Validation chưa khả dụng", expanded=False):
        st.info(
            "Chưa có dữ liệu trong `daily_sentiment_index`. "
            "Chạy `build_sentiment_aggregates` để mở khoá các tab còn lại. "
            "Tab 📰 News và 🧾 Explain Evidence hoạt động bình thường."
        )
        st.code("python -m src.jobs.build_sentiment_aggregates", language="bash")

if pipeline_ready:
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    future_articles_df = load_future_date_articles()
    if not future_articles_df.empty:
        with st.expander("⚠️ Cảnh báo: Có bài báo bị parse ngày tương lai", expanded=False):
            st.warning(
                "DB đang có bài báo có published_at nằm trong tương lai. "
                "Khả năng cao parser ngày CafeF bị đảo DD-MM-YYYY thành MM-DD-YYYY. "
                "Hãy sửa cafef_parser.py với dayfirst=True rồi reset/crawl lại DB."
            )
            st.dataframe(future_articles_df, width="stretch", hide_index=True)

    available_tickers = ["ALL"] + sorted(daily_df["ticker"].dropna().unique().tolist())
    available_sectors = ["ALL"] + sorted(daily_df["sector"].dropna().unique().tolist())
    min_date = daily_df["date"].min().date()
    max_date = daily_df["date"].max().date()
else:
    import datetime
    try:
        _tickers = read_sql("SELECT DISTINCT ticker FROM entity_sentiments ORDER BY ticker")["ticker"].tolist()
        available_tickers = ["ALL"] + _tickers
    except Exception:
        available_tickers = ["ALL"]
    available_sectors = ["ALL"]
    min_date = datetime.date(2024, 1, 1)
    max_date = datetime.date.today()

with st.sidebar:
    if pipeline_ready:
        selected_ticker = st.selectbox("Ticker", available_tickers)
        selected_sector = st.selectbox("Sector", available_sectors)
    else:
        selected_ticker = "ALL"
        selected_sector = "ALL"

    import datetime as _datetime_mod
    date_range = st.date_input(
        "Khoảng ngày",
        value=(min_date, _datetime_mod.date.today()),
        min_value=min_date,
        max_value=_datetime_mod.date.today(),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

if pipeline_ready:
    filtered_df = apply_common_filters(
        daily_df,
        start_date=start_date,
        end_date=end_date,
        selected_ticker=selected_ticker,
        selected_sector=selected_sector,
    )
else:
    filtered_df = pd.DataFrame()


# =========================================================
# UPDATE EXECUTION
# =========================================================

if _do_update and not _PIPELINE_STATE["running"]:
    start_update_pipeline(_update_info["last_market_date"])
    st.rerun()

if _PIPELINE_STATE["active"]:
    render_update_progress()


# =========================================================
# TAB LAYOUT
# =========================================================

tab_overview, tab_ticker, tab_market, tab_news, tab_evidence, tab_validation, tab_data_quality, tab_research, tab_signal = st.tabs(
    [
        "📊 Overview",
        "🔎 Ticker Detail",
        "📈 Market",
        "📰 News",
        "🧾 Explain Evidence",
        "🧪 Market Validation",
        "🛠 Data Quality",
        "🔬 Research",
        "📡 Tín hiệu hôm nay",
    ]
)


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab_overview:
    st.subheader("Tổng quan sentiment")

    if filtered_df.empty:
        st.caption("Chưa có dữ liệu — chạy `python -m src.jobs.build_sentiment_aggregates` trước.")

    latest_date = filtered_df["date"].max() if not filtered_df.empty else None
    if latest_date is not None:
        latest_df = filtered_df[filtered_df["date"] == latest_date].copy()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ngày mới nhất", format_date_value(latest_date))
        c2.metric("Số mã", safe_int(latest_df["ticker"].nunique()))
        c3.metric("Positive", safe_int((latest_df["final_sentiment"] == "positive").sum()))
        c4.metric("Neutral", safe_int((latest_df["final_sentiment"] == "neutral").sum()))
        c5.metric("Negative", safe_int((latest_df["final_sentiment"] == "negative").sum()))

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Top positive")
            top_pos = latest_df.sort_values("sentiment_index", ascending=False).head(10)
            if top_pos.empty:
                st.info("Không có dữ liệu.")
            else:
                fig = px.bar(
                    top_pos, x="ticker", y="sentiment_index",
                    hover_data=["sector", "final_sentiment", "article_count",
                                "avg_confidence", "positive_count", "neutral_count", "negative_count"],
                    title="Top mã sentiment tích cực",
                )
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, width="stretch")

        with col_right:
            st.markdown("### Top negative")
            top_neg = latest_df.sort_values("sentiment_index", ascending=True).head(10)
            if top_neg.empty:
                st.info("Không có dữ liệu.")
            else:
                fig = px.bar(
                    top_neg, x="ticker", y="sentiment_index",
                    hover_data=["sector", "final_sentiment", "article_count",
                                "avg_confidence", "positive_count", "neutral_count", "negative_count"],
                    title="Top mã sentiment tiêu cực",
                )
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, width="stretch")

        st.markdown("### Sentiment trung bình theo ngành")
        sector_df = (
            filtered_df.groupby(["date", "sector"], as_index=False)
            .agg(sentiment_index=("sentiment_index", "mean"),
                 article_count=("article_count", "sum"),
                 avg_confidence=("avg_confidence", "mean"),
                 ticker_count=("ticker", "nunique"))
        )
        if sector_df.empty:
            st.info("Không có dữ liệu ngành.")
        else:
            fig = px.line(sector_df, x="date", y="sentiment_index", color="sector",
                          markers=True, hover_data=["article_count", "avg_confidence", "ticker_count"],
                          title="Sector Sentiment Over Time")
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, width="stretch")

        st.markdown("### Bảng sentiment mới nhất")
        st.dataframe(latest_df.sort_values("sentiment_index", ascending=False),
                     width="stretch", hide_index=True)


# =========================================================
# TAB 2: TICKER DETAIL
# =========================================================

with tab_ticker:
    st.subheader("Chi tiết theo mã cổ phiếu")

    ticker_options = sorted(filtered_df["ticker"].dropna().unique().tolist()) if not filtered_df.empty else []

    if not ticker_options:
        st.warning("Không có ticker phù hợp với bộ lọc.")
    else:
        default_ticker = (
            selected_ticker
            if selected_ticker != "ALL" and selected_ticker in ticker_options
            else ticker_options[0]
        )

        ticker = st.selectbox(
            "Chọn mã để phân tích",
            ticker_options,
            index=ticker_options.index(default_ticker),
            key="ticker_detail_select",
        )

        ticker_df = filtered_df[filtered_df["ticker"] == ticker].sort_values("date").copy()

        if ticker_df.empty:
            st.warning("Không có dữ liệu cho mã này.")
        else:
            latest_row = ticker_df.sort_values("date").iloc[-1]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Ticker", ticker)
            c2.metric("Sentiment", format_score(latest_row["sentiment_index"]))
            c3.metric("Final", sentiment_badge(str(latest_row["final_sentiment"])))
            c4.metric("Article count", safe_int(latest_row["article_count"]))
            c5.metric("Avg confidence", format_score(latest_row["avg_confidence"]))

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["sentiment_index"],
                    mode="lines+markers",
                    name="sentiment_index",
                )
            )

            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                title=f"Sentiment Index theo ngày - {ticker}",
                xaxis_title="Date",
                yaxis_title="Sentiment index",
                height=480,
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("### Breakdown positive / neutral / negative")

            count_df = ticker_df[
                [
                    "date",
                    "positive_count",
                    "neutral_count",
                    "negative_count",
                ]
            ].melt(
                id_vars="date",
                var_name="sentiment_type",
                value_name="count",
            )

            fig = px.bar(
                count_df,
                x="date",
                y="count",
                color="sentiment_type",
                title=f"Sentiment count breakdown - {ticker}",
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("### Dữ liệu chi tiết")
            st.dataframe(
                ticker_df.sort_values("date", ascending=False),
                width="stretch",
                hide_index=True,
            )


# =========================================================
# TAB 3: MARKET
# =========================================================

with tab_market:
    st.subheader("Dữ liệu thị trường")

    market_ticker_options = (
        sorted(daily_df["ticker"].dropna().unique().tolist()) if pipeline_ready else []
    )

    if not market_ticker_options:
        st.warning("Chưa có dữ liệu ticker.")
    else:
        default_mkt = (
            selected_ticker
            if selected_ticker != "ALL" and selected_ticker in market_ticker_options
            else market_ticker_options[0]
        )
        mkt_ticker = st.selectbox(
            "Chọn mã",
            market_ticker_options,
            index=market_ticker_options.index(default_mkt),
            key="market_ticker_select",
        )

        price_df = load_market_prices(mkt_ticker, start_date=start_date, end_date=end_date)

        if price_df.empty:
            st.warning(f"Không có dữ liệu giá cho {mkt_ticker} trong khoảng ngày đã chọn.")
        else:
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_df = price_df.sort_values("date")

            latest_price = price_df.iloc[-1]
            prev_price = price_df.iloc[-2] if len(price_df) > 1 else None

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Ticker", mkt_ticker)
            m2.metric("Giá đóng cửa", f"{latest_price['close']:,.1f}")
            if prev_price is not None:
                chg = latest_price["close"] - prev_price["close"]
                pct = chg / prev_price["close"] * 100
                m3.metric("Thay đổi 1 phiên", f"{chg:+,.1f}", f"{pct:+.2f}%")
            else:
                m3.metric("Thay đổi 1 phiên", "-")
            if len(price_df) >= 5:
                ret_5 = (latest_price["close"] / price_df.iloc[-5]["close"] - 1) * 100
                m4.metric("5-phiên", f"{ret_5:+.2f}%")
            else:
                m4.metric("5-phiên", "-")
            if len(price_df) >= 20:
                ret_20 = (latest_price["close"] / price_df.iloc[-20]["close"] - 1) * 100
                m5.metric("20-phiên", f"{ret_20:+.2f}%")
            else:
                m5.metric("20-phiên", "-")
            m6.metric("Khối lượng", f"{int(latest_price['volume']):,}")

            st.divider()

            # Candlestick + volume
            fig_candle = go.Figure()

            fig_candle.add_trace(go.Candlestick(
                x=price_df["date"],
                open=price_df["open"],
                high=price_df["high"],
                low=price_df["low"],
                close=price_df["close"],
                name="OHLC",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ))

            fig_candle.update_layout(
                title=f"Giá {mkt_ticker}",
                xaxis_title="Ngày",
                yaxis_title="Giá (VNĐ nghìn)",
                xaxis_rangeslider_visible=False,
                height=420,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_candle, width="stretch")

            # Volume bar chart
            colors = [
                "#26a69a" if price_df["close"].iloc[i] >= price_df["open"].iloc[i]
                else "#ef5350"
                for i in range(len(price_df))
            ]
            fig_vol = go.Figure(go.Bar(
                x=price_df["date"],
                y=price_df["volume"],
                marker_color=colors,
                name="Khối lượng",
            ))
            fig_vol.update_layout(
                title="Khối lượng giao dịch",
                xaxis_title="Ngày",
                yaxis_title="Khối lượng",
                height=220,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_vol, width="stretch")

            # Sentiment overlay if available
            if pipeline_ready:
                sent_df = filtered_df[filtered_df["ticker"] == mkt_ticker].sort_values("date").copy()
                if not sent_df.empty:
                    st.markdown("### Giá đóng cửa + Sentiment Index")

                    merged = pd.merge(
                        price_df[["date", "close"]],
                        sent_df[["date", "sentiment_index", "final_sentiment", "article_count"]],
                        on="date",
                        how="inner",
                    )

                    if not merged.empty:
                        color_map = {"positive": "#26a69a", "neutral": "#bdbdbd", "negative": "#ef5350"}

                        fig_overlay = go.Figure()

                        fig_overlay.add_trace(go.Scatter(
                            x=merged["date"],
                            y=merged["close"],
                            mode="lines",
                            name="Giá đóng cửa",
                            line=dict(color="#1976d2", width=2),
                            yaxis="y1",
                        ))

                        for label, color in color_map.items():
                            sub = merged[merged["final_sentiment"] == label]
                            if sub.empty:
                                continue
                            fig_overlay.add_trace(go.Scatter(
                                x=sub["date"],
                                y=sub["sentiment_index"],
                                mode="markers",
                                name=f"Sentiment {label}",
                                marker=dict(color=color, size=6, opacity=0.8),
                                yaxis="y2",
                                customdata=sub[["article_count"]].values,
                                hovertemplate=(
                                    "%{x}<br>sentiment=%{y:.3f}<br>articles=%{customdata[0]}<extra></extra>"
                                ),
                            ))

                        fig_overlay.update_layout(
                            title=f"Giá đóng cửa và Sentiment Index - {mkt_ticker}",
                            xaxis=dict(title="Ngày"),
                            yaxis=dict(title="Giá (VNĐ nghìn)", side="left"),
                            yaxis2=dict(
                                title="Sentiment Index",
                                overlaying="y",
                                side="right",
                                zeroline=True,
                                zerolinecolor="#999",
                            ),
                            height=480,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(t=60, b=10),
                        )
                        fig_overlay.add_hline(y=0, line_dash="dash", line_color="#999", yref="y2")
                        st.plotly_chart(fig_overlay, width="stretch")

            st.markdown("### Dữ liệu OHLCV")
            st.dataframe(
                price_df.sort_values("date", ascending=False).assign(
                    date=lambda d: d["date"].dt.date
                ),
                width="stretch",
                hide_index=True,
            )


# =========================================================
# TAB 4: NEWS
# =========================================================

with tab_news:
    st.subheader("Tin tức")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        news_ticker = st.selectbox(
            "Ticker",
            available_tickers,
            index=0,
            key="news_ticker",
        )

    with col_f2:
        news_sentiment = st.selectbox(
            "Sentiment",
            ["ALL", "positive", "neutral", "negative"],
            key="news_sentiment",
        )

    with col_f3:
        news_page_size = st.selectbox("Bài / trang", [10, 20, 50], index=1, key="news_page_size")

    if "news_page" not in st.session_state:
        st.session_state.news_page = 0

    # Reset page when filters change
    filter_key = f"{news_ticker}|{news_sentiment}|{news_page_size}|{start_date}|{end_date}"
    if st.session_state.get("news_filter_key") != filter_key:
        st.session_state.news_page = 0
        st.session_state.news_filter_key = filter_key

    news_df, news_total = load_news(
        ticker=news_ticker,
        sentiment=news_sentiment,
        start_date=start_date,
        end_date=end_date,
        page=st.session_state.news_page,
        page_size=news_page_size,
    )

    total_pages = max(1, (news_total + news_page_size - 1) // news_page_size)

    st.caption(
        f"Tổng: {news_total} bài · Trang {st.session_state.news_page + 1}/{total_pages}"
    )

    nav1, nav2, nav3 = st.columns([1, 8, 1])
    with nav1:
        if st.button("◀ Trước", disabled=st.session_state.news_page == 0, key="news_prev"):
            st.session_state.news_page -= 1
            st.rerun()
    with nav3:
        if st.button("Sau ▶", disabled=st.session_state.news_page >= total_pages - 1, key="news_next"):
            st.session_state.news_page += 1
            st.rerun()

    if news_df.empty:
        st.info("Không có bài báo phù hợp với bộ lọc.")
    else:
        for _, row in news_df.iterrows():
            label = str(row.get("sentiment_label", "neutral"))
            badge = sentiment_badge(label)
            ticker_val = row.get("ticker", "")
            conf = row.get("confidence", None)
            score = row.get("sentiment_score", None)
            pub = row.get("published_at_vn", "")
            title = row.get("title", "(no title)")
            url = row.get("url", "")
            category = row.get("category", "")

            pub_str = pd.to_datetime(pub).strftime("%Y-%m-%d %H:%M") if pd.notna(pub) else "-"
            display_title = title if len(title) <= 80 else title[:77] + "..."
            with st.expander(
                f"{pub_str}  |  {ticker_val}  |  {badge}  |  conf={format_score(conf)}  |  {display_title}"
            ):
                if url:
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")

                st.caption(f"Danh mục: {category} · {pub_str}")

                pcols = st.columns(3)
                pcols[0].metric("P(positive)", format_score(row.get("prob_positive")))
                pcols[1].metric("P(neutral)", format_score(row.get("prob_neutral")))
                pcols[2].metric("P(negative)", format_score(row.get("prob_negative")))


# =========================================================
# TAB 4: EXPLAIN EVIDENCE
# =========================================================

with tab_evidence:
    st.subheader("Explain Evidence: Vì sao model ra sentiment?")

    evidence_ticker_options = available_tickers

    evidence_ticker = st.selectbox(
        "Chọn mã evidence",
        evidence_ticker_options,
        index=evidence_ticker_options.index(selected_ticker)
        if selected_ticker in evidence_ticker_options
        else 0,
        key="evidence_ticker_select",
    )

    evidence_limit = st.slider(
        "Số evidence hiển thị",
        min_value=20,
        max_value=500,
        value=100,
        step=20,
    )

    evidence_df = load_evidence(
        ticker=None if evidence_ticker == "ALL" else evidence_ticker,
        limit=evidence_limit,
    )

    if evidence_df.empty:
        st.warning(
            "Không có evidence. Kiểm tra view sentiment_evidence_view hoặc bảng entity_sentiments."
        )
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Evidence rows", safe_int(len(evidence_df)))
        e2.metric("Tickers", safe_int(evidence_df["ticker"].nunique()))
        e3.metric("Aspects", safe_int(evidence_df["aspect"].nunique()))
        e4.metric("Avg confidence", format_score(evidence_df["confidence"].mean()))

        st.markdown("### Phân bổ aspect")
        aspect_count = (
            evidence_df.groupby(["aspect", "sentiment_label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        fig = px.bar(
            aspect_count,
            x="aspect",
            y="count",
            color="sentiment_label",
            barmode="group",
            title="Số evidence theo aspect và sentiment label",
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("### Evidence table")

        display_cols = [
            "published_at_vn",
            "ticker",
            "company_name",
            "sector",
            "aspect",
            "sentiment_label",
            "sentiment_score",
            "confidence",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
            "title",
            "context",
            "url",
        ]

        existing_cols = [col for col in display_cols if col in evidence_df.columns]

        st.dataframe(
            evidence_df[existing_cols],
            width="stretch",
            hide_index=True,
        )

        st.markdown("### Xem từng evidence dễ đọc")

        for _, row in evidence_df.head(20).iterrows():
            label = row.get("sentiment_label", "")
            score = row.get("sentiment_score", None)
            conf = row.get("confidence", None)
            aspect = row.get("aspect", "")
            ticker_val = row.get("ticker", "")
            title = row.get("title", "")
            url = row.get("url", "")

            with st.expander(
                f"{ticker_val} | {aspect} | {sentiment_badge(str(label))} | score={format_score(score)} | conf={format_score(conf)}"
            ):
                st.markdown(f"**Title:** {title}")

                if url:
                    st.markdown(f"**URL:** {url}")

                st.markdown("**Context model đã đọc:**")
                st.write(row.get("context", ""))

                pcols = st.columns(3)
                pcols[0].metric("P(negative)", format_score(row.get("prob_negative")))
                pcols[1].metric("P(neutral)", format_score(row.get("prob_neutral")))
                pcols[2].metric("P(positive)", format_score(row.get("prob_positive")))


# =========================================================
# TAB 4: MARKET VALIDATION
# =========================================================

with tab_validation:
    st.subheader("Kiểm định sentiment với biến thị trường")

    if filtered_df.empty:
        st.info("Chưa có dữ liệu. Hãy chạy build_sentiment_aggregates và build_market_features trước.")

    market_df = load_market_validation()
    aspect_market_df = load_aspect_market_validation()

    if market_df.empty:
        st.warning(
            "Chưa có dữ liệu sentiment_market_forward_dataset. "
            "Hãy chạy market validation pipeline hoặc kiểm tra overlap giữa ngày sentiment và market_prices."
        )
        st.code(
            """
python -m src.jobs.run_market_validation_pipeline --start-date 2025-01-01 --limit-tickers 60
            """.strip(),
            language="bash",
        )
    else:
        market_df["date"] = pd.to_datetime(market_df["date"])

        market_filtered = apply_common_filters(
            market_df,
            start_date=start_date,
            end_date=end_date,
            selected_ticker=selected_ticker,
            selected_sector=selected_sector,
        )

        if market_filtered.empty:
            st.warning(
                "Không có dữ liệu validation phù hợp với bộ lọc hiện tại. "
                "Hãy chọn lại ticker, sector hoặc khoảng ngày."
            )
        else:
            st.markdown("### Chọn biến thị trường để kiểm định")

            target_col = st.selectbox(
                "Chọn target kiểm định",
                ["forward_return_1d", "forward_return_3d", "forward_return_5d"],
                index=1,
                help=(
                    "forward_return_1d/3d/5d là lợi suất sau 1/3/5 phiên giao dịch, "
                    "không phải 1/3/5 ngày lịch."
                ),
            )

            validation_clean = market_filtered[
                [
                    "date",
                    "ticker",
                    "sector",
                    "sentiment_index",
                    "final_sentiment",
                    "article_count",
                    "avg_confidence",
                    target_col,
                ]
            ].dropna(subset=["sentiment_index", target_col]).copy()

            st.markdown("### Tổng quan tập kiểm định sau bộ lọc")

            v1, v2, v3, v4, v5 = st.columns(5)
            v1.metric("Validation rows", safe_int(len(validation_clean)))
            v2.metric("Tickers", safe_int(validation_clean["ticker"].nunique()))
            v3.metric("Sectors", safe_int(validation_clean["sector"].nunique()))
            v4.metric("Avg confidence", format_score(validation_clean["avg_confidence"].mean()))
            v5.metric("Avg article/day", format_score(validation_clean["article_count"].mean()))

            st.caption(
                f"Bộ lọc hiện tại: ticker={selected_ticker}, sector={selected_sector}, "
                f"date={start_date} → {end_date}, target={target_col}"
            )

            if len(validation_clean) < 30:
                st.warning(
                    f"Sample size sau bộ lọc chỉ có {len(validation_clean)} dòng. "
                    "Correlation/hit-rate chưa đủ mạnh để kết luận thống kê."
                )

            corr = (
                validation_clean["sentiment_index"].corr(validation_clean[target_col])
                if len(validation_clean) >= 3
                else None
            )

            hit_rate, directional_sample = calculate_directional_hit_rate(
                validation_clean,
                target_col,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Correlation với {target_col}", format_score(corr))
            c2.metric("Directional hit-rate", format_pct(hit_rate))
            c3.metric("Directional sample", safe_int(directional_sample))

            st.caption(
                "Correlation > 0 nghĩa là sentiment_index càng cao thì forward return có xu hướng càng cao. "
                "Hit-rate đo tỷ lệ positive đi cùng return dương và negative đi cùng return âm."
            )

            st.markdown("### Sentiment Index vs Forward Return")

            if validation_clean.empty:
                st.info("Không có dữ liệu sạch sau khi bỏ NULL/NaN.")
            else:
                fig = px.scatter(
                    validation_clean,
                    x="sentiment_index",
                    y=target_col,
                    color="final_sentiment",
                    size="article_count",
                    hover_data={
                        "date": True,
                        "ticker": True,
                        "sector": True,
                        "sentiment_index": ":.3f",
                        target_col: ":.3%",
                        "article_count": True,
                        "avg_confidence": ":.3f",
                        "final_sentiment": True,
                    },
                    title=f"Sentiment Index vs {target_col}",
                )

                fig.add_hline(y=0, line_dash="dash")
                fig.add_vline(x=0, line_dash="dash")

                fig.update_layout(
                    xaxis_title="Sentiment index",
                    yaxis_title=target_col,
                    legend_title="Final sentiment",
                    height=560,
                )

                st.plotly_chart(fig, width="stretch")

                with st.expander("Cách đọc biểu đồ"):
                    st.markdown(
                        """
- **Góc phải trên**: sentiment tích cực, return tương lai dương → tín hiệu đúng chiều.
- **Góc trái dưới**: sentiment tiêu cực, return tương lai âm → tín hiệu đúng chiều.
- **Góc phải dưới**: sentiment tích cực nhưng return âm → tín hiệu sai chiều.
- **Góc trái trên**: sentiment tiêu cực nhưng return dương → tín hiệu sai chiều.
- Điểm càng to nghĩa là ngày đó có nhiều bài báo hơn.
                        """
                    )

            st.markdown("### Return trung bình theo nhóm sentiment")

            if validation_clean.empty:
                st.info("Không có dữ liệu để tính return trung bình.")
            else:
                group_df = (
                    validation_clean.groupby("final_sentiment", as_index=False)
                    .agg(
                        sample_size=("ticker", "count"),
                        avg_sentiment_index=("sentiment_index", "mean"),
                        avg_forward_return=(target_col, "mean"),
                        median_forward_return=(target_col, "median"),
                        avg_confidence=("avg_confidence", "mean"),
                        avg_article_count=("article_count", "mean"),
                    )
                    .sort_values("avg_sentiment_index", ascending=False)
                )

                g1, g2 = st.columns([1, 1])

                with g1:
                    st.dataframe(group_df, width="stretch", hide_index=True)

                with g2:
                    fig_group = px.bar(
                        group_df,
                        x="final_sentiment",
                        y="avg_forward_return",
                        hover_data=[
                            "sample_size",
                            "avg_sentiment_index",
                            "median_forward_return",
                            "avg_confidence",
                            "avg_article_count",
                        ],
                        title=f"Avg {target_col} theo nhóm sentiment",
                    )
                    fig_group.add_hline(y=0, line_dash="dash")
                    fig_group.update_layout(
                        xaxis_title="Final sentiment",
                        yaxis_title=f"Avg {target_col}",
                        height=420,
                    )
                    st.plotly_chart(fig_group, width="stretch")

            st.markdown("### Diễn biến sentiment và forward return theo thời gian")

            if validation_clean.empty:
                st.info("Không có dữ liệu để vẽ time series.")
            else:
                time_df = (
                    validation_clean.groupby("date", as_index=False)
                    .agg(
                        sentiment_index=("sentiment_index", "mean"),
                        forward_return=(target_col, "mean"),
                        article_count=("article_count", "sum"),
                    )
                    .sort_values("date")
                )

                fig_time = go.Figure()

                fig_time.add_trace(
                    go.Scatter(
                        x=time_df["date"],
                        y=time_df["sentiment_index"],
                        mode="lines+markers",
                        name="Avg sentiment_index",
                        yaxis="y1",
                    )
                )

                fig_time.add_trace(
                    go.Scatter(
                        x=time_df["date"],
                        y=time_df["forward_return"],
                        mode="lines+markers",
                        name=f"Avg {target_col}",
                        yaxis="y2",
                    )
                )

                fig_time.update_layout(
                    title=f"Sentiment và {target_col} theo thời gian",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Sentiment index"),
                    yaxis2=dict(
                        title=target_col,
                        overlaying="y",
                        side="right",
                    ),
                    height=480,
                    legend=dict(orientation="h"),
                )

                st.plotly_chart(fig_time, width="stretch")

            st.markdown("### Một số tín hiệu đúng/sai chiều nổi bật")

            if validation_clean.empty:
                st.info("Không có dữ liệu để hiển thị tín hiệu.")
            else:
                signal_df = validation_clean.copy()
                signal_df["direction_result"] = "neutral_or_ignored"

                signal_df.loc[
                    (signal_df["final_sentiment"] == "positive") & (signal_df[target_col] > 0),
                    "direction_result",
                ] = "hit_positive"

                signal_df.loc[
                    (signal_df["final_sentiment"] == "negative") & (signal_df[target_col] < 0),
                    "direction_result",
                ] = "hit_negative"

                signal_df.loc[
                    (signal_df["final_sentiment"] == "positive") & (signal_df[target_col] < 0),
                    "direction_result",
                ] = "miss_positive"

                signal_df.loc[
                    (signal_df["final_sentiment"] == "negative") & (signal_df[target_col] > 0),
                    "direction_result",
                ] = "miss_negative"

                show_cols = [
                    "date",
                    "ticker",
                    "sector",
                    "sentiment_index",
                    "final_sentiment",
                    target_col,
                    "direction_result",
                    "article_count",
                    "avg_confidence",
                ]

                st.dataframe(
                    signal_df[show_cols]
                    .sort_values(target_col, ascending=False)
                    .head(100),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("### Aspect-level validation")

            if aspect_market_df.empty:
                st.info("Chưa có dữ liệu aspect-level validation.")
            else:
                aspect_market_df["date"] = pd.to_datetime(aspect_market_df["date"])

                aspect_filtered = apply_common_filters(
                    aspect_market_df,
                    start_date=start_date,
                    end_date=end_date,
                    selected_ticker=selected_ticker,
                    selected_sector=selected_sector,
                )

                aspect_target = st.selectbox(
                    "Target aspect validation",
                    ["forward_return_1d", "forward_return_3d", "forward_return_5d"],
                    index=["forward_return_1d", "forward_return_3d", "forward_return_5d"].index(target_col),
                    key="aspect_target_select",
                )

                aspect_clean = aspect_filtered.dropna(
                    subset=["aspect_sentiment_score", aspect_target]
                ).copy()

                if aspect_clean.empty:
                    st.info("Không có dữ liệu aspect phù hợp với bộ lọc hiện tại.")
                else:
                    fig_aspect = px.scatter(
                        aspect_clean,
                        x="aspect_sentiment_score",
                        y=aspect_target,
                        color="aspect",
                        size="sample_count",
                        hover_data={
                            "date": True,
                            "ticker": True,
                            "sector": True,
                            "sample_count": True,
                            "avg_confidence": ":.3f",
                            "aspect_sentiment_score": ":.3f",
                            aspect_target: ":.3%",
                        },
                        title=f"Aspect sentiment vs {aspect_target}",
                    )

                    fig_aspect.add_hline(y=0, line_dash="dash")
                    fig_aspect.add_vline(x=0, line_dash="dash")
                    fig_aspect.update_layout(
                        xaxis_title="Aspect sentiment score",
                        yaxis_title=aspect_target,
                        height=560,
                    )

                    st.plotly_chart(fig_aspect, width="stretch")

                    aspect_corr_rows = []

                    for aspect, group in aspect_clean.groupby("aspect"):
                        clean_group = group.dropna(
                            subset=["aspect_sentiment_score", aspect_target]
                        )

                        corr_val = (
                            clean_group["aspect_sentiment_score"].corr(clean_group[aspect_target])
                            if len(clean_group) >= 3
                            else None
                        )

                        aspect_corr_rows.append(
                            {
                                "aspect": aspect,
                                "sample_size": len(clean_group),
                                "correlation": corr_val,
                                "avg_aspect_sentiment": clean_group["aspect_sentiment_score"].mean(),
                                "avg_forward_return": clean_group[aspect_target].mean(),
                                "avg_confidence": clean_group["avg_confidence"].mean(),
                            }
                        )

                    aspect_corr = pd.DataFrame(aspect_corr_rows).sort_values(
                        "correlation",
                        ascending=False,
                        na_position="last",
                    )

                    st.markdown("### Tương quan theo aspect sau bộ lọc")
                    st.dataframe(aspect_corr, width="stretch", hide_index=True)


# =========================================================
# TAB 5: DATA QUALITY
# =========================================================

with tab_data_quality:
    st.subheader("Data Quality & Coverage")

    q1, q2 = st.columns(2)

    with q1:
        st.markdown("### Market price coverage")

        price_summary = load_market_prices_summary()

        if price_summary.empty:
            st.warning("Chưa có dữ liệu market_prices.")
        else:
            st.dataframe(price_summary, width="stretch", hide_index=True)

    with q2:
        st.markdown("### Sentiment coverage")

        sentiment_summary = (
            daily_df.groupby("ticker", as_index=False)
            .agg(
                min_date=("date", "min"),
                max_date=("date", "max"),
                sentiment_rows=("date", "count"),
                avg_sentiment=("sentiment_index", "mean"),
                avg_confidence=("avg_confidence", "mean"),
                article_count=("article_count", "sum"),
            )
            .sort_values("sentiment_rows", ascending=False)
        )

        st.dataframe(sentiment_summary, width="stretch", hide_index=True)

    st.markdown("### Overlap sentiment + market")

    market_validation_df = load_market_validation()

    if market_validation_df.empty:
        st.warning(
            "Overlap giữa daily_sentiment_index và market_features đang bằng 0 hoặc rất thấp."
        )
    else:
        overlap_summary = (
            market_validation_df.groupby("ticker", as_index=False)
            .agg(
                min_date=("date", "min"),
                max_date=("date", "max"),
                rows=("date", "count"),
            )
            .sort_values("rows", ascending=False)
        )
        st.dataframe(overlap_summary, width="stretch", hide_index=True)

    st.markdown("### Bài bị nghi parse sai ngày tương lai")

    future_articles_df = load_future_date_articles()

    if future_articles_df.empty:
        st.success("Không phát hiện bài báo có published_at nằm quá xa trong tương lai.")
    else:
        st.warning(
            "Có bài báo có published_at nằm trong tương lai. "
            "Cần sửa parser ngày CafeF và crawl lại."
        )
        st.dataframe(future_articles_df, width="stretch", hide_index=True)

    st.markdown("### SQL kiểm tra nhanh")

    st.code(
        """
SELECT COUNT(*) FROM daily_sentiment_index;
SELECT COUNT(*) FROM sentiment_evidence_view;
SELECT COUNT(*) FROM market_prices;
SELECT COUNT(*) FROM market_features;
SELECT COUNT(*) FROM sentiment_market_forward_dataset;

SELECT ticker, MIN(date), MAX(date), COUNT(*)
FROM market_prices
GROUP BY ticker
ORDER BY ticker;

SELECT
    date,
    ticker,
    sentiment_index,
    article_count,
    EXTRACT(DOW FROM date) AS dow
FROM daily_sentiment_index
WHERE EXTRACT(DOW FROM date) IN (0, 6)
ORDER BY date DESC;

SELECT
    title,
    published_at,
    crawl_at,
    url
FROM articles
WHERE published_at > CURRENT_DATE + INTERVAL '7 days'
ORDER BY published_at DESC;
        """.strip(),
        language="sql",
    )

# =========================================================
# TAB 8: RESEARCH
# =========================================================
# NOTE: TAB 9 (Tín hiệu hôm nay) cached functions are defined after tab_research block

@st.cache_data(ttl=1800, show_spinner="Đang tải panel data từ DB...")
def load_research_panel(sd: str, ed: str) -> pd.DataFrame:
    from src.jobs.export_daily_panel import export_panel
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as _f:
        tmp = _f.name
    try:
        df = export_panel(output_path=tmp, start_date=sd, end_date=ed)
    finally:
        try:
            _os.unlink(tmp)
        except Exception:
            pass
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "volatility" not in df.columns and "volatility_5d" in df.columns:
            df["volatility"] = df["volatility_5d"]
    return df


@st.cache_data(ttl=1800, show_spinner="Đang tính cross-correlations...")
def cached_cross_correlations(sd: str, ed: str, max_lag: int) -> pd.DataFrame:
    from src.analysis.cross_correlations import compute_cross_correlations
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    return compute_cross_correlations(panel, max_lag)


@st.cache_data(ttl=1800, show_spinner="Đang chạy Granger OLS test...")
def cached_granger_ols(sd: str, ed: str) -> pd.DataFrame:
    from src.analysis.granger_sentiment import granger_for_group
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for window in [10, 21, 63]:
            rows.append(granger_for_group(grp, str(ticker), "log_return", window, 80))
    return pd.DataFrame(rows).sort_values(["p_value", "ticker", "window"], na_position="last")


@st.cache_data(ttl=1800, show_spinner="Đang chạy VAR Granger test...")
def cached_var_granger(sd: str, ed: str) -> pd.DataFrame:
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        return pd.DataFrame({"status": ["statsmodels not installed"]})
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in panel.groupby("ticker"):
        data = grp.sort_values("date")[["log_return", "volume_growth", "sentiment_score"]].replace([float("inf"), float("-inf")], float("nan")).dropna()
        for lag in [10, 21, 63]:
            if len(data) < max(120, lag * 3 + 10):
                rows.append({"ticker": ticker, "lag": lag, "n": len(data), "test_stat": None, "p_value": None, "status": "too_few_obs"})
                continue
            try:
                res = VAR(data).fit(maxlags=lag, ic=None, trend="c")
                test = res.test_causality("log_return", ["sentiment_score"], kind="f")
                rows.append({"ticker": ticker, "lag": lag, "n": len(data), "test_stat": float(test.test_statistic), "p_value": float(test.pvalue), "status": "ok"})
            except Exception as exc:
                rows.append({"ticker": ticker, "lag": lag, "n": len(data), "test_stat": None, "p_value": None, "status": str(exc)[:80]})
    return pd.DataFrame(rows).sort_values(["p_value", "ticker", "lag"], na_position="last")


@st.cache_data(ttl=1800, show_spinner="Đang chạy Event Study...")
def cached_event_study(sd: str, ed: str) -> pd.DataFrame:
    from src.analysis.event_study_sentiment import add_market_return, add_future_returns, summarize, sentiment_group
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    panel = add_market_return(panel)
    panel = pd.concat(
        [add_future_returns(grp, [1, 3, 5, 10]) for _, grp in panel.groupby("ticker")],
        ignore_index=True,
    )
    panel["sentiment_group"] = panel["sentiment_score"].map(lambda s: sentiment_group(s, 0.0))
    events = panel[panel["news_count"].fillna(0) > 0].copy()
    summaries = []
    for h in [1, 3, 5, 10]:
        for metric in [f"future_return_{h}", f"future_abret_{h}"]:
            summaries.append(summarize(events, metric, ["sentiment_group"]))
    return pd.concat(summaries, ignore_index=True).sort_values(["metric", "sentiment_group"])


@st.cache_data(ttl=1800, show_spinner="Đang tính sentiment-target confusion...")
def cached_sentiment_confusion(sd: str, ed: str, threshold: float = 0.0) -> pd.DataFrame:
    from src.analysis.sentiment_target_confusion import evaluate_binary, make_sentiment_up
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    import numpy as _np
    if "target_up" not in panel.columns:
        panel["target_up"] = _np.where(panel["log_return"].notna(), (panel["log_return"] > 0).astype(float), _np.nan)
    panel = panel.sort_values(["ticker", "date"]).copy()
    panel["sentiment_up"] = make_sentiment_up(panel["sentiment_score"], threshold, False)
    rows = []
    for lag in range(-5, 6):
        parts = []
        for _, grp in panel.groupby("ticker"):
            grp = grp.sort_values("date").reset_index(drop=True)
            parts.append(pd.DataFrame({
                "target_up": grp["target_up"],
                "sentiment_up_lagged": grp["sentiment_up"].shift(lag),
            }))
        combined = pd.concat(parts, ignore_index=True)
        metrics = evaluate_binary(combined["target_up"], combined["sentiment_up_lagged"])
        rows.append({"lag": lag, **metrics})
    return pd.DataFrame(rows).sort_values("lag")


@st.cache_data(ttl=3600, show_spinner="Đang chạy ML comparison (có thể mất vài phút)...")
def cached_ml_comparison(sd: str, ed: str) -> pd.DataFrame:
    from src.analysis.compare_ml_models import (
        build_dataset, time_split, run_scope, build_sentiment_delta, SENTIMENT_FEATURE_PREFIXES,
    )
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    all_results: list = []
    all_preds: list = []
    for window in [10, 21]:
        data = build_dataset(panel, window, horizon=1)
        feat_all = [c for c in data.columns if c not in ["ticker", "date", "target_return", "target_up"]]
        feat_sets = {
            "with_sentiment": feat_all,
            "no_sentiment": [c for c in feat_all if not c.startswith(SENTIMENT_FEATURE_PREFIXES)],
        }
        train_all, test_all = time_split(data, 0.7, 1)
        if len(train_all) >= 100 and len(test_all) >= 40:
            for fs_name, fs_cols in feat_sets.items():
                run_scope("ALL", fs_name, window, fs_cols, train_all, test_all, 42, all_results, all_preds)
    if not all_results:
        return pd.DataFrame()
    results = pd.DataFrame(all_results)
    delta = build_sentiment_delta(results)
    return delta


@st.cache_data(ttl=3600, show_spinner="Đang chạy Granger-style model comparison...")
def cached_granger_style(sd: str, ed: str) -> pd.DataFrame:
    from src.analysis.compare_granger_style_models import build_lagged_dataset, run_one_scope, build_delta
    panel = load_research_panel(sd, ed)
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for window in [10, 21]:
        data = build_lagged_dataset(panel, window, 1).dropna(subset=["target_return"])
        rows.extend(run_one_scope("ALL", data, window, 0.7, 100, 40, 42, 1, fast=True))
    if not rows:
        return pd.DataFrame()
    return build_delta(pd.DataFrame(rows))


with tab_research:
    st.header("🔬 Research Analysis")
    st.caption("Phân tích thống kê và ML về mối quan hệ sentiment – giá cổ phiếu.")

    res_sd = str(start_date)
    res_ed = str(end_date)

    (
        res_tab_ccf, res_tab_granger, res_tab_var,
        res_tab_event, res_tab_conf, res_tab_ml, res_tab_granger_ml,
    ) = st.tabs([
        "📉 Cross-Correlation",
        "📐 Granger OLS",
        "🔁 VAR Granger",
        "📅 Event Study",
        "🎯 Confusion Matrix",
        "🤖 ML Models",
        "⚖️ Granger-style ML",
    ])

    # ── 1. Cross-correlations ──────────────────────────────────────────────
    with res_tab_ccf:
        st.subheader("Lagged cross-correlations: sentiment vs trading metrics")
        max_lag = st.slider("Max lag (trading days)", 1, 10, 5, key="ccf_lag")
        if st.button("Chạy", key="run_ccf"):
            cached_cross_correlations.clear()
        ccf_df = cached_cross_correlations(res_sd, res_ed, max_lag)
        if ccf_df.empty:
            st.warning("Không đủ dữ liệu panel.")
        else:
            pair_options = ["ALL_MEAN"] + [p for p in ccf_df["pair"].unique() if p != "ALL_MEAN"]
            chosen_pair = st.selectbox("Pair", pair_options, key="ccf_pair")
            sub = ccf_df[ccf_df["pair"] == chosen_pair]
            pivot = sub.pivot_table(index="ticker", columns="lag", values="correlation")
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="r"),
            ))
            fig.update_layout(title=f"Correlation heatmap — {chosen_pair}", xaxis_title="Lag", yaxis_title="Ticker", height=600)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(sub.sort_values(["ticker", "lag"]), width="stretch", hide_index=True)

    # ── 2. Granger OLS ─────────────────────────────────────────────────────
    with res_tab_granger:
        st.subheader("Granger causality (OLS F-test): sentiment → return")
        st.caption("H₀: past sentiment không thêm thông tin dự báo khi đã có past return + volume.")
        if st.button("Chạy", key="run_granger"):
            cached_granger_ols.clear()
        granger_df = cached_granger_ols(res_sd, res_ed)
        if granger_df.empty:
            st.warning("Không đủ dữ liệu.")
        else:
            sig = granger_df[granger_df["status"] == "ok"].copy()
            sig["significant"] = sig["p_value"] < 0.05
            fig = px.scatter(
                sig, x="ticker", y="p_value", color="window",
                symbol="significant",
                hover_data=["f_stat", "n", "window"],
                title="Granger OLS p-value by ticker & window (dưới 0.05 = significant)",
            )
            fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="p=0.05")
            fig.update_layout(yaxis_type="log", height=450)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(sig[["ticker", "window", "n", "f_stat", "p_value", "status", "significant"]].sort_values("p_value"), width="stretch", hide_index=True)

    # ── 3. VAR Granger ─────────────────────────────────────────────────────
    with res_tab_var:
        st.subheader("VAR Granger causality: sentiment → log_return")
        st.caption("Fit VAR(lag) trên [log_return, volume_growth, sentiment_score] rồi test Granger causality.")
        if st.button("Chạy", key="run_var"):
            cached_var_granger.clear()
        var_df = cached_var_granger(res_sd, res_ed)
        if var_df.empty:
            st.warning("Không đủ dữ liệu.")
        else:
            ok = var_df[var_df["status"] == "ok"].copy()
            if ok.empty:
                st.warning("Không có ticker nào đủ điều kiện.")
            else:
                ok["significant"] = ok["p_value"] < 0.05
                fig = px.scatter(
                    ok, x="ticker", y="p_value", color="lag",
                    symbol="significant",
                    hover_data=["test_stat", "n"],
                    title="VAR Granger p-value (log scale) — dưới 0.05 = significant",
                )
                fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="p=0.05")
                fig.update_layout(yaxis_type="log", height=450)
                st.plotly_chart(fig, width="stretch")
            st.dataframe(var_df[["ticker", "lag", "n", "test_stat", "p_value", "status"]].sort_values("p_value", na_position="last"), width="stretch", hide_index=True)

    # ── 4. Event Study ─────────────────────────────────────────────────────
    with res_tab_event:
        st.subheader("Event Study: cumulative return after sentiment event")
        st.caption("POS/NEG/NEU được gán theo sentiment_score ngày có tin. Abnormal return = return – market average cùng ngày.")
        if st.button("Chạy", key="run_event"):
            cached_event_study.clear()
        event_df = cached_event_study(res_sd, res_ed)
        if event_df.empty:
            st.warning("Không đủ dữ liệu.")
        else:
            metric_type = st.radio("Metric", ["future_abret", "future_return"], horizontal=True, key="evt_metric")
            horizons_available = [1, 3, 5, 10]
            rows_plot = event_df[event_df["metric"].str.startswith(metric_type)].copy()
            rows_plot["horizon"] = rows_plot["metric"].str.extract(r"(\d+)$").astype(int)
            rows_plot = rows_plot[rows_plot["ticker_col_missing"] if "ticker_col_missing" in rows_plot.columns else rows_plot["n_tickers"] >= 1]
            fig = px.bar(
                rows_plot, x="horizon", y="mean", color="sentiment_group",
                barmode="group", error_y="std",
                hover_data=["n", "n_tickers", "t_stat_vs_0", "p_value_vs_0"],
                title=f"{metric_type} by horizon & sentiment group",
                labels={"mean": "Mean return", "horizon": "Horizon (days)"},
                color_discrete_map={"POS": "#2ecc71", "NEG": "#e74c3c", "NEU": "#95a5a6"},
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(rows_plot[["sentiment_group", "horizon", "metric", "n", "n_tickers", "mean", "median", "t_stat_vs_0", "p_value_vs_0"]].sort_values(["horizon", "sentiment_group"]), width="stretch", hide_index=True)

    # ── 5. Sentiment-Target Confusion ──────────────────────────────────────
    with res_tab_conf:
        st.subheader("Sentiment direction vs stock direction (ALL tickers)")
        st.caption("sentiment_up = sentiment_score > threshold. target_up = log_return > 0. Lag k = sentiment từ k ngày trước.")
        conf_threshold = st.slider("Sentiment threshold", min_value=-0.5, max_value=0.5, value=0.0, step=0.05, key="conf_threshold")
        if st.button("Chạy", key="run_conf"):
            cached_sentiment_confusion.clear()
        conf_df = cached_sentiment_confusion(res_sd, res_ed, conf_threshold)
        if conf_df.empty:
            st.warning("Không đủ dữ liệu.")
        else:
            fig = go.Figure()
            for metric, color in [("accuracy", "#3498db"), ("f1", "#e67e22"), ("precision", "#9b59b6"), ("recall", "#1abc9c")]:
                if metric in conf_df.columns:
                    fig.add_trace(go.Scatter(x=conf_df["lag"], y=conf_df[metric], mode="lines+markers", name=metric, line=dict(color=color)))
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(title="Accuracy / F1 / Precision / Recall by lag", xaxis_title="Lag (ngày)", yaxis_title="Score", height=400)
            st.plotly_chart(fig, width="stretch")
            display_cols = [c for c in ["lag", "n", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "target_up_rate", "sentiment_up_rate"] if c in conf_df.columns]
            st.dataframe(conf_df[display_cols], width="stretch", hide_index=True)

    # ── 6. ML Model Comparison ─────────────────────────────────────────────
    with res_tab_ml:
        st.subheader("ML models: with sentiment vs without sentiment")
        st.caption("Dự đoán chiều giá ngày tiếp theo. Train/test split theo thời gian (70/30). Window = số ngày lookback.")
        if st.button("Chạy (mất ~1–2 phút)", key="run_ml"):
            cached_ml_comparison.clear()
        ml_df = cached_ml_comparison(res_sd, res_ed)
        if ml_df.empty:
            st.warning("Không đủ dữ liệu hoặc chưa chạy.")
        else:
            for metric_col, label in [("f1_delta", "F1 delta"), ("accuracy_delta", "Accuracy delta"), ("auc_delta", "AUC delta")]:
                if metric_col not in ml_df.columns:
                    continue
                sub = ml_df[["scope", "window", "model", metric_col, f"f1_with_sentiment", f"f1_no_sentiment"]].copy() if "f1_with_sentiment" in ml_df.columns else ml_df[["scope", "window", "model", metric_col]].copy()
                sub = sub[sub["scope"] == "ALL"].dropna(subset=[metric_col])
                if sub.empty:
                    continue
                fig = px.bar(sub, x="model", y=metric_col, color="window", barmode="group",
                             title=f"{label} (with_sentiment – no_sentiment) — positif = sentiment helps",
                             labels={metric_col: label})
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, width="stretch")
                break
            st.dataframe(ml_df, width="stretch", hide_index=True)

    # ── 7. Granger-style ML ────────────────────────────────────────────────
    with res_tab_granger_ml:
        st.subheader("Granger-style ML: restricted vs sentiment-augmented")
        st.caption("Restricted = past return + past volume. With_sentiment = restricted + past sentiment_score.")
        if st.button("Chạy (mất ~1–2 phút)", key="run_gml"):
            cached_granger_style.clear()
        gml_df = cached_granger_style(res_sd, res_ed)
        if gml_df.empty:
            st.warning("Không đủ dữ liệu hoặc chưa chạy.")
        else:
            cls_delta = gml_df[gml_df.get("task_with_sentiment", gml_df.get("task", pd.Series(dtype=str))) == "classification"] if "task_with_sentiment" in gml_df.columns or "task" in gml_df.columns else gml_df
            for delta_col, label in [("f1_delta", "F1 delta"), ("auc_delta", "AUC delta"), ("r2_delta", "R² delta")]:
                if delta_col not in gml_df.columns:
                    continue
                sub = gml_df[["scope", "window", "model", "task_with_sentiment" if "task_with_sentiment" in gml_df.columns else "task", delta_col]].dropna(subset=[delta_col]) if ("task_with_sentiment" in gml_df.columns or "task" in gml_df.columns) else gml_df[["scope", "window", "model", delta_col]].dropna(subset=[delta_col])
                sub = sub[sub["scope"] == "ALL"] if "scope" in sub.columns else sub
                if sub.empty:
                    continue
                task_col = "task_with_sentiment" if "task_with_sentiment" in sub.columns else ("task" if "task" in sub.columns else None)
                color_col = task_col if task_col else "window"
                fig = px.bar(sub, x="model", y=delta_col, color=color_col, barmode="group",
                             title=f"{label} — with_sentiment vs restricted",
                             labels={delta_col: label})
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, width="stretch")
                break
            st.dataframe(gml_df, width="stretch", hide_index=True)


# =========================================================
# TAB 9: TÍN HIỆU HÔM NAY  (analog forecasting)
# =========================================================

@st.cache_data(ttl=1800, show_spinner="Đang tìm pattern tương đồng từ lịch sử...")
def cached_analog_signals(
    as_of_date: str,
    window: int = 5,
    top_k: int = 15,
    horizon: int = 5,
) -> pd.DataFrame:
    from src.analysis.analog_forecast import run_analog_forecast
    panel = load_research_panel("2024-01-01", as_of_date)
    if panel.empty:
        return pd.DataFrame()
    return run_analog_forecast(panel, as_of_date=as_of_date,
                               window=window, top_k=top_k, horizon=horizon)


@st.cache_data(ttl=1800, show_spinner="Đang tính lịch sử tín hiệu...")
def cached_ticker_signal_history(
    ticker: str,
    end_date: str,
    n_days: int = 15,
    window: int = 5,
    top_k: int = 15,
    horizon: int = 5,
) -> pd.DataFrame:
    """Run analog forecast for each of the last n_days trading dates for one ticker."""
    from src.analysis.analog_forecast import run_analog_forecast
    panel = load_research_panel("2024-01-01", end_date)
    if panel.empty:
        return pd.DataFrame()
    ticker_panel = panel[panel["ticker"] == ticker]
    if ticker_panel.empty:
        return pd.DataFrame()
    # Get last n_days unique dates for this ticker
    dates = sorted(ticker_panel["date"].dropna().unique())[-n_days:]
    rows = []
    for d in dates:
        result = run_analog_forecast(panel, as_of_date=d,
                                     window=window, top_k=top_k, horizon=horizon)
        if result.empty:
            continue
        row = result[result["ticker"] == ticker]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append({
            "date": pd.Timestamp(d),
            "signal": r["signal"],
            "today_sentiment": r["today_sentiment"],
            "today_news_count": r["today_news_count"],
            "win_rate": r["win_rate"],
            "avg_fwd_5d": r["avg_fwd_5d"],
            "confidence": r["confidence"],
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _spaghetti_chart(fwd_paths: list, horizon: int, ticker: str) -> go.Figure:
    import numpy as _np
    fig = go.Figure()
    x = list(range(0, horizon + 1))
    for path in fwd_paths:
        y = [0.0] + [float(v) for v in path]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="rgba(100, 120, 200, 0.18)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    if fwd_paths:
        mean_path = _np.array(fwd_paths).mean(axis=0)
        fig.add_trace(go.Scatter(
            x=x, y=[0.0] + mean_path.tolist(),
            mode="lines", name="Trung bình",
            line=dict(color="#1f77b4", width=3),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"{ticker} — Các kịch bản lịch sử sau pattern tương tự",
        xaxis_title="Ngày sau sự kiện",
        yaxis_title="Cumulative log-return",
        height=350,
        margin=dict(t=45, b=30),
    )
    return fig


def _signal_history_chart(hist_df: pd.DataFrame, ticker: str, horizon: int) -> go.Figure:
    _sig_colors = {"BULLISH": "#2ecc71", "BEARISH": "#e74c3c", "NEUTRAL": "#95a5a6"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["win_rate"] * 100,
        mode="lines+markers",
        marker=dict(
            color=[_sig_colors.get(s, "#aaa") for s in hist_df["signal"]],
            size=10, line=dict(width=1, color="white"),
        ),
        line=dict(color="rgba(150,150,150,0.4)", width=1),
        text=hist_df["signal"],
        hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{text}<br>Win rate: %{y:.1f}%<extra></extra>",
        name="Win rate",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="50%")
    fig.update_layout(
        title=f"{ticker} — Lịch sử tín hiệu (win rate analog {horizon}d)",
        xaxis_title="Ngày", yaxis_title="Win rate (%)",
        yaxis=dict(range=[0, 105]),
        height=300, margin=dict(t=45, b=30),
    )
    return fig


with tab_signal:
    st.header("📡 Tín hiệu hôm nay")
    st.caption(
        "Tìm các khoảng thời gian trong lịch sử (từ 2024) có pattern giống nhất "
        "với những ngày gần đây, rồi xem chuyện gì đã xảy ra sau đó."
    )

    # ── Settings ─────────────────────────────────────────────────────────────
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    with sig_col1:
        import datetime as _dt_signal
        _max_sig_date = _dt_signal.date.today()
        sig_date = st.date_input(
            "Ngày tham chiếu", value=_max_sig_date,
            max_value=_max_sig_date, key="sig_date",
        )
    with sig_col2:
        sig_window = st.slider("Window (ngày nhìn lại)", 3, 15, 5, key="sig_window")
    with sig_col3:
        sig_topk = st.slider("Top-K analogs", 5, 30, 15, key="sig_topk")
    with sig_col4:
        sig_horizon = st.slider("Horizon dự báo (ngày)", 1, 10, 5, key="sig_horizon")

    if st.button("🔄 Tính lại", key="sig_refresh"):
        cached_analog_signals.clear()
        cached_ticker_signal_history.clear()

    # ── Compute all-ticker signals ────────────────────────────────────────────
    sig_df = cached_analog_signals(
        str(sig_date), window=sig_window, top_k=sig_topk, horizon=sig_horizon,
    )

    if sig_df.empty:
        st.warning("Không đủ dữ liệu. Hãy chạy pipeline để có dữ liệu từ 2024.")
    else:
        # ── Ticker selector (primary filter) ─────────────────────────────────
        _ALL = "— Tất cả —"
        ticker_options = [_ALL] + sig_df["ticker"].tolist()
        selected_ticker = st.selectbox(
            "🔍 Xem chi tiết theo ticker", ticker_options, index=0, key="sig_ticker",
        )

        def _fmt_signal(s: str) -> str:
            return {"BULLISH": "🟢 BULLISH", "BEARISH": "🔴 BEARISH", "NEUTRAL": "⚪ NEUTRAL"}.get(s, s)

        # ── All-ticker view ───────────────────────────────────────────────────
        if selected_ticker == _ALL:
            n_bull = int((sig_df["signal"] == "BULLISH").sum())
            n_bear = int((sig_df["signal"] == "BEARISH").sum())
            n_neu  = int((sig_df["signal"] == "NEUTRAL").sum())
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🟢 BULLISH", n_bull)
            mc2.metric("🔴 BEARISH", n_bear)
            mc3.metric("⚪ NEUTRAL", n_neu)

            display = sig_df[[
                "ticker", "signal", "today_sentiment", "today_news_count",
                "recent_5d_return", "win_rate", "avg_fwd_5d", "n_analogs", "confidence",
            ]].copy()
            display["signal"] = display["signal"].map(_fmt_signal)
            display["win_rate_pct"]   = display["win_rate"] * 100
            display["avg_fwd_pct"]    = display["avg_fwd_5d"] * 100
            display["recent_ret_pct"] = display["recent_5d_return"] * 100
            display["conf_pct"]       = display["confidence"] * 100
            st.dataframe(
                display[[
                    "ticker", "signal", "today_sentiment", "today_news_count",
                    "recent_ret_pct", "win_rate_pct", "avg_fwd_pct", "n_analogs", "conf_pct",
                ]],
                column_config={
                    "ticker":           st.column_config.TextColumn("Ticker"),
                    "signal":           st.column_config.TextColumn("Signal"),
                    "today_sentiment":  st.column_config.NumberColumn("Sentiment hôm nay", format="%.3f"),
                    "today_news_count": st.column_config.NumberColumn("Số tin", format="%d"),
                    "recent_ret_pct":   st.column_config.NumberColumn(f"Return {sig_window}d gần (%)", format="%.2f%%"),
                    "win_rate_pct":     st.column_config.NumberColumn("Win rate (%)", format="%.1f%%"),
                    "avg_fwd_pct":      st.column_config.NumberColumn(f"Avg return {sig_horizon}d tới (%)", format="%.2f%%"),
                    "n_analogs":        st.column_config.NumberColumn("Số analogs", format="%d"),
                    "conf_pct":         st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
                },
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Window = {sig_window} ngày | Top-K = {sig_topk} analogs | "
                f"Horizon = {sig_horizon} ngày | Chỉ dùng dữ liệu từ 2024."
            )

        # ── Single-ticker detail view ─────────────────────────────────────────
        else:
            row = sig_df[sig_df["ticker"] == selected_ticker].iloc[0]

            # Big signal card
            _sig_label = _fmt_signal(row["signal"])
            st.subheader(f"{selected_ticker}   {_sig_label}")
            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            dc1.metric("Sentiment hôm nay", f"{row['today_sentiment']:.3f}")
            dc2.metric("Số tin hôm nay", int(row["today_news_count"]))
            dc3.metric(f"Return {sig_window}d gần", f"{row['recent_5d_return']*100:.2f}%")
            dc4.metric("Win rate analogs", f"{row['win_rate']*100:.1f}%")
            dc5.metric(f"Avg return {sig_horizon}d tới", f"{row['avg_fwd_5d']*100:.2f}%")

            # Signal history for this ticker (last 15 trading days)
            st.subheader(f"Lịch sử tín hiệu — {selected_ticker} (15 ngày gần nhất)")
            hist_df = cached_ticker_signal_history(
                selected_ticker, str(sig_date),
                n_days=15, window=sig_window, top_k=sig_topk, horizon=sig_horizon,
            )
            if not hist_df.empty:
                st.plotly_chart(
                    _signal_history_chart(hist_df, selected_ticker, sig_horizon),
                    width="stretch",
                )
                # Mini table of recent days
                hist_display = hist_df.copy()
                hist_display["signal"] = hist_display["signal"].map(_fmt_signal)
                hist_display["win_rate_pct"] = hist_display["win_rate"] * 100
                hist_display["avg_fwd_pct"]  = hist_display["avg_fwd_5d"] * 100
                hist_display["conf_pct"]     = hist_display["confidence"] * 100
                hist_display["date_str"]     = hist_display["date"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    hist_display[[
                        "date_str", "signal", "today_sentiment", "today_news_count",
                        "win_rate_pct", "avg_fwd_pct", "conf_pct",
                    ]],
                    column_config={
                        "date_str":         st.column_config.TextColumn("Ngày"),
                        "signal":           st.column_config.TextColumn("Signal"),
                        "today_sentiment":  st.column_config.NumberColumn("Sentiment", format="%.3f"),
                        "today_news_count": st.column_config.NumberColumn("Số tin", format="%d"),
                        "win_rate_pct":     st.column_config.NumberColumn("Win rate (%)", format="%.1f%%"),
                        "avg_fwd_pct":      st.column_config.NumberColumn(f"Avg {sig_horizon}d (%)", format="%.2f%%"),
                        "conf_pct":         st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
                    },
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("Chưa đủ dữ liệu để tính lịch sử tín hiệu.")

            # Analog dates table with raw feature values
            st.subheader(f"Các ngày tương đồng được chọn — {selected_ticker}")
            analog_details = row.get("analog_details", [])
            current_window = row.get("current_window", [])
            if analog_details:
                ad_df = pd.DataFrame(analog_details)
                ad_df["analog_date"] = pd.to_datetime(ad_df["analog_date"]).dt.strftime("%Y-%m-%d")
                ad_df["fwd_return_pct"] = ad_df["fwd_return"] * 100
                ad_df["outcome"] = ad_df["fwd_return"].apply(lambda x: "✅ UP" if x > 0 else "❌ DOWN")

                # Build "today" row with same feature columns
                today_row: dict = {
                    "rank": 0, "analog_date": f"📍 Hôm nay ({sig_date})",
                    "distance": 0.0, "fwd_return_pct": float("nan"), "outcome": "—",
                }
                if current_window:
                    cw = current_window  # list of [ret, vol, sent] per day
                    for day_i in range(sig_window):
                        today_row[f"ret_{day_i - sig_window + 1}"] = float(cw[day_i][0])
                        today_row[f"vol_{day_i - sig_window + 1}"] = float(cw[day_i][1])
                        today_row[f"sent_{day_i - sig_window + 1}"] = float(cw[day_i][2])

                display_df = pd.concat(
                    [pd.DataFrame([today_row]), ad_df], ignore_index=True
                )

                col_cfg: dict = {
                    "rank":           st.column_config.NumberColumn("#", format="%d"),
                    "analog_date":    st.column_config.TextColumn("Ngày"),
                    "distance":       st.column_config.NumberColumn("Distance", format="%.2f"),
                    "fwd_return_pct": st.column_config.NumberColumn(f"Return {sig_horizon}d (%)", format="%.2f%%"),
                    "outcome":        st.column_config.TextColumn("Kết quả"),
                }
                for day_i in range(sig_window):
                    d = day_i - sig_window + 1
                    col_cfg[f"ret_{d}"]  = st.column_config.NumberColumn(f"ret[{d}]",  format="%.3f")
                    col_cfg[f"vol_{d}"]  = st.column_config.NumberColumn(f"vol[{d}]",  format="%.3f")
                    col_cfg[f"sent_{d}"] = st.column_config.NumberColumn(f"sent[{d}]", format="%.3f")

                show_cols = (
                    ["rank", "analog_date", "distance", "fwd_return_pct", "outcome"]
                    + [f"ret_{day_i - sig_window + 1}"  for day_i in range(sig_window)]
                    + [f"vol_{day_i - sig_window + 1}"  for day_i in range(sig_window)]
                    + [f"sent_{day_i - sig_window + 1}" for day_i in range(sig_window)]
                )
                show_cols = [c for c in show_cols if c in display_df.columns]
                st.dataframe(display_df[show_cols], column_config=col_cfg,
                             width="stretch", hide_index=True)
            else:
                st.info("Không có thông tin analog.")

            # Spaghetti chart for today's analogs
            st.subheader(f"Kịch bản các analog — {selected_ticker} (ngày {sig_date})")
            paths = row.get("fwd_paths", [])
            if paths:
                st.plotly_chart(
                    _spaghetti_chart(paths, sig_horizon, selected_ticker),
                    width="stretch",
                )
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Win Rate", f"{row['win_rate']*100:.1f}%")
                sc2.metric(f"Avg {sig_horizon}d Return", f"{row['avg_fwd_5d']*100:.2f}%")
                sc3.metric(f"Median {sig_horizon}d Return", f"{row['median_fwd_5d']*100:.2f}%")
                sc4.metric("Số analogs", int(row["n_analogs"]))
            else:
                st.info("Không đủ analog paths để vẽ.")

    st.divider()
    st.warning(
        "⚠️ Đây là phân tích tương đồng lịch sử — **không phải khuyến nghị đầu tư**. "
        "Win rate ~60% có nghĩa là 40% lần vẫn đi ngược chiều. "
        "Luôn kết hợp với phân tích cơ bản, thanh khoản và rủi ro thị trường."
    )


st.divider()
st.caption(
    "Sentiment index phản ánh sắc thái tin tức, không phải khuyến nghị đầu tư. "
    "Cần xem cùng giá, thanh khoản, thị trường chung và rủi ro doanh nghiệp."
)

# Auto-rerun every 1 s while the background update thread is running
if _PIPELINE_STATE["running"]:
    import time
    time.sleep(1)
    st.rerun()