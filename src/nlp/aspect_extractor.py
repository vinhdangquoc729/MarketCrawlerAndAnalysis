"""Rule-based aspect extractor using the same aspect schema as LLM labeling / fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing.text_cleaner import clean_text, normalize_for_matching


CANONICAL_ASPECTS = {
    "business_financials",
    "leadership_insider",
    "macro_policy",
    "capital_structure",
    "legal_esg",
}


ASPECT_DESCRIPTIONS = {
    "business_financials": (
        "Doanh thu, lợi nhuận, dòng tiền, core kinh doanh, hợp đồng mới, "
        "mở rộng quy mô, chi phí đầu vào."
    ),
    "leadership_insider": (
        "Động thái ban lãnh đạo, giao dịch nội bộ, thay đổi nhân sự cấp cao."
    ),
    "macro_policy": (
        "Yếu tố vĩ mô, chính sách ngành, tỷ giá, lãi suất."
    ),
    "capital_structure": (
        "Cấu trúc vốn, trả cổ tức, phát hành thêm, mua bán cổ phiếu quỹ, huy động vốn."
    ),
    "legal_esg": (
        "Rủi ro pháp lý và tuân thủ ESG: khởi tố, xả thải, đình công, vi phạm."
    ),
}


ASPECT_KEYWORDS: dict[str, list[str]] = {
    "business_financials": [
        # doanh thu / lợi nhuận
        "doanh thu",
        "lợi nhuận",
        "lãi ròng",
        "lãi sau thuế",
        "lãi trước thuế",
        "lãi gộp",
        "biên lợi nhuận",
        "biên gộp",
        "eps",
        "roe",
        "roa",
        "ebitda",

        # dòng tiền / tài chính vận hành
        "dòng tiền",
        "lưu chuyển tiền",
        "cash flow",
        "tiền mặt",
        "chi phí",
        "chi phí đầu vào",
        "giá vốn",
        "biên lãi",
        "core kinh doanh",
        "mảng kinh doanh",
        "hoạt động kinh doanh",

        # hợp đồng / đơn hàng / mở rộng
        "hợp đồng mới",
        "gói thầu",
        "trúng thầu",
        "đơn hàng",
        "backlog",
        "mở rộng quy mô",
        "mở rộng công suất",
        "công suất",
        "nhà máy mới",
        "dự án mới",
        "khai trương",
        "vận hành",
        "sản lượng",
        "tiêu thụ",
        "xuất khẩu",
        "tăng trưởng",
        "kết quả kinh doanh",
        "báo cáo tài chính",
        "bctc",
    ],

    "leadership_insider": [
        "ban lãnh đạo",
        "lãnh đạo",
        "chủ tịch",
        "phó chủ tịch",
        "tổng giám đốc",
        "phó tổng giám đốc",
        "ceo",
        "cfo",
        "thành viên hđqt",
        "hội đồng quản trị",
        "hđqt",
        "ban kiểm soát",
        "bổ nhiệm",
        "miễn nhiệm",
        "từ nhiệm",
        "thay đổi nhân sự",
        "nhân sự cấp cao",
        "người nội bộ",
        "cổ đông nội bộ",
        "giao dịch nội bộ",
        "đăng ký mua",
        "đăng ký bán",
        "mua vào",
        "bán ra",
        "thoái vốn của lãnh đạo",
        "con trai chủ tịch",
        "con gái chủ tịch",
        "vợ chủ tịch",
        "người liên quan",
    ],

    "macro_policy": [
        "vĩ mô",
        "chính sách",
        "chính sách ngành",
        "ngân hàng nhà nước",
        "nhnn",
        "bộ tài chính",
        "chính phủ",
        "quốc hội",
        "nghị định",
        "thông tư",
        "quy định",
        "lãi suất",
        "hạ lãi suất",
        "tăng lãi suất",
        "tỷ giá",
        "usd",
        "vnd",
        "ngoại hối",
        "lạm phát",
        "cpi",
        "gdp",
        "pmi",
        "tăng trưởng tín dụng",
        "room tín dụng",
        "đầu tư công",
        "giải ngân",
        "thuế",
        "thuế quan",
        "xuất nhập khẩu",
        "fdI",
        "giá dầu",
        "giá thép",
        "giá vàng",
        "giá hàng hóa",
        "fed",
        "trung quốc",
        "mỹ",
        "địa chính trị",
    ],

    "capital_structure": [
        "cấu trúc vốn",
        "vốn điều lệ",
        "tăng vốn",
        "huy động vốn",
        "phát hành",
        "phát hành thêm",
        "chào bán",
        "chào bán riêng lẻ",
        "phát hành riêng lẻ",
        "quyền mua",
        "pha loãng",
        "cổ tức",
        "trả cổ tức",
        "chia cổ tức",
        "tạm ứng cổ tức",
        "cổ phiếu thưởng",
        "mua cổ phiếu quỹ",
        "bán cổ phiếu quỹ",
        "cổ phiếu quỹ",
        "trái phiếu",
        "trái phiếu chuyển đổi",
        "nợ vay",
        "vay nợ",
        "dư nợ",
        "đáo hạn",
        "tái cơ cấu nợ",
        "room ngoại",
        "sở hữu nước ngoài",
        "free float",
    ],

    "legal_esg": [
        # pháp lý
        "pháp lý",
        "rủi ro pháp lý",
        "tuân thủ",
        "khởi tố",
        "bị khởi tố",
        "điều tra",
        "bị điều tra",
        "bắt tạm giam",
        "tạm giam",
        "truy tố",
        "xét xử",
        "thi hành án",
        "cưỡng chế",
        "phong tỏa",
        "kê biên",
        "xử phạt",
        "phạt tiền",
        "vi phạm",
        "sai phạm",
        "thanh tra",
        "kiểm tra",
        "kiểm toán",
        "hủy niêm yết",
        "đình chỉ giao dịch",
        "hạn chế giao dịch",
        "cảnh báo",
        "kiểm soát",
        "kiểm soát đặc biệt",
        "tranh chấp",
        "kiện",
        "khởi kiện",
        "tố cáo",

        # ESG / môi trường / lao động
        "esg",
        "môi trường",
        "xả thải",
        "ô nhiễm",
        "cháy nổ",
        "tai nạn lao động",
        "đình công",
        "ngừng việc",
        "lao động",
        "an toàn lao động",
        "phát thải",
        "carbon",
        "quản trị công ty",
        "minh bạch",
    ],
}


# Một số keyword có thể thuộc nhiều aspect.
# Ví dụ "trái phiếu" vừa liên quan capital_structure, vừa có thể legal_esg nếu đi cùng "vi phạm", "điều tra".
LEGAL_STRONG_KEYWORDS = {
    "khởi tố",
    "bị khởi tố",
    "điều tra",
    "bị điều tra",
    "bắt tạm giam",
    "tạm giam",
    "truy tố",
    "xét xử",
    "thi hành án",
    "cưỡng chế",
    "phong tỏa",
    "kê biên",
    "xử phạt",
    "vi phạm",
    "sai phạm",
    "xả thải",
    "ô nhiễm",
    "đình công",
}


@dataclass(frozen=True)
class AspectResult:
    aspect: str
    keywords: list[str]
    score: float = 0.0


def _score_aspect(aspect: str, matched_keywords: list[str], normalized_context: str) -> float:
    """Score aspect by number and importance of matched keywords."""
    unique_keywords = set(matched_keywords)
    score = float(len(unique_keywords))

    if aspect == "legal_esg":
        for keyword in LEGAL_STRONG_KEYWORDS:
            keyword_norm = normalize_for_matching(keyword)
            if keyword_norm in normalized_context:
                score += 2.0

    return score


def extract_aspects(
    context: str,
    max_aspects: int = 3,
    allow_general: bool = True,
) -> list[AspectResult]:
    """Extract canonical aspects from entity context.

    Output aspect labels are restricted to:
    - business_financials
    - leadership_insider
    - macro_policy
    - capital_structure
    - legal_esg

    Args:
        context: Entity-specific context.
        max_aspects: Maximum aspects returned per entity context.
        allow_general: If True and no aspect is detected, return business_financials
            as a safe default for company news. If False, return empty list.

    Returns:
        List of AspectResult sorted by score descending.
    """
    raw_context = clean_text(context)
    normalized_context = normalize_for_matching(raw_context)

    results: list[AspectResult] = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if aspect not in CANONICAL_ASPECTS:
            continue

        matched: list[str] = []

        for keyword in keywords:
            keyword_norm = normalize_for_matching(keyword)

            if keyword_norm and keyword_norm in normalized_context:
                matched.append(keyword)

        if matched:
            unique_keywords = sorted(set(matched))
            results.append(
                AspectResult(
                    aspect=aspect,
                    keywords=unique_keywords,
                    score=_score_aspect(aspect, unique_keywords, normalized_context),
                )
            )

    results = sorted(results, key=lambda x: x.score, reverse=True)

    if results:
        return results[:max_aspects]

    # Không nên sinh nhãn ngoài 5 aspect vì model train chỉ biết 5 aspect này.
    # Nếu bắt buộc phải có 1 aspect để chạy model, dùng business_financials làm default nhẹ.
    if allow_general:
        return [
            AspectResult(
                aspect="business_financials",
                keywords=[],
                score=0.0,
            )
        ]

    return []