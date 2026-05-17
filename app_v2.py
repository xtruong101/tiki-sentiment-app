"""
app.py – Demo Phân Tích Cảm Xúc Bình Luận Tiki
====================================================
Chạy:   streamlit run app.py
Yêu cầu:  thư mục artifacts/ chứa label_model.joblib và rating_model.joblib
           (được sinh ra sau khi chạy notebook huấn luyện)
"""

import io
import re
import html
import json
import time
import random
import unicodedata
import warnings
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CẤU HÌNH TRANG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tiki Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CSS CUSTOM
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Màu nền nhẹ */
    .main { background-color: #f7f9fc; }

    /* Card chứa kết quả */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Badge cảm xúc */
    .badge-pos  { background:#d1fae5; color:#065f46; padding:4px 14px; border-radius:20px; font-weight:600; }
    .badge-neu  { background:#fef9c3; color:#713f12; padding:4px 14px; border-radius:20px; font-weight:600; }
    .badge-neg  { background:#fee2e2; color:#991b1b; padding:4px 14px; border-radius:20px; font-weight:600; }

    /* Tiêu đề phần */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 6px;
    }

    /* Chatbot bubble */
    .chat-bubble {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.97rem;
        color: #1e40af;
    }

    /* Star display */
    .stars { font-size: 1.5rem; letter-spacing: 2px; }

    /* Divider */
    hr.light { border: 0; border-top: 1px solid #e2e8f0; margin: 16px 0; }

    /* ── Priority alert boxes ── */
    .priority-high {
        background: #fff1f2;
        border: 1.5px solid #f43f5e;
        border-left: 5px solid #f43f5e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
    }
    .priority-high .p-title  { color:#be123c; font-size:1.05rem; font-weight:700; margin-bottom:6px; }
    .priority-high .p-body   { color:#9f1239; font-size:0.93rem; }

    .priority-mid {
        background: #fffbeb;
        border: 1.5px solid #f59e0b;
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
    }
    .priority-mid .p-title   { color:#92400e; font-size:1.05rem; font-weight:700; margin-bottom:6px; }
    .priority-mid .p-body    { color:#78350f; font-size:0.93rem; }

    .priority-low {
        background: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
    }
    .priority-low .p-title   { color:#15803d; font-size:1.05rem; font-weight:700; margin-bottom:6px; }
    .priority-low .p-body    { color:#166534; font-size:0.93rem; }

    /* Badge ưu tiên inline (dùng trong bảng) */
    .p-badge-high { background:#fda4af; color:#9f1239; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.82rem; }
    .p-badge-mid  { background:#fde68a; color:#92400e; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.82rem; }
    .p-badge-low  { background:#bbf7d0; color:#166534; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TIỀN XỬ LÝ TIẾNG VIỆT  (copy từ notebook)
# ──────────────────────────────────────────────
SLANG_MAP = {
    r'\bko\b': 'không', r'\bk\b': 'không', r'\bkhong\b': 'không', r'\bhok\b': 'không',
    r'\bhong\b': 'không', r'\bkg\b': 'không', r'\bkh\b': 'không', r'\bđc\b': 'được',
    r'\bdc\b': 'được', r'\bok\b': 'ổn', r'\boke\b': 'ổn', r'\boki\b': 'ổn',
    r'\bsp\b': 'sản phẩm', r'\bspham\b': 'sản phẩm', r'\bsx\b': 'sản xuất',
    r'\bhdsd\b': 'hướng dẫn sử dụng', r'\bmn\b': 'mọi người', r'\bmk\b': 'mình',
    r'\bmik\b': 'mình', r'\bvs\b': 'với', r'\bwa\b': 'quá', r'\bqa\b': 'quá',
    r'\bqá\b': 'quá', r'\bdep\b': 'đẹp', r'\btot\b': 'tốt', r'\bxau\b': 'xấu',
    r'\bnhanhh\b': 'nhanh', r'\bship\b': 'giao hàng',
}
POS_EMOJI_RE = re.compile('[😀😃😄😁😆😊🙂😍🥰😘👍❤♥️✨⭐🌟]')
NEG_EMOJI_RE = re.compile('[😞😔😟😢😭😡😠👎💔]')
NEGATIONS    = {'không', 'chưa', 'chẳng', 'chả', 'ko', 'k', 'khong', 'hok', 'hong'}
PUNCT_RE     = re.compile(r'[^0-9a-zA-ZÀ-ỹ_\s]')
MULTISPACE_RE = re.compile(r'\s+')
REPEAT_CHAR_RE = re.compile(r'(.)\1{2,}')
URL_RE       = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE     = re.compile(r'\S+@\S+')
PHONE_RE     = re.compile(r'\b\d{9,11}\b')

def normalize_slang(text: str) -> str:
    for pattern, repl in SLANG_MAP.items():
        text = re.sub(pattern, repl, text)
    return text

def join_negation(text: str) -> str:
    tokens = text.split()
    new_tokens, i = [], 0
    while i < len(tokens):
        if tokens[i] in NEGATIONS and i + 1 < len(tokens):
            new_tokens.append(tokens[i] + '_' + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return ' '.join(new_tokens)

def clean_comment(text: str) -> str:
    text = unicodedata.normalize('NFC', str(text))
    text = html.unescape(text)
    text = text.lower().strip()
    text = POS_EMOJI_RE.sub(' positive_emoji ', text)
    text = NEG_EMOJI_RE.sub(' negative_emoji ', text)
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = PHONE_RE.sub(' ', text)
    text = REPEAT_CHAR_RE.sub(r'\1\1', text)
    text = normalize_slang(text)
    text = PUNCT_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text).strip()
    text = normalize_slang(text)
    text = MULTISPACE_RE.sub(' ', text).strip()
    text = join_negation(text)
    return text

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
ARTIFACT_DIR = Path("artifacts")

LABEL_NAME = {'NEG': '😞 Tiêu cực', 'NEU': '😐 Trung tính', 'POS': '😊 Tích cực'}
RATING_NAME = {
    1: '⭐ Rất không hài lòng',
    2: '⭐⭐ Không hài lòng',
    3: '⭐⭐⭐ Bình thường',
    4: '⭐⭐⭐⭐ Hài lòng',
    5: '⭐⭐⭐⭐⭐ Rất hài lòng',
}

@st.cache_resource
def load_models():
    label_path  = ARTIFACT_DIR / "label_model.joblib"
    rating_path = ARTIFACT_DIR / "rating_model.joblib"
    if not label_path.exists() or not rating_path.exists():
        return None, None
    return joblib.load(label_path), joblib.load(rating_path)

label_model, rating_model = load_models()

def models_ready():
    return label_model is not None and rating_model is not None

# ──────────────────────────────────────────────
# HÀM DỰ ĐOÁN
# ──────────────────────────────────────────────
def margin_confidence(model, texts):
    scores = np.asarray(model.decision_function(texts))
    if scores.ndim == 1:
        return np.abs(scores)
    sorted_s = np.sort(scores, axis=1)
    return sorted_s[:, -1] - sorted_s[:, -2]

def predict_single(comment: str) -> dict:
    """Dự đoán một bình luận → trả về dict đầy đủ (kể cả mức ưu tiên)."""
    clean  = clean_comment(comment)
    label  = label_model.predict([clean])[0]
    rate   = int(rating_model.predict([clean])[0])
    conf   = float(margin_confidence(label_model, [clean])[0])
    p      = get_priority(label, rate)
    return {
        'original':    comment,
        'clean':       clean,
        'label':       label,
        'label_name':  LABEL_NAME[label],
        'rating':      rate,
        'rating_name': RATING_NAME[rate],
        'confidence':  round(conf, 4),
        'priority':    p,                      # dict đầy đủ
        'priority_level':      p['level'],
        'priority_suggestion': p['suggestion'],
    }

# ──────────────────────────────────────────────
# CHATBOT TRẢ LỜI TỰ ĐỘNG
# ──────────────────────────────────────────────
RESPONSE_TEMPLATES = {
    'POS': [
        'Cảm ơn bạn rất nhiều vì đã tin tưởng và ủng hộ shop! Shop rất vui khi bạn hài lòng với sản phẩm. 🙏',
        'Shop cảm ơn đánh giá tích cực của bạn. Hy vọng sẽ tiếp tục được phục vụ bạn trong những lần mua sau! 😊',
        'Cảm ơn bạn đã phản hồi. Sự hài lòng của bạn là động lực để shop tiếp tục cải thiện chất lượng dịch vụ. ❤️',
    ],
    'NEU': [
        'Cảm ơn bạn đã dành thời gian phản hồi. Shop ghi nhận góp ý của bạn và sẽ cố gắng cải thiện hơn.',
        'Shop cảm ơn đánh giá trung thực của bạn. Nếu cần hỗ trợ thêm, bạn có thể nhắn trực tiếp cho shop nhé.',
        'Cảm ơn bạn đã góp ý. Shop sẽ tiếp tục cải thiện sản phẩm và dịch vụ để mang lại trải nghiệm tốt hơn. 🌟',
    ],
    'NEG': [
        'Shop thành thật xin lỗi vì trải nghiệm chưa tốt của bạn. Bạn vui lòng liên hệ shop để được hỗ trợ xử lý sớm nhất. 🙏',
        'Rất xin lỗi bạn về vấn đề gặp phải. Shop ghi nhận phản hồi và sẽ kiểm tra lại để hỗ trợ bạn tốt hơn.',
        'Shop xin lỗi vì sản phẩm/dịch vụ chưa đáp ứng kỳ vọng. Bạn vui lòng nhắn cho shop để được hỗ trợ đổi trả nếu cần.',
    ],
}
KEYWORD_EXTRA = {
    'giao': ' Shop sẽ kiểm tra thêm với đơn vị vận chuyển để cải thiện tốc độ giao hàng.',
    'ship': ' Shop sẽ kiểm tra thêm với đơn vị vận chuyển để cải thiện tốc độ giao hàng.',
    'đóng gói': ' Shop sẽ lưu ý kỹ hơn khâu đóng gói để sản phẩm đến tay khách hàng an toàn hơn.',
    'lỗi': ' Shop sẽ kiểm tra lại tình trạng lỗi và hỗ trợ theo chính sách đổi trả/bảo hành.',
    'hỏng': ' Shop sẽ kiểm tra lại và hỗ trợ bạn theo chính sách đổi trả/bảo hành.',
    'size': ' Shop sẽ bổ sung hướng dẫn chọn size rõ ràng hơn để khách dễ lựa chọn.',
    'màu': ' Shop sẽ cố gắng cập nhật hình ảnh/màu sắc sản phẩm sát thực tế hơn.',
    'chất lượng': ' Shop luôn ghi nhận phản hồi về chất lượng để cải thiện sản phẩm tốt hơn.',
}

def generate_reply(pred: dict) -> str:
    reply = random.choice(RESPONSE_TEMPLATES[pred['label']])
    for kw, extra in KEYWORD_EXTRA.items():
        if kw in pred['clean']:
            reply += extra
            break
    if pred['confidence'] < 0.2:
        reply += ' _(Lưu ý: bình luận này có sắc thái chưa rõ, nên kiểm tra thêm.)_'
    return reply

# ──────────────────────────────────────────────
# CẢNH BÁO ƯU TIÊN (NEG + rating thấp)
# ──────────────────────────────────────────────

# Quy tắc phân loại mức ưu tiên:
#   CAO        : NEG  + rating 1 hoặc 2  → Xử lý gấp
#   TRUNG BÌNH : NEG  + rating 3         → Theo dõi
#              : NEU  + rating 1 hoặc 2  → Chú ý
#   THẤP       : các trường hợp còn lại

PRIORITY_META = {
    'CAO': {
        'level': 'CAO',
        'emoji': '🔴',
        'css_class': 'priority-high',
        'badge_class': 'p-badge-high',
        'title': '🔴 Ưu tiên CAO – Cần xử lý gấp',
        'suggestion': (
            'Cần phản hồi xin lỗi ngay lập tức. '
            'Hỗ trợ đổi trả hoặc bảo hành theo chính sách. '
            'Kiểm tra lại quy trình kiểm hàng và giao vận để tránh tái diễn.'
        ),
    },
    'TRUNG BÌNH': {
        'level': 'TRUNG BÌNH',
        'emoji': '🟡',
        'css_class': 'priority-mid',
        'badge_class': 'p-badge-mid',
        'title': '🟡 Ưu tiên TRUNG BÌNH – Cần theo dõi',
        'suggestion': (
            'Ghi nhận phản hồi và chủ động liên hệ khách hàng. '
            'Hỏi thêm về tình trạng sản phẩm; nếu phát sinh vấn đề thì hỗ trợ kịp thời.'
        ),
    },
    'THẤP': {
        'level': 'THẤP',
        'emoji': '🟢',
        'css_class': 'priority-low',
        'badge_class': 'p-badge-low',
        'title': '🟢 Bình thường – Không cần ưu tiên',
        'suggestion': '',
    },
}


def get_priority(label: str, rating: int) -> dict:
    """Trả về dict mô tả mức ưu tiên dựa vào label và rating dự đoán."""
    if label == 'NEG' and rating in (1, 2):
        return PRIORITY_META['CAO']
    if (label == 'NEG' and rating == 3) or (label == 'NEU' and rating in (1, 2)):
        return PRIORITY_META['TRUNG BÌNH']
    return PRIORITY_META['THẤP']


def render_priority_alert(pred: dict) -> None:
    """Hiển thị hộp cảnh báo ưu tiên cho một bình luận đơn (Tab 1)."""
    p = pred['priority']
    if p['level'] == 'THẤP':
        st.markdown(
            f'<div class="{p["css_class"]}">'
            f'  <div class="p-title">{p["title"]}</div>'
            f'  <div class="p-body">Bình luận này không cần xử lý đặc biệt.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="{p["css_class"]}">'
        f'  <div class="p-title">{p["title"]}</div>'
        f'  <div class="p-body">'
        f'    <b>Mức ưu tiên:</b> {p["level"]}<br>'
        f'    <b>Gợi ý xử lý:</b> {p["suggestion"]}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_priority_table(df_pred: pd.DataFrame) -> None:
    """
    Hiển thị bảng + thống kê cảnh báo ưu tiên cho kết quả phân tích hàng loạt.
    Dùng trong _render_analysis() (Tab 2 & Tab 3).
    """
    # ── Thống kê tổng hợp ──
    total   = len(df_pred)
    n_high  = (df_pred['priority_level'] == 'CAO').sum()
    n_mid   = (df_pred['priority_level'] == 'TRUNG BÌNH').sum()
    n_low   = (df_pred['priority_level'] == 'THẤP').sum()

    st.markdown("### 🚨 Cảnh báo bình luận cần xử lý")

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Ưu tiên CAO",        n_high,
              help="NEG + rating 1-2 sao → cần phản hồi gấp")
    m2.metric("🟡 Ưu tiên TRUNG BÌNH", n_mid,
              help="NEG rating 3 sao, hoặc NEU + rating 1-2 sao")
    m3.metric("🟢 Bình thường",         n_low)

    # ── Biểu đồ ưu tiên ──
    priority_counts = (
        df_pred['priority_level']
        .value_counts()
        .reindex(['CAO', 'TRUNG BÌNH', 'THẤP'], fill_value=0)
    )
    fig_p = go.Figure(go.Bar(
        x=priority_counts.index,
        y=priority_counts.values,
        marker_color=['#f43f5e', '#f59e0b', '#22c55e'],
        text=priority_counts.values,
        textposition='outside',
    ))
    fig_p.update_layout(
        title='Phân bố mức ưu tiên xử lý',
        xaxis_title='Mức ưu tiên',
        yaxis_title='Số bình luận',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=320,
        showlegend=False,
        margin=dict(t=46, b=36, l=36, r=16),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # ── Bảng bình luận CAO ──
    df_high = df_pred[df_pred['priority_level'] == 'CAO']
    if df_high.empty:
        st.success("✅ Không có bình luận nào cần xử lý gấp.")
    else:
        st.error(f"⚠️ Có **{len(df_high)}** bình luận cần xử lý GẤP (NEG + 1-2 ⭐):")
        show_high = df_high[['original', 'label', 'rating', 'priority_suggestion']].copy()
        show_high.columns = ['Bình luận', 'Cảm xúc', 'Rating', 'Gợi ý xử lý']
        st.dataframe(show_high, use_container_width=True, height=280)

    # ── Bảng bình luận TRUNG BÌNH ──
    df_mid = df_pred[df_pred['priority_level'] == 'TRUNG BÌNH']
    if not df_mid.empty:
        with st.expander(f"🟡 Xem {len(df_mid)} bình luận ưu tiên TRUNG BÌNH"):
            show_mid = df_mid[['original', 'label', 'rating', 'priority_suggestion']].copy()
            show_mid.columns = ['Bình luận', 'Cảm xúc', 'Rating', 'Gợi ý xử lý']
            st.dataframe(show_mid, use_container_width=True, height=240)

    # ── Download bảng ưu tiên ──
    urgent_df = df_pred[df_pred['priority_level'].isin(['CAO', 'TRUNG BÌNH'])][
        ['original', 'label', 'rating', 'priority_level', 'priority_suggestion', 'reply']
    ].copy()
    urgent_df.columns = [
        'Bình luận', 'Cảm xúc', 'Rating',
        'Mức ưu tiên', 'Gợi ý xử lý', 'Trả lời gợi ý'
    ]
    if not urgent_df.empty:
        csv_urgent = urgent_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "⬇️ Tải danh sách bình luận cần xử lý (CSV)",
            csv_urgent,
            file_name="binh_luan_can_xu_ly.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────
# PHÂN TÍCH NHIỀU BÌNH LUẬN → DataFrame + biểu đồ
# ──────────────────────────────────────────────
def analyze_comments_df(comments: list[str]) -> pd.DataFrame:
    rows = []
    for c in comments:
        if str(c).strip():
            pred = predict_single(str(c))
            pred['reply'] = generate_reply(pred)
            rows.append(pred)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def draw_sentiment_chart(df_pred: pd.DataFrame) -> go.Figure:
    counts = df_pred['label'].value_counts().reindex(['POS', 'NEU', 'NEG'], fill_value=0)
    colors = {'POS': '#10b981', 'NEU': '#f59e0b', 'NEG': '#ef4444'}
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[colors[l] for l in counts.index],
        text=counts.values,
        textposition='outside',
    ))
    fig.update_layout(
        title='Phân bố cảm xúc bình luận',
        xaxis_title='Nhãn cảm xúc',
        yaxis_title='Số lượng bình luận',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=380,
        showlegend=False,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig

def draw_donut_chart(df_pred: pd.DataFrame) -> go.Figure:
    counts = df_pred['label'].value_counts().reindex(['POS', 'NEU', 'NEG'], fill_value=0)
    labels = ['Tích cực', 'Trung tính', 'Tiêu cực']
    colors = ['#10b981', '#f59e0b', '#ef4444']
    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.55,
        marker_colors=colors,
        textinfo='label+percent',
    ))
    fig.update_layout(
        title='Tỉ lệ cảm xúc (%)',
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig

def auto_conclusion(df_pred: pd.DataFrame) -> str:
    total = len(df_pred)
    if total == 0:
        return "Không có dữ liệu."
    counts = df_pred['label'].value_counts()
    pos = counts.get('POS', 0)
    neu = counts.get('NEU', 0)
    neg = counts.get('NEG', 0)
    pos_r = pos / total * 100
    neg_r = neg / total * 100
    avg_rating = df_pred['rating'].mean()

    if pos_r >= 60:
        verdict = "✅ **Đánh giá: Tích cực** – Phần lớn khách hàng hài lòng với sản phẩm."
    elif neg_r >= 40:
        verdict = "❌ **Đánh giá: Tiêu cực** – Sản phẩm có nhiều phản hồi tiêu cực. Người mua nên xem kỹ trước khi quyết định."
    else:
        verdict = "⚠️ **Đánh giá: Trung bình** – Phản hồi của khách hàng khá đa chiều."

    return (
        f"{verdict}\n\n"
        f"- 📊 Tổng bình luận phân tích: **{total}**\n"
        f"- 😊 Tích cực: **{pos}** ({pos_r:.1f}%)\n"
        f"- 😐 Trung tính: **{neu}** ({neu / total * 100:.1f}%)\n"
        f"- 😞 Tiêu cực: **{neg}** ({neg_r:.1f}%)\n"
        f"- ⭐ Rating dự đoán trung bình: **{avg_rating:.2f} / 5**"
    )

# ──────────────────────────────────────────────
# TÁCH product_id + spid TỪ URL TIKI
# ──────────────────────────────────────────────
def parse_tiki_url(url: str) -> tuple[str | None, str | None]:
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)

        # product_id: lấy từ phần cuối path  …-p{digits}.html
        pid_match = re.search(r'-p(\d+)\.html', parsed.path)
        product_id = pid_match.group(1) if pid_match else None

        # spid: có thể trong query string
        spid = qs.get('spid', [None])[0]
        if spid is None:
            # thử lấy từ fragment hoặc path (một số URL không có spid)
            spid_match = re.search(r'spid[=_](\d+)', url)
            spid = spid_match.group(1) if spid_match else product_id  # fallback = product_id+1 thường không đúng

        return product_id, spid
    except Exception:
        return None, None

# ──────────────────────────────────────────────
# CRAWLER TIKI  (từ API.ipynb)
# ──────────────────────────────────────────────
TIKI_API = "https://tiki.vn/api/v2/reviews"

def crawl_tiki(product_id: str, spid: str, max_pages: int = 5, limit: int = 20) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/134.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": f"https://tiki.vn/p{product_id}.html",
    }
    all_rows = []
    progress = st.progress(0, text="Đang crawl bình luận...")

    for page in range(1, max_pages + 1):
        params = {
            "product_id": product_id,
            "spid": spid,
            "limit": limit,
            "page": page,
            "sort": "id|desc",
            "include": "comments,contribute_info",
        }
        try:
            resp = requests.get(TIKI_API, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                st.warning(f"Trang {page}: HTTP {resp.status_code}")
                break
            data = resp.json()
            reviews = data.get("data", [])
            if not reviews:
                break

            for rv in reviews:
                content = str(rv.get("content") or "").strip()
                title   = str(rv.get("title")   or "").strip()
                comment = content if content else title
                if comment:
                    all_rows.append({
                        "review_id": rv.get("id"),
                        "comment":   comment,
                        "title":     title,
                        "rating":    rv.get("rating"),
                        "author":    (rv.get("created_by") or {}).get("full_name", ""),
                        "created_at": rv.get("created_at", ""),
                        "product_id": product_id,
                        "spid": spid,
                    })

            paging = data.get("paging", {})
            last_p = paging.get("last_page", page)
            progress.progress(
                min(page / min(max_pages, max(last_p, 1)), 1.0),
                text=f"Trang {page}/{min(max_pages, last_p)} – đã lấy {len(all_rows)} bình luận"
            )

            if page >= last_p:
                break
            time.sleep(random.uniform(1.2, 2.5))

        except Exception as e:
            st.error(f"Lỗi ở trang {page}: {e}")
            break

    progress.empty()
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

# ──────────────────────────────────────────────
# LOAD / PHÂN TÍCH CSV DỰ PHÒNG
# ──────────────────────────────────────────────
def load_csv(uploaded_or_path) -> pd.DataFrame:
    """Đọc CSV từ file upload hoặc path, chuẩn hóa cột comment."""
    if isinstance(uploaded_or_path, (str, Path)):
        df = pd.read_csv(uploaded_or_path)
    else:
        df = pd.read_csv(uploaded_or_path)

    # Nhận diện cột comment
    for col in ['comment', 'content', 'review', 'text', 'noidung']:
        if col in df.columns:
            df = df.rename(columns={col: 'comment'})
            break
    return df

# ──────────────────────────────────────────────
# HELPER: badge màu
# ──────────────────────────────────────────────
def label_badge(label: str) -> str:
    css = {'POS': 'badge-pos', 'NEU': 'badge-neu', 'NEG': 'badge-neg'}.get(label, '')
    text = {'POS': 'Tích cực', 'NEU': 'Trung tính', 'NEG': 'Tiêu cực'}.get(label, label)
    return f'<span class="{css}">{text}</span>'

def stars(n: int) -> str:
    return '⭐' * n + '☆' * (5 - n)

# ──────────────────────────────────────────────
# RENDER CHUNG: biểu đồ + ưu tiên + bảng chi tiết
# (phải đặt TRƯỚC st.tabs() để tránh NameError)
# ──────────────────────────────────────────────
def _render_analysis(df_pred: pd.DataFrame, label: str = "") -> None:
    # ── Biểu đồ cảm xúc ────────────────────────
    col_bar, col_pie = st.columns(2)
    col_bar.plotly_chart(draw_sentiment_chart(df_pred), use_container_width=True)
    col_pie.plotly_chart(draw_donut_chart(df_pred),     use_container_width=True)

    # ── Kết luận tự động ───────────────────────
    st.markdown("### 📝 Kết luận tự động")
    st.markdown(auto_conclusion(df_pred))

    st.markdown('<hr class="light"/>', unsafe_allow_html=True)

    # ── Bảng cảnh báo ưu tiên ──────────────────
    render_priority_table(df_pred)

    st.markdown('<hr class="light"/>', unsafe_allow_html=True)

    # ── Bảng chi tiết đầy đủ ───────────────────
    st.markdown("### 📋 Chi tiết tất cả bình luận")
    badge_map = {'CAO': '🔴 CAO', 'TRUNG BÌNH': '🟡 TRUNG BÌNH', 'THẤP': '🟢 THẤP'}
    disp_df = df_pred[['original', 'label', 'rating', 'priority_level', 'reply']].copy()
    disp_df['priority_level'] = disp_df['priority_level'].map(badge_map)
    disp_df.columns = ['Bình luận', 'Cảm xúc', 'Rating', 'Ưu tiên', 'Trả lời gợi ý']
    st.dataframe(disp_df, use_container_width=True, height=380)

    csv_out = df_pred.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        "⬇️ Tải kết quả phân tích đầy đủ (CSV)",
        csv_out,
        file_name=f"sentiment_{label or 'result'}.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════
# GIAO DIỆN CHÍNH
# ══════════════════════════════════════════════
st.markdown("## 🛒 Tiki Sentiment Analyzer")
st.markdown("Phân tích cảm xúc bình luận sản phẩm Tiki · TF-IDF + SVM")

if not models_ready():
    st.error(
        "⚠️ Không tìm thấy model trong thư mục `artifacts/`. "
        "Hãy chạy notebook huấn luyện trước để sinh ra `label_model.joblib` và `rating_model.joblib`."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "💬  Phân tích bình luận đơn",
    "🔗  Crawl từ link Tiki",
    "📁  Upload CSV dự phòng",
])

# ══════════════════════════════════════════════
# TAB 1: PHÂN TÍCH BÌNH LUẬN ĐƠN
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Nhập bình luận cần phân tích</p>', unsafe_allow_html=True)

    col_inp, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_inp:
        user_comment = st.text_area(
            label="Bình luận",
            placeholder='Ví dụ: "Sản phẩm đẹp, giao hàng nhanh, đóng gói cẩn thận! ❤️"',
            height=100,
            label_visibility="collapsed",
        )
    with col_btn:
        run_single = st.button("Phân tích ▶", use_container_width=True, type="primary")

    # Ví dụ nhanh
    st.caption("Thử ngay:")
    ex_cols = st.columns(3)
    examples = [
        "Sản phẩm đẹp, giao hàng nhanh! ❤️",
        "Hàng bình thường, không có gì đặc biệt.",
        "Hàng lỗi, mới dùng đã hỏng, rất thất vọng 👎",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex[:35] + "…", key=f"ex_{i}"):
            user_comment = ex
            run_single   = True

    if run_single and user_comment.strip():
        with st.spinner("Đang phân tích..."):
            pred  = predict_single(user_comment)
            reply = generate_reply(pred)

        st.markdown('<hr class="light"/>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Cảm xúc",    pred['label_name'])
        r2.metric("Rating dự đoán", f"{pred['rating']} / 5")
        r3.metric("Độ tin cậy", f"{pred['confidence']:.3f}")

        st.markdown('<hr class="light"/>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🧹 Văn bản sau tiền xử lý:**")
            st.code(pred['clean'], language=None)
        with c2:
            st.markdown("**⭐ Rating:**")
            st.markdown(
                f'<p class="stars">{stars(pred["rating"])}</p> {RATING_NAME[pred["rating"]]}',
                unsafe_allow_html=True
            )
            st.markdown(f"**Nhãn:** {label_badge(pred['label'])}", unsafe_allow_html=True)

        st.markdown("**🤖 Chatbot trả lời tự động:**")
        st.markdown(f'<div class="chat-bubble">{reply}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="light"/>', unsafe_allow_html=True)
        st.markdown("**🚨 Mức độ ưu tiên xử lý:**")
        render_priority_alert(pred)

# ══════════════════════════════════════════════
# TAB 2: CRAWL TỪ LINK TIKI
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Nhập link sản phẩm Tiki</p>', unsafe_allow_html=True)

    tiki_url = st.text_input(
        "Link Tiki",
        placeholder="https://tiki.vn/ten-san-pham-p123456.html?spid=123457",
        label_visibility="collapsed",
    )

    col_a, col_b, col_c = st.columns(3)
    max_pages = col_a.slider("Số trang tối đa", 1, 20, 5)
    limit     = col_b.selectbox("Bình luận / trang", [10, 20, 40], index=1)
    run_crawl = col_c.button("🕷️ Crawl & Phân tích", type="primary", use_container_width=True)

    if run_crawl and tiki_url.strip():
        product_id, spid = parse_tiki_url(tiki_url.strip())

        if not product_id:
            st.error("Không tách được product_id từ link. Hãy kiểm tra lại URL.")
        else:
            st.info(f"**product_id:** `{product_id}`  |  **spid:** `{spid}`")
            df_raw = crawl_tiki(product_id, spid or product_id, max_pages, limit)

            if df_raw.empty:
                st.warning(
                    "Không lấy được bình luận. Tiki có thể đang chặn request hoặc sản phẩm chưa có đánh giá.\n\n"
                    "👉 Hãy chuyển sang tab **Upload CSV dự phòng** để demo."
                )
            else:
                st.success(f"Crawl thành công **{len(df_raw)}** bình luận.")

                # Lưu CSV để download
                csv_bytes = df_raw.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("⬇️ Tải CSV bình luận thô", csv_bytes,
                                   file_name=f"tiki_{product_id}_raw.csv", mime="text/csv")

                with st.spinner("Đang phân tích cảm xúc..."):
                    df_pred = analyze_comments_df(df_raw['comment'].tolist())

                if df_pred.empty:
                    st.warning("Không có bình luận hợp lệ để phân tích.")
                else:
                    _render_analysis(df_pred, product_id)

# ══════════════════════════════════════════════
# TAB 3: UPLOAD CSV DỰ PHÒNG
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Upload file CSV bình luận dự phòng</p>', unsafe_allow_html=True)
    st.caption("Dùng khi Tiki chặn crawler hoặc không có mạng lúc demo.")

    uploaded = st.file_uploader(
        "Chọn file CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    use_backup = st.checkbox(
        "Dùng file dự phòng mặc định `tiki_reviews_crawled.csv`",
        value=False
    )

    run_csv = st.button("📊 Phân tích CSV", type="primary")

    if run_csv:
        source = None
        if uploaded is not None:
            source = uploaded
        elif use_backup:
            backup_path = Path("tiki_reviews_crawled.csv")
            if backup_path.exists():
                source = backup_path
            else:
                st.error("Không tìm thấy `tiki_reviews_crawled.csv`. Hãy upload file thủ công.")

        if source is not None:
            try:
                df_csv = load_csv(source)
                if 'comment' not in df_csv.columns:
                    st.error(f"Không tìm thấy cột comment. Các cột hiện có: {list(df_csv.columns)}")
                else:
                    st.success(f"Đọc được **{len(df_csv)}** dòng từ CSV.")
                    st.dataframe(df_csv.head(5), use_container_width=True)

                    with st.spinner("Đang phân tích cảm xúc..."):
                        df_pred = analyze_comments_df(df_csv['comment'].tolist())

                    if df_pred.empty:
                        st.warning("Không có bình luận hợp lệ.")
                    else:
                        _render_analysis(df_pred, label="csv_backup")
            except Exception as e:
                st.error(f"Lỗi đọc CSV: {e}")

# ──────────────────────────────────────────────
# Fix: Tab2 render (phải khai báo _render_analysis trước khi dùng trong tab2)
# Do Streamlit chạy top-to-bottom, tab2 body đã chạy khi _render_analysis chưa có.
# Cách khắc phục: đặt _render_analysis trước khi mở tabs. ← đã làm ở trên.
# Tuy nhiên, vì Python scope global, hàm đã được define trước nên hoạt động đúng.
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown('<hr class="light"/>', unsafe_allow_html=True)
st.caption(
    "🛠️ Mô hình: **TF-IDF (1-3 gram) + LinearSVC** · "
    "Nhãn: **POS / NEU / NEG** · "
    "Rating: **1-5 sao** · "
    "Dữ liệu: Bình luận sản phẩm Tiki"
)
