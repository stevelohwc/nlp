"""Frontend/page rendering for the news-classification Streamlit app."""

from __future__ import annotations

from html import escape

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app_core import (
    LOGGER,
    MALAYSIA_SAMPLE_URLS,
    benchmark_evaluation,
    extract_text_from_url,
    is_valid_http_url,
    load_holdout_features,
    load_model_bundle,
    looks_like_section_or_home_url,
    predict_article,
    read_samples,
    validate_extracted_article_text,
    validate_text_input,
)

_STRATEGY_LABELS = ["Trafilatura (precise)", "BeautifulSoup (fallback)"]
_STRATEGY_VALUES = ["trafilatura", "beautifulsoup"]


def render_open_link(label: str, url: str, disabled: bool = False) -> None:
    safe_label = escape(label)
    if disabled:
        st.markdown(
            f'<div class="open-link-wrap"><span class="open-link-btn disabled">{safe_label}</span></div>',
            unsafe_allow_html=True,
        )
        return

    safe_url = escape(url, quote=True)
    st.markdown(
        (
            '<div class="open-link-wrap">'
            f'<a class="open-link-btn" href="{safe_url}" target="_blank" '
            f'rel="noopener noreferrer">{safe_label}</a>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_prediction_card(result: dict) -> None:
    label = result["label"]
    confidence_pct = result["confidence"] * 100
    badge_class = "news-real" if "real" in label.lower() else "news-fake"

    st.markdown(
        f"""
        <div class="result-card {badge_class}">
            <div class="result-label">{label}</div>
            <div class="result-confidence">Confidence: {confidence_pct:.2f}%</div>
            <div class="conf-bar-track">
                <div class="conf-bar {badge_class}" style="width:{confidence_pct:.1f}%"></div>
            </div>
            <div class="result-note">Model prediction only. Treat this as a decision-support signal, not verified fact-checking.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

        :root {
            --ink: #102028;
            --sky: #e8f1f8;
            --sand: #f7f3ea;
            --accent: #0b7a75;
            --warn: #af3b2f;
        }

        .stApp {
            font-family: 'Source Sans 3', sans-serif;
            background: #f8fafc;
            color: var(--ink);
        }

        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: 2.2rem;
            line-height: 1.1;
            margin-bottom: 0.3rem;
            color: #17303a;
            letter-spacing: 0.01em;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.88;
            margin-bottom: 1.2rem;
        }

        .stTabs [role="tab"] {
            color: #41545d !important;
            font-weight: 700;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            color: #0b7a75 !important;
        }

        .stTextInput label,
        .stTextArea label,
        .stSelectbox label {
            color: #334a55 !important;
            font-weight: 700 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(19, 43, 58, 0.20) !important;
            border-radius: 12px !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: #17303a !important;
            -webkit-text-fill-color: #17303a !important;
        }

        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {
            color: #6a7b86 !important;
            -webkit-text-fill-color: #6a7b86 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(19, 43, 58, 0.20) !important;
            border-radius: 12px !important;
            color: #17303a !important;
        }

        div[data-baseweb="select"] * {
            color: #17303a !important;
            -webkit-text-fill-color: #17303a !important;
        }

        div[data-testid="stExpander"] details > summary {
            background: rgba(255, 255, 255, 0.92) !important;
            color: #17303a !important;
            border: 1px solid rgba(19, 43, 58, 0.20) !important;
            border-radius: 12px !important;
        }

        div[data-testid="stExpander"] details[open] > summary {
            border-bottom-left-radius: 0 !important;
            border-bottom-right-radius: 0 !important;
        }

        div[data-testid="stExpander"] details > div {
            background: rgba(255, 255, 255, 0.82) !important;
            border: 1px solid rgba(19, 43, 58, 0.12) !important;
            border-top: none !important;
            border-bottom-left-radius: 12px !important;
            border-bottom-right-radius: 12px !important;
            padding-top: 0.65rem !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #162841 0%, #1e3454 100%);
            color: #f6f9ff !important;
            border: 1px solid rgba(200, 216, 236, 0.30);
            border-radius: 12px;
            font-weight: 700;
            min-height: 2.85rem;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 10px rgba(19, 31, 49, 0.18);
        }

        .stButton > button:hover {
            color: #ffffff !important;
            border-color: rgba(224, 238, 255, 0.55);
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(19, 31, 49, 0.24);
        }

        .stButton > button:focus {
            color: #ffffff !important;
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(11, 122, 117, 0.35);
        }

        .stButton > button:disabled {
            background: #b7c0c7;
            color: #36464f !important;
            border-color: #9ca8b0;
            box-shadow: none;
            cursor: not-allowed;
        }

        .open-link-wrap {
            display: flex;
            align-items: stretch;
            margin: 0;
            padding: 0;
        }

        .open-link-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.85rem;
            text-decoration: none !important;
            background: rgba(255, 255, 255, 0.90);
            color: #17303a !important;
            border: 1px solid rgba(19, 43, 58, 0.22);
            border-radius: 12px;
            font-weight: 700;
            line-height: 1.15;
            padding: 0.5rem 1rem;
            transition: all 0.18s ease;
            box-shadow: 0 2px 8px rgba(19, 43, 58, 0.08);
        }

        .open-link-btn:hover {
            color: #102028 !important;
            border-color: rgba(19, 43, 58, 0.38);
            transform: translateY(-1px);
            box-shadow: 0 6px 12px rgba(19, 43, 58, 0.13);
        }

        .open-link-btn.disabled {
            color: #7b8b95 !important;
            border-color: rgba(122, 141, 154, 0.28);
            background: rgba(255, 255, 255, 0.72);
            box-shadow: none;
            pointer-events: none;
            user-select: none;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(19, 43, 58, 0.12);
            border-radius: 12px;
            padding: 0.65rem 0.9rem;
            box-shadow: 0 4px 12px rgba(19, 43, 58, 0.08);
        }

        div[data-testid="stMetric"] label {
            color: #334a55 !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #132b39 !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em;
        }

        div[data-testid="stMetricDelta"] {
            color: #3f5763 !important;
        }

        .result-card {
            border-radius: 14px;
            padding: 16px 18px;
            margin-top: 12px;
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 8px 22px rgba(18, 41, 52, 0.12);
            background: #ffffff;
        }

        .result-card.news-real {
            border-left: 8px solid var(--accent);
            background: linear-gradient(120deg, #ffffff 0%, #e9f8f6 100%);
        }

        .result-card.news-fake {
            border-left: 8px solid var(--warn);
            background: linear-gradient(120deg, #ffffff 0%, #fdece8 100%);
        }

        .result-label {
            font-family: 'Fraunces', serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .result-confidence {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .conf-bar-track {
            background: #e2e8f0;
            border-radius: 3px;
            height: 6px;
            margin-bottom: 0.6rem;
            overflow: hidden;
        }

        .conf-bar {
            height: 6px;
            border-radius: 3px;
        }

        .conf-bar.news-real {
            background: var(--accent);
        }

        .conf-bar.news-fake {
            background: var(--warn);
        }

        .result-note {
            font-size: 0.9rem;
            opacity: 0.82;
        }

        .model-card {
            background: rgba(11, 122, 117, 0.08);
            border: 1px solid rgba(11, 122, 117, 0.28);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 4px;
        }

        .model-card-name {
            font-weight: 700;
            font-size: 0.95rem;
            color: #0b7a75;
            word-break: break-all;
            font-family: 'Source Sans 3', monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle: dict) -> tuple[float, int, int, str]:
    """Render sidebar and return (threshold, min_chars, max_chars, url_strategy)."""
    with st.sidebar:
        # Section A: Model
        st.markdown("#### Model")
        model_name = bundle.get("model_name", "unknown")
        st.markdown(
            f'<div class="model-card"><div class="model-card-name">{escape(model_name)}</div></div>',
            unsafe_allow_html=True,
        )
        metrics = bundle.get("metrics", {})
        if metrics:
            metrics_lower = {k.lower(): v for k, v in metrics.items()}
            accuracy = metrics_lower.get("accuracy")
            f1 = metrics_lower.get("f1")
            if accuracy is not None or f1 is not None:
                m_col1, m_col2 = st.columns(2)
                if accuracy is not None:
                    m_col1.metric("Accuracy", f"{float(accuracy):.3f}")
                if f1 is not None:
                    m_col2.metric("F1", f"{float(f1):.3f}")

        st.markdown("---")

        # Section B: Prediction Settings
        st.markdown("#### Prediction Settings")
        threshold = st.slider(
            "Real-news threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
        )
        st.caption("Scores below threshold → Fake News.")
        min_chars = st.number_input(
            "Min chars",
            min_value=10,
            max_value=500,
            value=30,
            step=10,
            key="sidebar_min_chars",
        )
        max_chars = st.number_input(
            "Max chars",
            min_value=1000,
            max_value=100000,
            value=20000,
            step=1000,
            key="sidebar_max_chars",
        )

        st.markdown("---")

        # Section C: Live URL Settings
        st.markdown("#### Live URL Settings")
        st.caption("Controls how text is extracted in the Live URL tab.")
        strategy_choice = st.radio("Extraction strategy", _STRATEGY_LABELS, index=0)
        url_strategy = _STRATEGY_VALUES[_STRATEGY_LABELS.index(strategy_choice)]

        st.markdown("---")

        # Section D: Environment
        st.markdown("#### Environment")
        lib_versions = bundle.get("library_versions", {})
        if lib_versions:
            lib_df = pd.DataFrame(lib_versions.items(), columns=["Package", "Version"])
            st.dataframe(lib_df, hide_index=True, use_container_width=True)

    return float(threshold), int(min_chars), int(max_chars), url_strategy


def render_benchmark_tab(bundle: dict) -> None:
    st.caption("Runs on the labeled holdout split produced by run_preprocessing.py")
    if st.button("Run Holdout Benchmark", type="primary"):
        try:
            X_test, y_test = load_holdout_features()
            scores = benchmark_evaluation(bundle["model"], X_test, y_test)
        except Exception as exc:
            LOGGER.exception("Benchmark evaluation failed")
            st.error(f"Benchmark failed: {exc}")
            return

        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{scores['accuracy']:.4f}")
        cols[1].metric("Precision", f"{scores['precision']:.4f}")
        cols[2].metric("Recall", f"{scores['recall']:.4f}")
        cols[3].metric("F1", f"{scores['f1']:.4f}")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            scores["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Holdout Confusion Matrix")
        st.pyplot(fig)


def render_single_prediction_tab(
    bundle: dict, threshold: float, min_chars: int, max_chars: int, samples: dict[str, str]
) -> None:
    col_a, col_b, col_c = st.columns([1, 1, 1])
    if col_a.button("Load Real Example"):
        st.session_state.single_input = samples["real"]
    if col_b.button("Load Fake Example"):
        st.session_state.single_input = samples["fake"]
    if col_c.button("Clear"):
        st.session_state.single_input = ""

    text = st.text_area(
        "Article Text",
        key="single_input",
        height=240,
        placeholder="Paste a full article or long excerpt for better reliability...",
    )

    if st.button("Analyze Article", type="primary"):
        ok, message = validate_text_input(text, min_chars, max_chars)
        if not ok:
            st.warning(message)
            return

        with st.spinner("Running prediction..."):
            try:
                result = predict_article(bundle, text, threshold)
                render_prediction_card(result)
            except Exception as exc:
                LOGGER.exception("Single prediction failed")
                st.error(f"Prediction failed: {exc}")


def render_live_url_tab(
    bundle: dict,
    threshold: float,
    url_strategy: str,
    min_chars: int,
    max_chars: int,
) -> None:
    st.caption("Live URL mode performs model inference only. It does not verify ground-truth claims.")
    sample_url_map = dict(MALAYSIA_SAMPLE_URLS)
    with st.expander("Use Sample Malaysian News URLs", expanded=True):
        selected_label = st.selectbox(
            "Sample URL",
            options=list(sample_url_map.keys()),
            key="live_url_sample_select",
        )
        selected_url = sample_url_map[selected_label]

        sample_col_1, sample_col_2, _sample_spacer = st.columns([1, 1, 2.2])
        if sample_col_1.button("Load Sample URL"):
            st.session_state.live_url_input = selected_url
        with sample_col_2:
            render_open_link("Open selected source", selected_url)
        st.caption(
            "Use direct article URLs for prediction. Section/home pages are blocked because they extract mixed headlines."
        )

    url = st.text_input(
        "News URL",
        key="live_url_input",
        placeholder="https://example.com/news/article",
    )

    action_col_1, action_col_2, _action_spacer = st.columns([1, 1, 2.2])
    with action_col_1:
        analyze_clicked = st.button("Fetch and Analyze URL", type="primary")

    extracted_text = ""
    with action_col_2:
        if is_valid_http_url(url):
            render_open_link("Open current URL", url)
        else:
            render_open_link("Open current URL", "#", disabled=True)

    if analyze_clicked:
        if not is_valid_http_url(url):
            st.warning("Please enter a valid http/https URL.")
            return

        section_like, section_message = looks_like_section_or_home_url(url)
        if section_like:
            st.warning(
                section_message + " Please open one specific article page and paste that URL instead."
            )
            return

        with st.spinner("Fetching and extracting article text..."):
            try:
                extracted_text = extract_text_from_url(url, url_strategy)
            except Exception as exc:
                LOGGER.exception("URL extraction failed")
                st.error(f"URL fetch/extraction failed: {exc}")

        if extracted_text:
            ok_text, msg = validate_extracted_article_text(extracted_text)
            if not ok_text:
                st.warning(msg)
                st.info("Prediction skipped. Paste a direct article URL or use manual text below.")
                st.text_area("Extracted Preview", value=extracted_text[:3000], height=220)
                return

            st.success("Article text extracted successfully.")
            st.text_area("Extracted Preview", value=extracted_text[:3000], height=220)
            try:
                result = predict_article(bundle, extracted_text, threshold)
                render_prediction_card(result)
            except Exception as exc:
                LOGGER.exception("URL prediction failed")
                st.error(f"Prediction failed: {exc}")
        else:
            st.warning("No article body could be extracted from this page. Try a direct article URL.")


def render_frontend_page() -> None:
    st.set_page_config(
        page_title="News Authenticity Classifier",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_css()

    st.markdown('<div class="hero-title">News Authenticity Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Standalone fake-news project with robust validation workflows and artifact checks.</div>',
        unsafe_allow_html=True,
    )

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        LOGGER.exception("Model bundle load failed")
        st.error(f"Startup check failed: {exc}")
        st.info(
            "Run: `python run_preprocessing.py`, `python train_baseline_models.py`, "
            "`python train_advanced_models.py` before starting the app."
        )
        st.stop()

    st.session_state.setdefault("single_input", "")
    st.session_state.setdefault("live_url_input", "")

    samples = read_samples()
    threshold, min_chars, max_chars, url_strategy = render_sidebar(bundle)

    tab_predict, tab_benchmark, tab_live = st.tabs(
        ["📝 Predict", "📊 Benchmark", "🔗 Live URL"]
    )

    with tab_predict:
        render_single_prediction_tab(bundle, threshold, min_chars, max_chars, samples)

    with tab_benchmark:
        render_benchmark_tab(bundle)

    with tab_live:
        render_live_url_tab(bundle, threshold, url_strategy, min_chars, max_chars)
