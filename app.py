from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import CLASS_NAMES, JedgeNetPredictor, load_rgb_image


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "weights" / "jedgenet_4class_seed5.pth"
SAMPLE_DIR = ROOT / "assets" / "samples"
SAMPLES = {
    "Cracked": SAMPLE_DIR / "cracked.png",
    "Dry": SAMPLE_DIR / "dry.png",
    "Insect damaged": SAMPLE_DIR / "insect_damaged.png",
    "Invalid": SAMPLE_DIR / "invalid.png",
}

PAPER_MACRO_F1 = "95.21%"
STM32_LATENCY = "300.6 ms"


st.set_page_config(
    page_title="JedgeNet | Jujube defect classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #19211e;
        --muted: #607069;
        --line: #d7dfda;
        --paper: #f4f7f4;
        --surface: #ffffff;
        --accent: #b43a32;
        --teal: #236c62;
      }

      [data-testid="stAppViewContainer"] { background: var(--paper); }
      [data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
      [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
      }
      h1, h2, h3, p, label, button { letter-spacing: 0 !important; }

      .brand-lockup {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 20px;
      }
      .brand-lockup > div:last-child { min-width: 0; }
      .brand-mark {
        width: 46px;
        height: 46px;
        display: grid;
        place-items: center;
        background: var(--accent);
        color: white;
        border-radius: 6px;
        font: 700 23px/1 sans-serif;
      }
      .brand-name {
        color: var(--ink);
        font: 700 32px/1.05 sans-serif;
        margin: 0;
      }
      .brand-subtitle {
        color: var(--muted);
        font: 400 14px/1.45 sans-serif;
        margin-top: 4px;
        white-space: normal;
        overflow-wrap: anywhere;
      }
      .mobile-subtitle { display: none; }

      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
      }
      .metric-item {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 13px 15px;
        min-height: 92px;
      }
      .metric-label {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.25;
      }
      .metric-value {
        color: var(--ink);
        font-size: 29px;
        line-height: 1.15;
        margin-top: 12px;
        white-space: nowrap;
      }

      div[data-testid="stSegmentedControl"] { margin-top: 8px; }
      div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] {
        background: var(--surface);
        border-radius: 7px;
      }
      .stButton > button {
        border-radius: 6px;
        min-height: 40px;
        font-weight: 650;
      }
      .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
      }
      [data-testid="stImage"] img { border-radius: 6px; }

      .section-label {
        color: var(--teal);
        font: 700 12px/1 sans-serif;
        text-transform: uppercase;
        margin-top: 26px;
        margin-bottom: 9px;
      }
      .prediction-banner {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        border-radius: 7px;
        padding: 19px 20px;
        margin-bottom: 14px;
      }
      .prediction-kicker {
        color: var(--muted);
        font: 700 11px/1 sans-serif;
        text-transform: uppercase;
      }
      .prediction-label {
        color: var(--ink);
        font: 750 28px/1.1 sans-serif;
        margin-top: 8px;
      }
      .prediction-score {
        color: var(--accent);
        font: 700 15px/1.2 sans-serif;
        margin-top: 7px;
      }
      .score-row {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        color: var(--ink);
        font-size: 14px;
        margin: 8px 0 4px;
      }
      .runtime-note {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
        border-top: 1px solid var(--line);
        margin-top: 18px;
        padding-top: 12px;
      }
      @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; }
        .brand-name { font-size: 27px; }
        .brand-subtitle { font-size: 13px; }
        .desktop-subtitle { display: none; }
        .mobile-subtitle { display: inline; }
        .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .metric-item { min-height: 82px; padding: 11px 12px; }
        .metric-value { font-size: 23px; margin-top: 9px; }
        .prediction-label { font-size: 24px; }
        div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          width: 100%;
        }
        div[data-testid="stSegmentedControl"] button {
          min-width: 0 !important;
          padding-left: 8px !important;
          padding-right: 8px !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_predictor(checkpoint_path: str) -> JedgeNetPredictor:
    return JedgeNetPredictor(checkpoint_path)


def render_score_breakdown(scores: tuple[float, ...]) -> None:
    st.markdown("#### Class scores")
    for class_name, score in sorted(
        zip(CLASS_NAMES, scores), key=lambda item: item[1], reverse=True
    ):
        st.markdown(
            f'<div class="score-row"><span>{html.escape(class_name)}</span>'
            f"<strong>{score * 100:.1f}%</strong></div>",
            unsafe_allow_html=True,
        )
        st.progress(score)


st.markdown(
    """
    <div class="brand-lockup">
      <div class="brand-mark">J</div>
      <div>
        <div class="brand-name">JedgeNet</div>
        <div class="brand-subtitle">
          <span class="desktop-subtitle">Jujube surface defect classification · four classes · 64 × 64 RGB</span>
          <span class="mobile-subtitle">Four-class inspection · 64 × 64 RGB</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metrics-grid">
      <div class="metric-item"><div class="metric-label">Model</div><div class="metric-value">JedgeNet</div></div>
      <div class="metric-item"><div class="metric-label">Classes</div><div class="metric-value">4</div></div>
      <div class="metric-item"><div class="metric-label">Test macro-F1</div><div class="metric-value">{PAPER_MACRO_F1}</div></div>
      <div class="metric-item"><div class="metric-label">STM32H750 latency</div><div class="metric-value">{STM32_LATENCY}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
input_mode = st.segmented_control(
    "Input source",
    options=("Upload", "Camera"),
    default="Upload",
    label_visibility="collapsed",
    width="stretch",
)

image = None
source_label = None

if input_mode == "Upload":
    uploaded = st.file_uploader(
        "Upload a jujube image",
        type=("jpg", "jpeg", "png", "webp", "bmp"),
        max_upload_size=15,
    )
    if uploaded is not None:
        try:
            image = load_rgb_image(uploaded)
            source_label = uploaded.name
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("The uploaded file could not be decoded as an image.", icon=":material/error:")

    st.markdown('<div class="section-label">Try a sample</div>', unsafe_allow_html=True)
    sample_cols = st.columns(4)
    for column, (class_name, sample_path) in zip(sample_cols, SAMPLES.items()):
        with column:
            st.image(str(sample_path), width="stretch")
            if st.button(
                class_name,
                key=f"sample-{class_name}",
                width="stretch",
            ):
                st.session_state["selected_sample"] = str(sample_path)

    if image is None:
        selected_sample = st.session_state.get("selected_sample")
        if selected_sample:
            selected_path = Path(selected_sample)
            if selected_path.is_file():
                image = load_rgb_image(selected_path)
                source_label = selected_path.stem.replace("_", " ").title()

elif input_mode == "Camera":
    captured = st.camera_input("Capture a jujube image")
    if captured is not None:
        try:
            image = load_rgb_image(captured)
            source_label = "Camera capture"
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("The camera image could not be decoded.", icon=":material/error:")

if image is None:
    st.info("Choose a sample, upload an image, or use the camera to run JedgeNet.", icon=":material/image_search:")
else:
    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
    try:
        with st.spinner("Running JedgeNet..."):
            predictor = get_predictor(str(CHECKPOINT_PATH))
            prediction = predictor.predict(image)
    except Exception as exc:
        st.error(f"Model inference failed: {exc}", icon=":material/error:")
        st.stop()

    image_col, result_col = st.columns((1.05, 0.95), gap="large")
    with image_col:
        st.image(image, caption=source_label, width="stretch")

    with result_col:
        st.markdown(
            f"""
            <div class="prediction-banner">
              <div class="prediction-kicker">Predicted class</div>
              <div class="prediction-label">{html.escape(prediction.class_name)}</div>
              <div class="prediction-score">Model score {prediction.scores[prediction.class_index] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_score_breakdown(prediction.scores)
        st.markdown(
            f"""
            <div class="runtime-note">
              Host CPU forward pass: <strong>{prediction.inference_ms:.1f} ms</strong>.<br>
              The paper's {STM32_LATENCY} result uses the fused INT8 graph on STM32H750 and is not the live Streamlit runtime.
            </div>
            """,
            unsafe_allow_html=True,
        )
