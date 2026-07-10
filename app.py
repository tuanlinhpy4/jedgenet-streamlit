from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import CLASS_NAMES, JedgeNetPredictor, load_rgb_image


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "weights" / "jedgenet_4class_seed5.pth"
TEST_DATA_DIR = ROOT / "assets" / "test_data"
TEST_CLASS_DIRS = {
    "Cracked": "cracked",
    "Dry": "dry",
    "Insect damaged": "insect_damaged",
    "Invalid": "invalid",
}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
      }
      h1, h2, h3, p, label, button { letter-spacing: 0 !important; }

      .brand-lockup {
        display: flex;
        align-items: center;
        gap: 11px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 8px;
      }
      .brand-lockup > div:last-child { min-width: 0; }
      .brand-mark {
        width: 40px;
        height: 40px;
        display: grid;
        place-items: center;
        background: var(--accent);
        color: white;
        border-radius: 6px;
        font: 700 20px/1 sans-serif;
      }
      .brand-name {
        color: var(--ink);
        font: 700 27px/1.05 sans-serif;
        margin: 0;
      }
      .brand-subtitle {
        color: var(--muted);
        font: 400 13px/1.35 sans-serif;
        margin-top: 2px;
        white-space: normal;
        overflow-wrap: anywhere;
      }
      .mobile-subtitle { display: none; }

      div[data-testid="stSegmentedControl"] { margin-top: 3px; }
      div[data-testid="stFileUploader"] {
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
      [data-testid="stImage"] img,
      [data-testid="stImageContainer"] img {
        width: 100%;
        height: 280px !important;
        max-height: 280px !important;
        object-fit: contain !important;
        background: #050706;
        border-radius: 6px;
      }
      [data-testid="stImage"] [data-testid="stImageCaption"] {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .section-label {
        color: var(--teal);
        font: 700 12px/1 sans-serif;
        text-transform: uppercase;
        margin-top: 12px;
        margin-bottom: 5px;
      }
      .prediction-banner {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        border-radius: 7px;
        padding: 12px 15px;
        margin-bottom: 7px;
      }
      .prediction-kicker {
        color: var(--muted);
        font: 700 11px/1 sans-serif;
        text-transform: uppercase;
      }
      .prediction-label {
        color: var(--ink);
        font: 750 24px/1.1 sans-serif;
        margin-top: 5px;
      }
      .prediction-score {
        color: var(--accent);
        font: 700 15px/1.2 sans-serif;
        margin-top: 4px;
      }
      .score-row {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        color: var(--ink);
        font-size: 14px;
        margin-bottom: 5px;
      }
      .confidence-heading {
        color: var(--ink);
        font: 700 20px/1.2 sans-serif;
        margin: 14px 0 10px;
      }
      .confidence-item { margin-top: 10px; }
      .confidence-track {
        width: 100%;
        height: 7px;
        overflow: hidden;
        background: #e4e9e6;
        border-radius: 4px;
      }
      .confidence-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 4px;
      }
      .truth-banner {
        border-radius: 6px;
        padding: 9px 12px;
        font-size: 14px;
        line-height: 1.35;
      }
      .truth-correct { color: #146c37; background: #dff3e6; }
      .truth-incorrect { color: #8a5700; background: #fff0cf; }
      @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] { padding-top: 0.7rem; }
        .brand-name { font-size: 25px; }
        .brand-subtitle { font-size: 13px; }
        .desktop-subtitle { display: none; }
        .mobile-subtitle { display: inline; }
        .prediction-label { font-size: 24px; }
        [data-testid="stImage"] img,
        [data-testid="stImageContainer"] img {
          height: 240px !important;
          max-height: 240px !important;
        }
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


@st.cache_data(show_spinner=False)
def get_test_images(class_directory: str) -> tuple[str, ...]:
    directory = TEST_DATA_DIR / class_directory
    if not directory.is_dir():
        return ()
    return tuple(
        path.name
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def render_score_breakdown(scores: tuple[float, ...]) -> None:
    rows = []
    for class_name, score in sorted(
        zip(CLASS_NAMES, scores), key=lambda item: item[1], reverse=True
    ):
        percentage = score * 100
        rows.append(
            '<div class="confidence-item">'
            f'<div class="score-row"><span>{html.escape(class_name)}</span>'
            f"<strong>{percentage:.1f}%</strong></div>"
            '<div class="confidence-track">'
            f'<div class="confidence-fill" style="width:{percentage:.2f}%"></div>'
            "</div></div>"
        )
    st.markdown(
        '<div class="confidence-heading">Confidence scores</div>' + "".join(rows),
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="brand-lockup">
      <div class="brand-mark">J</div>
      <div>
        <div class="brand-name">JedgeNet</div>
        <div class="brand-subtitle">
          <span class="desktop-subtitle">Jujube surface defect classification &middot; four classes &middot; 64 &times; 64 RGB</span>
          <span class="mobile-subtitle">Four-class inspection &middot; 64 &times; 64 RGB</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

image = None
source_label = None
expected_class = None

controls_col, _ = st.columns((0.92, 1.08), gap="large")

with controls_col:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    input_mode = st.segmented_control(
        "Input source",
        options=("Built-in test data", "Upload your image"),
        default="Built-in test data",
        label_visibility="collapsed",
        width="stretch",
    )

    if input_mode == "Built-in test data":
        test_image_counts = {
            class_name: len(get_test_images(class_directory))
            for class_name, class_directory in TEST_CLASS_DIRS.items()
        }
        st.caption(
            f"Browse all {sum(test_image_counts.values())} images from the held-out test split."
        )
        class_select_col, image_select_col = st.columns(2, gap="small")
        with class_select_col:
            expected_class = st.selectbox(
                "Ground-truth class",
                options=tuple(TEST_CLASS_DIRS),
                format_func=lambda class_name: (
                    f"{class_name} ({test_image_counts[class_name]})"
                ),
            )

        class_directory = TEST_CLASS_DIRS[expected_class]
        test_images = get_test_images(class_directory)
        with image_select_col:
            selected_name = st.selectbox(
                "Test image",
                options=test_images,
                placeholder="No test images found",
            )

        if selected_name:
            selected_path = TEST_DATA_DIR / class_directory / selected_name
            try:
                image = load_rgb_image(selected_path)
                source_label = selected_name
            except (UnidentifiedImageError, OSError, ValueError):
                st.error(
                    "The selected test image could not be decoded.",
                    icon=":material/error:",
                )

    elif input_mode == "Upload your image":
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
                st.error(
                    "The uploaded file could not be decoded.",
                    icon=":material/error:",
                )

preview_col, result_col = st.columns((0.92, 1.08), gap="large")

with preview_col:
    st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
    if image is None:
        st.info("No image selected.", icon=":material/image:")
    else:
        st.image(image, caption=source_label, width="stretch")

with result_col:
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
    if image is None:
        st.info(
            "Select a test image or upload your own image to run JedgeNet.",
            icon=":material/image_search:",
        )
    else:
        try:
            with st.spinner("Running JedgeNet..."):
                predictor = get_predictor(str(CHECKPOINT_PATH))
                prediction = predictor.predict(image)
        except Exception as exc:
            st.error(f"Model inference failed: {exc}", icon=":material/error:")
            st.stop()

        st.markdown(
            f"""
            <div class="prediction-banner">
              <div class="prediction-kicker">Predicted class</div>
              <div class="prediction-label">{html.escape(prediction.class_name)}</div>
              <div class="prediction-score">Confidence {prediction.scores[prediction.class_index] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if expected_class is not None:
            if prediction.class_name == expected_class:
                truth_class = "truth-correct"
                truth_message = f"Ground truth: {expected_class} - correct prediction"
            else:
                truth_class = "truth-incorrect"
                truth_message = f"Ground truth: {expected_class} - incorrect prediction"
            st.markdown(
                f'<div class="truth-banner {truth_class}">'
                f"{html.escape(truth_message)}</div>",
                unsafe_allow_html=True,
            )
        render_score_breakdown(prediction.scores)
