from pathlib import Path

import streamlit as st
from src.predict import predict_admission, load_model_bundle

st.set_page_config(
    page_title="Neural Network Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_bundle():
    return load_model_bundle()


def load_logo():
    logo_path = Path("assets/algonquin_logo.png")
    if logo_path.exists():
        return str(logo_path)
    return None


def local_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        }

        .hero-card {
            background: linear-gradient(135deg, #c8102e 0%, #8b0d22 100%);
            padding: 2rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
            line-height: 1.6;
        }

        .info-card {
            background: white;
            padding: 1.25rem;
            border-radius: 20px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }

        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f6f9ff 100%);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            border-left: 6px solid #c8102e;
            margin-bottom: 0.8rem;
        }

        .metric-title {
            font-size: 0.9rem;
            color: #555;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #111;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1f2937;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .footer-card {
            background: #111827;
            color: white;
            padding: 1.2rem;
            border-radius: 18px;
            margin-top: 1.5rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
        }

        .prediction-good {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            padding: 1rem;
            border-radius: 16px;
            color: #166534;
            font-weight: 700;
            border: 1px solid #86efac;
        }

        .prediction-bad {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            padding: 1rem;
            border-radius: 16px;
            color: #991b1b;
            font-weight: 700;
            border: 1px solid #fca5a5;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #c8102e 0%, #8b0d22 100%);
            color: white;
            border: none;
        }

        .stButton > button:hover {
            filter: brightness(1.05);
            color: white;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle):
    st.sidebar.markdown("## Model Dashboard")
    st.sidebar.success("Model bundle loaded")

    st.sidebar.markdown(f"**Selected Model:** `{bundle['best_model_name']}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Performance")

    st.sidebar.metric("Accuracy", f"{bundle['metrics']['accuracy']:.4f}")
    st.sidebar.metric("Precision", f"{bundle['metrics']['precision']:.4f}")
    st.sidebar.metric("Recall", f"{bundle['metrics']['recall']:.4f}")
    st.sidebar.metric("F1-Score", f"{bundle['metrics']['f1_score']:.4f}")
    st.sidebar.metric("ROC-AUC", f"{bundle['metrics']['roc_auc']:.4f}")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app predicts whether a student is likely to belong to the high-admission category based on academic profile features."
    )


def render_header():
    logo = load_logo()

    left, right = st.columns([1, 4])

    with left:
        if logo:
            st.image(logo, width=150)

    with right:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">🎓 Neural Network Admission Predictor</div>
                <div class="hero-subtitle">
                    A modern machine learning application that predicts whether a student is likely
                    to fall into the high-admission category using a trained neural network model
                    built with scikit-learn for CST2216 – Modularizing and Deploying ML Code.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_input_section():
    st.markdown('<div class="section-title">Applicant Profile Input</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gre_score = st.slider("GRE Score", min_value=260, max_value=340, value=320)
        toefl_score = st.slider("TOEFL Score", min_value=0, max_value=120, value=105)
        university_rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2)
        research = st.selectbox(
            "Research Experience",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col2:
        sop = st.slider("Statement of Purpose (SOP)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
        lor = st.slider("Letter of Recommendation (LOR)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01)

    input_data = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research,
    }

    return input_data


def render_input_summary(input_data):
    st.markdown(
        """
        <div class="info-card">
            <div class="section-title">Input Summary</div>
        """,
        unsafe_allow_html=True,
    )
    st.write(input_data)
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction(result):
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    if result["prediction"] == 1:
        st.markdown(
            f'<div class="prediction-good">Prediction: {result["label"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="prediction-bad">Prediction: {result["label"]}</div>',
            unsafe_allow_html=True,
        )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Admission Probability</div>
                <div class="metric-value">{result["probability"] * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        confidence = max(result["probability"], 1 - result["probability"]) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Model Confidence</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-card">
            <div class="section-title">Interpretation</div>
        """,
        unsafe_allow_html=True,
    )

    if result["prediction"] == 1:
        st.write(
            "The model suggests that this applicant has a strong academic profile and is likely to be classified in the high-admission category."
        )
    else:
        st.write(
            "The model suggests that this applicant is less likely to be classified in the high-admission category based on the current academic profile."
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="footer-card">
            <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;">
                Academic Credit
            </div>
            <div>
                Developed by <b>Mohammed Laalahmi</b><br>
                Course: <b>CST2216 – Modularizing and Deploying ML Code</b><br>
                Algonquin College<br>
                Under the supervision of <b>Professor Dr. Umer Altaf</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    local_css()

    try:
        bundle = get_bundle()
    except Exception as e:
        st.error(f"Failed to load model bundle: {e}")
        st.stop()

    render_sidebar(bundle)
    render_header()

    st.markdown(
        '<p class="small-muted">Use the controls below to enter an applicant profile and generate a prediction.</p>',
        unsafe_allow_html=True,
    )

    input_data = render_input_section()

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        predict_button = st.button("Predict Admission Outcome")

    with col_right:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-title">About This Model</div>
                This application uses a modular machine learning pipeline with preprocessing,
                feature scaling, model comparison, saved artifacts, and interactive inference
                through Streamlit.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_button:
        try:
            result = predict_admission(input_data)
            render_prediction(result)
            render_input_summary(input_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Enter applicant information and click the prediction button.")

    render_footer()


if __name__ == "__main__":
    main()