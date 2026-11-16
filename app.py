"""
What's Next? - Deep Learning Next Word Predictor
Aesthetic Light-Themed Web Frontend built with Streamlit
"""

import os
import json
import time
import streamlit as st
import pandas as pd
from predictor import NextWordPredictor, TF_AVAILABLE

# Page Configuration
st.set_page_config(
    page_title="What's Next? • LSTM Next Word Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light-Themed Aesthetic CSS
CUSTOM_CSS = """
<style>
/* Global styles */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1E293B;
}

/* Light background */
.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
}

/* Header Banner */
.hero-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #1E293B 0%, #4338CA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #64748B;
    font-weight: 400;
}

/* Cards */
.card-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
    transition: all 0.2s ease;
}

.card-box:hover {
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Status Badges */
.badge-ready {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-demo {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFFBEB;
    color: #92400E;
    border: 1px solid #FDE68A;
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Result Box */
.output-box {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 1.15rem;
    line-height: 1.6;
    color: #0F172A;
    font-weight: 500;
    margin-top: 10px;
}

.highlight-word {
    background: #EEF2FF;
    color: #4F46E5;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
    border: 1px solid #C7D2FE;
}

/* Pill buttons */
div.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid #E2E8F0;
    transition: all 0.15s ease-in-out;
}

div.stButton > button:hover {
    border-color: #6366F1;
    color: #6366F1;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.12);
}

/* Probability item */
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 0.9rem;
    font-weight: 500;
}

.prob-bar-container {
    background: #F1F5F9;
    border-radius: 8px;
    height: 8px;
    width: 100%;
    margin-bottom: 12px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366F1, #818CF8);
    border-radius: 8px;
}

/* Metric mini box */
.metric-tile {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}
.metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1E293B;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = "The only way to"
if "last_generated_text" not in st.session_state:
    st.session_state.last_generated_text = ""


# Load Predictor (Cached)
@st.cache_resource
def get_predictor():
    return NextWordPredictor(
        model_path="lstm_model.h5",
        tokenizer_path="tokenizer.pkl",
        max_len_path="max_len.pkl"
    )

predictor = get_predictor()


# Load Metadata if present
metadata = {}
if os.path.exists("model_metadata.json"):
    try:
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception:
        pass


# ========================================================
# HEADER SECTION
# ========================================================
status_badge = (
    '<span class="badge-ready">● Model Active (lstm_model.h5)</span>'
    if predictor.is_loaded else
    '<span class="badge-demo">⚡ Interactive Demo Mode (Awaiting .h5)</span>'
)

st.markdown(f"""
<div class="hero-container">
    <div>
        <div class="hero-title">What's Next? ✍️</div>
        <div class="hero-subtitle">Deep Learning LSTM Next-Word Prediction & Real-Time Sequence Generation</div>
    </div>
    <div>
        {status_badge}
    </div>
</div>
""", unsafe_allow_html=True)


# Notice if model is not loaded
if not predictor.is_loaded:
    with st.expander("ℹ️ How to load your trained .h5 model", expanded=False):
        st.markdown("""
        **No trained `lstm_model.h5` found yet** — the interface is currently running in **interactive demo preview mode**.
        
        To load your real Deep Learning model:
        1. **Train locally or in Colab**: Run `python train.py` or execute `train_colab.ipynb`.
        2. Place the generated files in this project folder:
           - `lstm_model.h5`
           - `tokenizer.pkl`
           - `max_len.pkl`
        3. Click **Reload Model** in the sidebar.
        """)


# ========================================================
# MAIN LAYOUT (2 Columns)
# ========================================================
col_left, col_right = st.columns([1.1, 0.9], gap="large")

# ----------------- LEFT COLUMN: PREDICTION & GENERATION -----------------
with col_left:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💡 Seed Prompt Input</div>', unsafe_allow_html=True)

    # Preset Inspiration Chips
    st.caption("Quick starter ideas:")
    chip_cols = st.columns(4)
    if chip_cols[0].button("The only way", use_container_width=True):
        st.session_state.prompt_input = "The only way to"
    if chip_cols[1].button("Life is what", use_container_width=True):
        st.session_state.prompt_input = "Life is what happens"
    if chip_cols[2].button("Success is", use_container_width=True):
        st.session_state.prompt_input = "Success is not"
    if chip_cols[3].button("Believe in", use_container_width=True):
        st.session_state.prompt_input = "Believe in yourself and"

    # Input Box
    user_prompt = st.text_area(
        label="Type a beginning of a sentence or quote:",
        value=st.session_state.prompt_input,
        height=95,
        placeholder="Type any phrase here (e.g., 'What are you')...",
        key="user_text_area"
    )
    # Sync with session state
    st.session_state.prompt_input = user_prompt

    st.markdown('</div>', unsafe_allow_html=True)

    # Next Word Prediction Output Box
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Instant Next-Word Candidates</div>', unsafe_allow_html=True)

    predictions = predictor.predict_top_k(user_prompt, top_k=5)

    if predictions:
        st.write("Click any predicted word below to append it to your sentence:")
        
        # Word suggestion buttons
        chip_button_cols = st.columns(len(predictions))
        for idx, item in enumerate(predictions):
            btn_label = f"{item['word']} ({item['pct']})"
            if chip_button_cols[idx].button(btn_label, key=f"chip_btn_{idx}", use_container_width=True):
                # Append clicked word
                st.session_state.prompt_input = user_prompt.strip() + " " + item['word']
                st.rerun()

        # Highest confidence recommendation
        top_word = predictions[0]
        st.markdown(f"""
        <div class="output-box">
            Recommended continuation: <br>
            <span>{user_prompt} </span><span class="highlight-word">{top_word['word']}</span>
            <span style="font-size:0.85rem; color:#6B7280; margin-left:8px;">({top_word['pct']} confidence)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Enter at least one word above to predict the next word.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Multi-Word Sequence Generator
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✨ Deep Sequence Generator</div>', unsafe_allow_html=True)
    st.caption("Generate complete quotes and multi-word sentences using recursive LSTM passes.")

    gen_col1, gen_col2 = st.columns(2)
    with gen_col1:
        n_words = st.slider("Words to generate", min_value=1, max_value=20, value=6)
    with gen_col2:
        temperature = st.slider("Temperature / Creativity", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
                                help="Lower is deterministic and focused; higher is diverse and creative.")

    if st.button("🚀 Generate Full Continuation", type="primary", use_container_width=True):
        if user_prompt.strip():
            with st.spinner("Generating sequence with LSTM..."):
                generated = predictor.generate_sequence(user_prompt, n_words=n_words, temperature=temperature)
                st.session_state.last_generated_text = generated
        else:
            st.warning("Please type a seed prompt first.")

    if st.session_state.last_generated_text:
        st.markdown(f"""
        <div class="output-box" style="border-left: 4px solid #6366F1;">
            <b>Generated Result:</b><br>
            "{st.session_state.last_generated_text}"
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ----------------- RIGHT COLUMN: PROBABILITY DISTRIBUTION & SPECS -----------------
with col_right:
    # Probability Distribution Card
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Top Candidates Probability Distribution</div>', unsafe_allow_html=True)

    if predictions:
        for item in predictions:
            prob_pct = item['prob'] * 100
            st.markdown(f"""
            <div class="prob-row">
                <span style="font-weight:600; color:#1E293B;">"{item['word']}"</span>
                <span style="color:#6366F1; font-weight:700;">{item['pct']}</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width: {min(max(prob_pct, 4), 100)}%;"></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No distribution available yet. Type in a word to see probability rankings.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Architecture Card
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧠 Deep Learning Architecture</div>', unsafe_allow_html=True)

    vocab_disp = metadata.get("vocab_size", len(predictor.tokenizer.word_index) + 1 if predictor.tokenizer else "—")
    maxlen_disp = metadata.get("max_len", predictor.max_len if predictor.is_loaded else "—")
    units_disp = metadata.get("lstm_units", 150)
    acc_disp = f"{metadata.get('final_accuracy', 0.0) * 100:.1f}%" if "final_accuracy" in metadata else "N/A"

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{vocab_disp}</div>
            <div class="metric-label">Vocabulary Size</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{maxlen_disp}</div>
            <div class="metric-label">Max Sequence Len</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    m_col3, m_col4 = st.columns(2)
    with m_col3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{units_disp}</div>
            <div class="metric-label">LSTM Units</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{acc_disp}</div>
            <div class="metric-label">Training Acc</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.85rem; color:#64748B; margin-top:16px; line-height:1.5;">
        <b>Pipeline Flow:</b><br>
        1. <b>Text Cleaning</b>: Lowercasing & punctuation stripping<br>
        2. <b>Tokenizer</b>: Words mapped to discrete index sequences<br>
        3. <b>Pad Sequences</b>: Pre-padded to fixed context window<br>
        4. <b>Embedding Layer</b>: Dense continuous semantic space<br>
        5. <b>LSTM Layer</b>: Recurrent temporal memory capturing context<br>
        6. <b>Softmax Output</b>: Multiclass probability across vocab
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================
# SIDEBAR
# ========================================================
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    if st.button("🔄 Refresh / Reload Model", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Model File Status")
    
    m_exists = os.path.exists("lstm_model.h5")
    t_exists = os.path.exists("tokenizer.pkl")
    l_exists = os.path.exists("max_len.pkl")

    st.markdown(f"• `lstm_model.h5`: {'✅ Ready' if m_exists else '❌ Not Found'}")
    st.markdown(f"• `tokenizer.pkl`: {'✅ Ready' if t_exists else '❌ Not Found'}")
    st.markdown(f"• `max_len.pkl`: {'✅ Ready' if l_exists else '❌ Not Found'}")

    st.markdown("---")
    st.markdown("### 📤 Upload Trained Files")
    uploaded_h5 = st.file_uploader("Upload lstm_model.h5", type=["h5", "keras"])
    if uploaded_h5:
        with open("lstm_model.h5", "wb") as f:
            f.write(uploaded_h5.getbuffer())
        st.success("Saved lstm_model.h5!")
        st.cache_resource.clear()

    uploaded_tok = st.file_uploader("Upload tokenizer.pkl", type=["pkl"])
    if uploaded_tok:
        with open("tokenizer.pkl", "wb") as f:
            f.write(uploaded_tok.getbuffer())
        st.success("Saved tokenizer.pkl!")
        st.cache_resource.clear()

    st.markdown("---")
    st.markdown("### 🎓 Training Quick Guide")
    st.caption("""
    To train manually:
    ```bash
    # 1. Install dependencies
    pip install tensorflow pandas matplotlib
    
    # 2. Run training script
    python train.py --data your_quotes.csv
    ```
    Or upload `train_colab.ipynb` to Google Colab for free GPU training.
    """)
