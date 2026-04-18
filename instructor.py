"""
instructor_tool/instructor.py
==============================
Streamlit app: Instructor Batch Prediction Dashboard.

Upload a class CSV → predict G3 for every student →
compute totals & pass/fail → visualise results → download CSV.

Run:
    streamlit run instructor.py
"""

import io
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Import shared helpers from train.py (same directory) ──────────────────────
# Guarantees preprocessing is 100% identical to training.
sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    preprocess_inference,
    compute_verdict,
    PASS_THRESHOLD,
    G3_MAX,
    MODEL_PATH,
    SCALER_PATH,
    COLUMNS_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Instructor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --navy:   #1e3a5f;
    --teal:   #0f766e;
    --teal2:  #14b8a6;
    --amber:  #d97706;
    --pass:   #10b981;
    --fail:   #ef4444;
    --ink:    #111827;
    --muted:  #6b7280;
    --bg:     #f8fafc;
    --white:  #ffffff;
    --shadow: 0 2px 16px rgba(0,0,0,.07);
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg);
    color: var(--ink);
}
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--navy) !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(120deg, #1e3a5f 0%, #0f5f5a 100%);
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 24px;
}
.page-header h1 {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px;
}
.page-header p {
    color: #94d5cf;
    font-size: .95rem;
    margin: 0;
    font-weight: 300;
}

/* ── Section card ── */
.section {
    background: var(--white);
    border-radius: 12px;
    padding: 20px 20px 12px;
    box-shadow: var(--shadow);
    margin-bottom: 16px;
}
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: .95rem;
    font-weight: 600;
    color: var(--navy);
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* ── Summary stat cards ── */
.stat-row { display:flex; gap:12px; margin-bottom:20px; }
.stat-card {
    flex:1; background:var(--white); border-radius:12px;
    padding:18px 14px; box-shadow:var(--shadow);
    text-align:center; border-top:3px solid transparent;
}
.sc-total { border-top-color: var(--navy); }
.sc-pass  { border-top-color: var(--pass); }
.sc-fail  { border-top-color: var(--fail); }
.sc-avg   { border-top-color: var(--amber); }
.sc-rate  { border-top-color: var(--teal2); }
.sc-num {
    font-family:'Sora',sans-serif; font-size:2rem;
    font-weight:700; line-height:1; margin-bottom:3px;
}
.sc-lbl { font-size:.7rem; color:var(--muted); text-transform:uppercase;
           letter-spacing:.08em; }

/* ── Info / warn / err boxes ── */
.info-box { background:#eff6ff; border-left:4px solid #3b82f6;
            border-radius:0 8px 8px 0; padding:10px 14px;
            font-size:.84rem; color:#1e40af; margin-bottom:10px; }
.warn-box { background:#fefce8; border-left:4px solid var(--amber);
            border-radius:0 8px 8px 0; padding:10px 14px;
            font-size:.84rem; color:#92400e; margin-bottom:10px; }
.err-box  { background:#fef2f2; border-left:4px solid var(--fail);
            border-radius:0 8px 8px 0; padding:10px 14px;
            font-size:.84rem; color:#991b1b; margin-bottom:10px; }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: var(--teal) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    transition: opacity .2s !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover { opacity: .85 !important; }

/* ── Empty state ── */
.empty-state {
    background: var(--white);
    border-radius: 14px;
    padding: 60px 24px;
    text-align: center;
    box-shadow: var(--shadow);
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 20px;">
        <div style="font-size:2.6rem;">📊</div>
        <div style="font-family:'Sora',sans-serif;font-size:1.2rem;
                    font-weight:700;color:#fff;margin-top:6px;">
            Instructor Tool
        </div>
        <div style="font-size:.76rem;color:#94a3b8;margin-top:2px;">
            Batch Grade Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Grading rules ────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:.7rem;font-weight:600;letter-spacing:.12em;
                text-transform:uppercase;color:#64748b;margin-bottom:8px;">
        Grading Rules
    </div>
    <div style="font-size:.82rem;color:#cbd5e1;line-height:1.9;">
        G1, G2 in CSV → school scale <strong style="color:#fff">0–25</strong><br>
        G3 (predicted) → display <strong style="color:#fff">0–50</strong><br>
        Total = G1<sub>disp</sub> + G2<sub>disp</sub> + G3<br>
        <span style="color:#4ade80;">✓ PASS</span> if total
        <strong style="color:#fff">≥ 60</strong><br>
        <span style="color:#f87171;">✗ FAIL</span> otherwise
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:.76rem;color:#94a3b8;line-height:1.7;">
        🔒 <strong style="color:#cbd5e1">Privacy-first</strong><br>
        Sensitive columns are automatically
        removed before prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.7rem;color:#475569;text-align:center;">
        UCI Student Performance Dataset<br>
        Streamlit &amp; scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artefacts():
    """Load and cache model, scaler, and feature columns."""
    missing = [p for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]
               if not os.path.exists(p)]
    if missing:
        return None, None, None
    return (joblib.load(MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(COLUMNS_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode DataFrame as UTF-8 CSV bytes for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")


def style_verdict(df: pd.DataFrame):
    """Row-colour the results table: green for PASS, red for FAIL."""
    def colour(row):
        bg = "#d1fae5" if row.get("Verdict") == "PASS" else "#fee2e2"
        return [f"background-color:{bg}"] * len(row)
    return df.style.apply(colour, axis=1)


def make_sample_csv() -> bytes:
    """Generate a valid sample class CSV for instructors to try immediately."""
    data = pd.DataFrame({
        "sex":        ["M","F","M","F","M","F","M"],
        "age":        [17, 16, 18, 15, 17, 16, 19],
        "Medu":       [3,  4,  2,  3,  1,  4,  3],
        "Fedu":       [3,  4,  2,  2,  1,  3,  2],
        "Mjob":       ["other","teacher","at_home","services","other","health","other"],
        "Fjob":       ["services","health","other","other","at_home","teacher","services"],
        "reason":     ["course","reputation","home","course","other","reputation","course"],
        "guardian":   ["mother","mother","father","mother","father","mother","mother"],
        "traveltime": [2,1,1,3,2,1,2],
        "studytime":  [3,4,1,2,1,3,2],
        "failures":   [0,0,1,0,2,0,0],
        "schoolsup":  ["no","no","yes","no","yes","no","no"],
        "famsup":     ["yes","no","yes","yes","no","yes","yes"],
        "activities": ["no","yes","no","yes","no","yes","no"],
        "higher":     ["yes","yes","no","yes","no","yes","yes"],
        "internet":   ["yes","yes","no","yes","no","yes","yes"],
        "famrel":     [4,5,3,4,2,5,4],
        "freetime":   [3,2,4,3,5,2,3],
        "health":     [4,5,3,4,2,5,3],
        "absences":   [2,0,8,4,15,1,3],
        "G1":         [14,16,10,13,8,17,12],  # school scale 0-25
        "G2":         [13,15,9,12,7,16,11],   # school scale 0-25
    })
    return to_csv_bytes(data)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
    <h1>📊 Instructor Dashboard</h1>
    <p>Upload a class CSV to predict G3 for all students, analyse results, and export.</p>
</div>
""", unsafe_allow_html=True)

model, scaler, feature_columns = load_artefacts()

if model is None:
    st.error(
        "⚠️ Model files not found. "
        "Please run `python train.py` first, then restart the app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section"><div class="section-title">📂 Upload Student Data (CSV)</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Upload a CSV with student records. Columns <code>G1</code> and <code>G2</code>
must be present on the <strong>school scale 0–25</strong>.
Sensitive columns are ignored automatically.
</div>
""", unsafe_allow_html=True)

col_up, col_sample = st.columns([2, 1], gap="large")

with col_up:
    uploaded = st.file_uploader(
        "Drop CSV here or click to browse",
        type=["csv"],
        help="Comma- or semicolon-separated. G1/G2 should be on the school scale 0–25."
    )

with col_sample:
    st.markdown("**No file yet? Download a sample:**")
    st.download_button(
        "⬇️  Sample CSV",
        data=make_sample_csv(),
        file_name="sample_class.csv",
        mime="text/csv",
    )
    st.markdown("""
    <small style="color:#6b7280;">
    G1 / G2: school scale 0–25<br>
    G3 will be predicted by the model<br>
    Sensitive columns auto-dropped
    </small>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING & RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is not None:

    # ── Parse CSV ─────────────────────────────────────────────────────────
    try:
        raw_bytes = uploaded.read()
        sample    = raw_bytes[:2000].decode("utf-8", errors="ignore")
        sep       = ";" if sample.count(";") > sample.count(",") else ","
        df_raw    = pd.read_csv(io.BytesIO(raw_bytes), sep=sep)
    except Exception as exc:
        st.markdown(f'<div class="err-box">❌ Could not parse CSV: {exc}</div>',
                    unsafe_allow_html=True)
        st.stop()

    st.success(
        f"✅ **{uploaded.name}** loaded — "
        f"{len(df_raw):,} rows × {df_raw.shape[1]} columns"
    )

    # ── Validate required columns ─────────────────────────────────────────
    required = ["G1", "G2"]
    missing_req = [c for c in required if c not in df_raw.columns]
    if missing_req:
        st.markdown(
            f'<div class="err-box">❌ Required columns missing: '
            f'<strong>{", ".join(missing_req)}</strong>. '
            'Cannot compute total scores without G1 and G2.</div>',
            unsafe_allow_html=True
        )
        st.stop()

    # ── Warn about missing optional columns ───────────────────────────────
    expected_optional = [
        "sex","age","Medu","Fedu","Mjob","Fjob","reason","guardian",
        "traveltime","studytime","failures","schoolsup","famsup",
        "activities","higher","internet","famrel","freetime","health","absences",
    ]
    missing_opt = [c for c in expected_optional if c not in df_raw.columns]
    if missing_opt:
        st.markdown(
            f'<div class="warn-box">⚠️ Missing optional columns: '
            f'<strong>{", ".join(missing_opt)}</strong>. '
            'These will be zero-filled — predictions may be less accurate.</div>',
            unsafe_allow_html=True
        )

    # ── Missing values ────────────────────────────────────────────────────
    n_null = df_raw.isnull().sum().sum()
    if n_null:
        st.markdown(
            f'<div class="warn-box">⚠️ Found {n_null} missing value(s) — auto-imputed.</div>',
            unsafe_allow_html=True
        )

    # ── Validate G1/G2 range ──────────────────────────────────────────────
    for col in ["G1", "G2"]:
        out_of_range = ((df_raw[col] < 0) | (df_raw[col] > 25)).sum()
        if out_of_range:
            st.markdown(
                f'<div class="warn-box">⚠️ {out_of_range} value(s) in <code>{col}</code> '
                'are outside 0–25 (school scale) and will be clipped.</div>',
                unsafe_allow_html=True
            )
            df_raw[col] = df_raw[col].clip(0, 25)

    # ── Preprocess + predict ──────────────────────────────────────────────
    try:
        X_batch = preprocess_inference(df_raw, scaler, feature_columns)
    except Exception as exc:
        st.markdown(f'<div class="err-box">❌ Preprocessing failed: {exc}</div>',
                    unsafe_allow_html=True)
        st.stop()

    # Model predicts G3 directly on school scale (0-50)
    # Model predicts G3 directly on school scale (0-50) — no conversion needed.
    # CSV G1/G2 must be on the school scale (0-25) already.
    g3_disp_arr = np.clip(model.predict(X_batch), 0, G3_MAX).round(1)
    g1_disp_arr = df_raw["G1"].values.astype(float)   # already 0-25
    g2_disp_arr = df_raw["G2"].values.astype(float)   # already 0-25

    # Compute totals and verdicts
    totals, verdicts = [], []
    for g1d, g2d, g3d in zip(g1_disp_arr, g2_disp_arr, g3_disp_arr):
        total, passed = compute_verdict(g1d, g2d, g3d)
        totals.append(total)
        verdicts.append("PASS" if passed else "FAIL")

    totals   = np.array(totals)
    verdicts = np.array(verdicts)

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY STATISTICS
    # ─────────────────────────────────────────────────────────────────────

    n_total  = len(verdicts)
    n_pass   = int((verdicts == "PASS").sum())
    n_fail   = int((verdicts == "FAIL").sum())
    avg_g3   = float(g3_disp_arr.mean())
    pass_rate = n_pass / n_total * 100 if n_total else 0

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card sc-total">
        <div class="sc-num" style="color:#1e3a5f">{n_total}</div>
        <div class="sc-lbl">Students</div>
      </div>
      <div class="stat-card sc-pass">
        <div class="sc-num" style="color:#10b981">{n_pass}</div>
        <div class="sc-lbl">PASS</div>
      </div>
      <div class="stat-card sc-fail">
        <div class="sc-num" style="color:#ef4444">{n_fail}</div>
        <div class="sc-lbl">FAIL</div>
      </div>
      <div class="stat-card sc-avg">
        <div class="sc-num" style="color:#d97706">{avg_g3:.1f}</div>
        <div class="sc-lbl">Avg G3 / 50</div>
      </div>
      <div class="stat-card sc-rate">
        <div class="sc-num" style="color:#14b8a6">{pass_rate:.1f}%</div>
        <div class="sc-lbl">Pass Rate</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # BUILD RESULTS DATAFRAME
    # ─────────────────────────────────────────────────────────────────────

    # Select safe display columns from the original upload
    safe_cols = [c for c in ["sex","age","studytime","failures",
                              "absences","health","internet","higher","G1","G2"]
                 if c in df_raw.columns]

    df_results = df_raw[safe_cols].copy()
    df_results["G3 Predicted (0-50)"] = g3_disp_arr.round(1)
    df_results["Total (0-100)"]       = totals.round(1)
    df_results["Verdict"]            = verdicts

    # ─────────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────

    col_v1, col_v2 = st.columns(2, gap="large")

    with col_v1:
        st.markdown('<div class="section"><div class="section-title">📊 Pass vs Fail</div>',
                    unsafe_allow_html=True)
        vc = pd.Series(verdicts).value_counts().rename("Count")
        st.bar_chart(vc, color="#0f766e", height=240)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_v2:
        st.markdown('<div class="section"><div class="section-title">📈 G3 Distribution (0–50)</div>',
                    unsafe_allow_html=True)
        # Bin G3 into 10-unit buckets for a readable histogram
        bins   = [0, 10, 20, 30, 40, 50]
        labels = ["0-10","11-20","21-30","31-40","41-50"]
        binned = pd.cut(g3_disp_arr, bins=bins, labels=labels, include_lowest=True)
        hist   = pd.Series(binned).value_counts().sort_index().rename("Students")
        st.bar_chart(hist, color="#14b8a6", height=240)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Total score distribution ──────────────────────────────────────────
    st.markdown('<div class="section"><div class="section-title">📉 Total Score Distribution</div>',
                unsafe_allow_html=True)
    total_bins   = [0,20,40,60,80,100]
    total_labels = ["0-20","21-40","41-60","61-80","81-100"]
    total_binned = pd.cut(totals, bins=total_bins, labels=total_labels, include_lowest=True)
    total_hist   = pd.Series(total_binned).value_counts().sort_index().rename("Students")
    st.bar_chart(total_hist, color="#1e3a5f", height=200)
    st.caption(f"Pass threshold: **{PASS_THRESHOLD}** | Max: **100**")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── At-risk spotlight ─────────────────────────────────────────────────
    df_fail = df_results[df_results["Verdict"] == "FAIL"].copy()
    if not df_fail.empty:
        st.markdown('<div class="section">'
                    '<div class="section-title">🚨 At-Risk Students</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="warn-box">⚠️ <strong>{len(df_fail)} student(s)</strong> '
            'predicted to fail. Consider early intervention.</div>',
            unsafe_allow_html=True
        )
        st.dataframe(
            style_verdict(df_fail.sort_values("Total (0-100)")),
            use_container_width=True,
            height=min(400, 58 + len(df_fail) * 36),
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # FULL RESULTS TABLE WITH FILTERS
    # ─────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section"><div class="section-title">📋 Full Results</div>',
                unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filt = st.selectbox("Filter", ["All","PASS only","FAIL only"])
    with fc2:
        sort_by = st.selectbox("Sort by",
            ["Total (0-100)","G3 Predicted (0-50)"])
    with fc3:
        order = st.radio("Order", ["Descending","Ascending"],
                         horizontal=True) == "Ascending"

    df_view = df_results.copy()
    if filt == "PASS only":
        df_view = df_view[df_view["Verdict"] == "PASS"]
    elif filt == "FAIL only":
        df_view = df_view[df_view["Verdict"] == "FAIL"]
    if sort_by in df_view.columns:
        df_view = df_view.sort_values(sort_by, ascending=order)

    st.dataframe(
        style_verdict(df_view),
        use_container_width=True,
        height=min(520, 58 + len(df_view) * 36),
    )
    st.caption(f"Showing {len(df_view):,} of {n_total:,} students")
    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # DOWNLOAD BUTTONS
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("---")
    dl1, dl2, _ = st.columns([1,1,2])

    with dl1:
        st.download_button(
            "⬇️  Download Full Results",
            data=to_csv_bytes(df_results),
            file_name="class_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        if not df_fail.empty:
            st.download_button(
                "⬇️  Download At-Risk Students",
            data=to_csv_bytes(df_fail.sort_values("Total (0-100)")),
                use_container_width=True,
            )

else:
    # ── Empty state ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:3rem;margin-bottom:12px;">📂</div>
      <div style="font-family:'Sora',sans-serif;font-size:1.35rem;
                  font-weight:600;color:#1e3a5f;margin-bottom:6px;">
        No file uploaded yet
      </div>
      <div style="font-size:.88rem;color:#6b7280;line-height:1.7;
                  max-width:380px;margin:0 auto;">
        Upload a student CSV above to generate G3 predictions,
        total scores, and pass/fail verdicts for your entire class.<br><br>
        Use the <strong>Sample CSV</strong> button above to try it instantly.
      </div>
    </div>
    """, unsafe_allow_html=True)