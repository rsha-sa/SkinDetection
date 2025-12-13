import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from src.model_loader import ModelManager, LightweightUNet, UpBlock, ConvBNReLU
import warnings
from src.model_downloader import ensure_models


# 1. SETUP & CONFIGURATION
warnings.filterwarnings("ignore")

#load models
with st.spinner("Preparing AI models..."):
    ensure_models()


st.set_page_config(
    layout="wide", 
    page_title="DermaCare AI", 
    page_icon="",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS  ---
st.markdown("""
    <style>
    /* 1. GLOBAL RESET & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* 2. MAIN BACKGROUND */
    .stApp {
        background: #0f172a;
        background-image: radial-gradient(circle at 50% 0%, #2a1b1b 0%, #0f172a 60%); /* Slight reddish tint at top to match Rose theme */
        color: #e2e8f0;
    }
    
    /* 3. HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px;
        animation: fadeIn 0.8s ease-out;
        max-width: 900px;
        margin: 0 auto;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    
    /* GRADIENT TEXT TITLE */
    h1 {
        font-size: 3.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffd4c9 0%, #d15e66 50%, #ffd4c9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px !important;
        text-align: center;
        line-height: 1.1 !important;
    }
    
    /* SUBTITLE */
    .subtitle {
        font-size: 1.3rem;
        color: #94a3b8;
        font-weight: 500;
        line-height: 1.6;
        text-align: center;
    }

    /* 4. MODERN CARDS */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* 5. UPLOAD WIDGET STYLING */
    [data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.3);
        border: 2px dashed #d15e66;
        border-radius: 20px;
        padding: 40px;
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #c85a54; /* Rose Border */
        background-color: rgba(200, 90, 84, 0.1);
    }
    
    /* 6. RISK BADGES */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .malignant {
        background: rgba(127, 29, 29, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    .benign {
        background: rgba(6, 78, 59, 0.2);
        color: #86efac;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    /* 7. DIAGNOSIS TEXT */
    .diag-val {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 10px 0;
    }
    
    /* 8. BUTTONS  */
    .stButton>button {
        background: linear-gradient(135deg, #d15e66 50%) !important; /* Rose Gradient */
        color: white !important;
        border: none !important;
        padding: 0.8rem 2.5rem !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100%;
        box-shadow: 0 4px 20px rgba(200, 90, 84, 0.3) !important;
        transition: transform 0.2s !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(200, 90, 84, 0.5) !important;
    }
    
    /* 9. TABS & DATAFRAMES */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(30, 41, 59, 0.3);
        padding: 5px;
        border-radius: 12px;
        display: inline-flex;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #ffd4c9;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c85a54 !important; 
        color: white !important;
        box-shadow: 0 2px 10px rgba(200, 90, 84, 0.3);
    }
    .stDataFrame { background: transparent !important; }
    
    /* HIDE SIDEBAR TOGGLE */
    [data-testid="collapsedControl"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'results' not in st.session_state:
    st.session_state['results'] = {}

# --- MODEL LOADING ---
@st.cache_resource
def get_manager():
    return ModelManager()

try:
    manager = get_manager()
except Exception as e:
    st.error(f"System Failure: {e}")
    st.stop()

def composite_layers(base_image, masks_dict, selected_keys):
    composite = base_image.convert("RGBA")
    base_size = composite.size
    for key in selected_keys:
        if key in masks_dict:
            mask = masks_dict[key] 
            if mask.size != base_size: mask = mask.resize(base_size, Image.BILINEAR)
            composite = Image.alpha_composite(composite, mask)
    return composite

st.markdown("""
<div class="hero-container">
    <h1>Early Skin Detection Powered by AI</h1>
    <div class="subtitle">Upload a photo of your skin lesion and receive instant detection, segmentation, and classification.</div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LOGIC ---
if 'last_file' not in st.session_state: st.session_state['last_file'] = None

# FILE UPLOADER 
if not st.session_state.get('uploaded_file'):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="glass-card" style="text-align:center"><h3>Upload Image</h3><p style="color:#94a3b8">Supported formats: JPG, PNG, JPEG</p></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
        
        if uploaded:
            st.session_state['uploaded_file'] = uploaded
            st.rerun()

else:
    # DASHBOARD VIEW
    uploaded_file = st.session_state['uploaded_file']
    
    if st.session_state['last_file'] != uploaded_file.name:
        st.session_state['analysis_done'] = False
        st.session_state['results'] = {}
        st.session_state['last_file'] = uploaded_file.name

    image = Image.open(uploaded_file).convert('RGB')

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Original Image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        c_reset1, c_reset2, c_reset3 = st.columns([1, 2, 1])
        with c_reset2:
            if st.button("Analyze New Image"):
                del st.session_state['uploaded_file']
                st.rerun()

    with col_right:
        if not st.session_state['analysis_done']:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Diagnostic Engine")
            st.markdown("Ready to execute multi-modal fusion pipeline.")
            
            if st.button("RUN DIAGNOSTICS"):
                with st.status("Initializing Neural Networks...", expanded=True) as status:
                    st.write("Fusing CNN + Texture Features...")
                    fusion_res = manager.predict_fusion_classification(image)
                    
                    st.write("Localizing ROI (YOLO/FasterRCNN)...")
                    det_res = manager.predict_multi_detection(image)
                    
                    st.write("Generating Morphology Masks...")
                    seg_res = manager.predict_multi_segmentation(image)
                    
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                
                st.session_state['results'] = {'fusion': fusion_res, 'det': det_res, 'seg': seg_res}
                st.session_state['analysis_done'] = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state['analysis_done']:
            res = st.session_state['results']
            fusion = res['fusion']
            
            risk_class = "malignant" if fusion['risk'] == "Malignant" else "benign"
            border_col = "#ac3e3e" if fusion['risk'] == "Malignant" else "#3fad67"
            
            st.markdown(f"""
            <div class="glass-card" style="border-left: 6px solid {border_col}; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; color:#94a3b8; font-weight:700;">Final Assessment</div>
                    <div class="diag-val">{fusion['final_diagnosis']}</div>
                    <div class="risk-badge {risk_class}">{fusion['risk']}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:3rem; font-weight:800; color:{border_col}">{fusion['confidence']}</div>
                    <div style="color:#94a3b8">Confidence Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            t1, t2, t3 = st.tabs(["Model Consensus", "Detection", "Segmentation"])
            
            with t1:
                votes = fusion['votes']
                votes_sorted = sorted(votes, key=lambda x: x['confidence'], reverse=True)
                df_votes = pd.DataFrame(votes_sorted)
                df_votes = df_votes[['model', 'disease', 'confidence', 'risk']]
                df_votes.columns = ['Model', 'Prediction', 'Confidence', 'Risk']
                
                st.dataframe(
                    df_votes,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence", format="%.1f%%", min_value=0, max_value=1
                        ),
                        "Risk": st.column_config.TextColumn("Risk Analysis")
                    }
                )

            with t2:
                c_d1, c_d2 = st.columns([1, 2])
                with c_d1:
                    st.info("Select a model to view bounding box localization.")
                    det_options = list(res['det'].keys())
                    
                    ordered_det = []
                    if "YOLO" in det_options: ordered_det.append("YOLO")
                    if "FasterRCNN" in det_options: ordered_det.append("FasterRCNN")
                    if "RetinaNet" in det_options: ordered_det.append("RetinaNet")
                    for k in det_options:
                        if k not in ordered_det: ordered_det.append(k)

                    format_func = lambda x: f"{x} (Recommended)" if x == "YOLO" else x
                    selected_det = st.radio("Select Model", ordered_det, format_func=format_func)
                
                with c_d2:
                    if selected_det: st.image(res['det'][selected_det], use_container_width=True)

            with t3:
                c_s1, c_s2 = st.columns([1, 1.5])
                
                seg_order = ["Fusion (Top 3)", "SegFormer", "Attention U-Net V2", "HybridNet V1", "DeepLabV3+ V2"]
                avail_seg = [k for k in seg_order if k in res['seg']]
                for k in res['seg'].keys():
                    if k not in avail_seg: avail_seg.append(k)
                
                sel_masks = []
                
                with c_s1:
                    st.caption("Select layers to overlay:")
                    for k in avail_seg:
                        clean = k.replace("🏆 ", "")
                        if "SegFormer" in clean: clean += " (Best Single)"
                        if "Fusion" in clean: clean += " (Recommended)"
                        
                        def_val = ("Fusion" in k or "SegFormer" in k)
                        
                        ui_dot = "⚪"
                        if "SegFormer" in k: ui_dot = "🔴"
                        elif "Attention" in k: ui_dot = "🔵"
                        elif "Hybrid" in k: ui_dot = "🟢"
                        elif "DeepLab" in k: ui_dot = "🟡"
                        elif "Fusion" in k: ui_dot = "🟣"
                        
                        if st.checkbox(f"{ui_dot} {clean}", value=def_val, key=k):
                            sel_masks.append(k)
                            
                with c_s2:
                    if sel_masks:
                        final_comp = composite_layers(image, res['seg'], sel_masks)
                        st.image(final_comp, use_container_width=True)
                    else:
                        st.image(image, caption="Original", use_container_width=True)