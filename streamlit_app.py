"""
BetelVision AI — Streamlit Web App
Deteksi kondisi daun sirih menggunakan MobileNetV2
"""

import os
import io
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import time

# ===========================================================
# KONFIGURASI
# ===========================================================
MODEL_PATH   = "model_sirih.h5"
CLASSES_FILE = "classes.txt"
IMAGE_SIZE   = (224, 224)
CONFIDENCE_THRESHOLD = 70.0
CONFIDENCE_MARGIN    = 12.0

CLASS_COLORS = {
    "Sirih_sehat": "#2ecc71",
    "Sirih_sakit": "#e74c3c",
    "Sirih_layu" : "#f39c12",
}

CLASS_DESCRIPTIONS = {
    "Sirih_sehat": "🌿 Daun **SEHAT** — Tidak ada tanda-tanda penyakit. Daun hijau segar dengan tekstur normal.",
    "Sirih_sakit": "🔴 Daun **SAKIT** — Terdeteksi infeksi bakteri/jamur. Perlu penanganan segera.",
    "Sirih_layu" : "🟡 Daun **LAYU** — Kekurangan air atau nutrisi. Perlu penyiraman atau pemupukan.",
}

CLASS_ICONS = {
    "Sirih_sehat": "✅",
    "Sirih_sakit": "🚨",
    "Sirih_layu" : "⚠️",
}

# ===========================================================
# CSS KUSTOM
# ===========================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0d1f2d 100%);
        min-height: 100vh;
    }

    /* Hide default streamlit menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f2d 0%, #1a2332 100%);
        border-right: 1px solid #2d4a6b;
    }
    [data-testid="stSidebar"] .stMarkdown {color: #c9d6e3;}

    /* Cards */
    .result-card {
        background: linear-gradient(135deg, #1e3a4a, #1a2d3d);
        border-radius: 20px;
        padding: 28px 32px;
        border: 1px solid #2d5a7a;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    .stat-card {
        background: linear-gradient(135deg, #16213e, #1a2744);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #2d4080;
        text-align: center;
        margin-bottom: 16px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0d3b2e 0%, #1a4a37 50%, #0d2e20 100%);
        border-radius: 24px;
        padding: 40px;
        border: 1px solid #2d7a5a;
        text-align: center;
        margin-bottom: 30px;
    }

    .warning-card {
        background: linear-gradient(135deg, #2d1b00, #3d2500);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #8b5e00;
        margin-bottom: 20px;
    }

    .unknown-card {
        background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
        border-radius: 20px;
        padding: 28px 32px;
        border: 1px solid #555;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Progress bars */
    .stProgress > div > div {
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 15px;
        padding: 12px 28px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #219150, #27ae60);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4);
    }

    /* Upload area */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #1a2d3d, #1e3a4a) !important;
        border: 2px dashed #2d7a5a !important;
        border-radius: 16px !important;
        color: #8ab4c9 !important;
    }

    /* Text colors */
    h1, h2, h3 { color: #e8f4f0 !important; }
    p { color: #b0c8d4; }
    .stMarkdown { color: #b0c8d4; }

    /* Metric labels */
    [data-testid="stMetricLabel"] { color: #8ab4c9 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }

    /* Divider */
    hr { border-color: #2d4a6b; }

    /* Image border */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 2px solid #2d5a7a;
    }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================
# UTILITAS WARNA / HEURISTIK
# ===========================================================
def color_layu_score_from_image(img: Image.Image) -> float:
    try:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(float)
        s = hsv[:, :, 1].astype(float)
        v = hsv[:, :, 2].astype(float)

        yellow_mask = ((h >= 10) & (h <= 45)) & (s >= 20) & (s <= 95) & (v >= 35) & (v <= 180)
        brown_mask  = ((h >= 0)  & (h <= 20)) & (s >= 20) & (s <= 120) & (v >= 20) & (v <= 140)
        green_mask  = ((h >= 35) & (h <= 90)) & (s >= 30) & (v >= 40)

        score  = 0.45 * float(np.mean(yellow_mask))
        score += 0.35 * float(np.mean(brown_mask))
        score += 0.20 * max(0.0, 1.0 - float(np.mean(green_mask)))
        score += 0.10 * max(0.0, (140.0 - float(np.mean(v))) / 140.0)
        score += 0.10 * max(0.0, (60.0  - float(np.mean(s))) /  60.0)
        return min(1.0, max(0.0, score))
    except Exception:
        return 0.0


def is_probably_not_leaf(img: Image.Image) -> bool:
    try:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(float)
        s = hsv[:, :, 1].astype(float)
        v = hsv[:, :, 2].astype(float)

        green_mask  = ((h >= 35) & (h <= 90)) & (s >= 25) & (v >= 30)
        yellow_mask = ((h >= 10) & (h <= 45)) & (s >= 20) & (s <= 95) & (v >= 35) & (v <= 180)
        brown_mask  = ((h >= 0)  & (h <= 20)) & (s >= 20) & (s <= 120) & (v >= 20) & (v <= 140)
        leaf_ratio  = float(np.mean(green_mask | yellow_mask | brown_mask))
        mean_s      = float(np.mean(s))
        mean_v      = float(np.mean(v))

        return leaf_ratio < 0.08 and mean_s < 25.0 and mean_v < 90.0
    except Exception:
        return True


# ===========================================================
# MODEL LOADING (cached)
# ===========================================================
@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        return None, []
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = []
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            classes = [ln.strip() for ln in f if ln.strip()]
    return model, classes


# ===========================================================
# PREDICTION
# ===========================================================
def predict(model, classes, img: Image.Image):
    """Jalankan model dan kembalikan hasil prediksi lengkap."""
    img_resized = img.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = keras_image.img_to_array(img_resized)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    preds    = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3     = [(classes[i], float(preds[i] * 100)) for i in top3_idx]

    top_label, top_conf   = top3[0]
    second_label, sec_conf = top3[1] if len(top3) > 1 else (top_label, 0.0)
    is_confident = (top_conf >= CONFIDENCE_THRESHOLD and
                    (top_conf - sec_conf) >= CONFIDENCE_MARGIN)

    color_layu = color_layu_score_from_image(img)
    not_leaf   = is_probably_not_leaf(img)

    # Layu fallback
    if (top_label == "Sirih_sakit" and color_layu >= 0.62 and
            len(classes) > 1 and classes[1] == "Sirih_layu"):
        top_label   = "Sirih_layu"
        top_conf    = max(top_conf, 70.0)
        is_confident = True

    all_probs = {classes[i]: float(preds[i] * 100) for i in range(len(classes))}

    return {
        "top_label"    : top_label,
        "top_conf"     : top_conf,
        "is_confident" : is_confident,
        "not_leaf"     : not_leaf,
        "top3"         : top3,
        "all_probs"    : all_probs,
    }


# ===========================================================
# HALAMAN: BERANDA
# ===========================================================
def page_home():
    st.markdown("""
    <div class="hero-card">
        <h1 style="font-size:2.8rem; font-weight:800; margin:0; color:#2ecc71;">
            🌿 BetelVision AI
        </h1>
        <p style="font-size:1.15rem; color:#a8d5b5; margin-top:10px; margin-bottom:0;">
            Deteksi Cerdas Kondisi Daun Sirih dengan Kecerdasan Buatan
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size:2rem;">✅</div>
            <div style="font-size:1.1rem; font-weight:700; color:#2ecc71; margin-top:8px;">Sirih Sehat</div>
            <div style="font-size:0.85rem; color:#8ab4c9; margin-top:4px;">Daun hijau segar tanpa bercak</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size:2rem;">🚨</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e74c3c; margin-top:8px;">Sirih Sakit</div>
            <div style="font-size:0.85rem; color:#8ab4c9; margin-top:4px;">Terdeteksi infeksi/penyakit</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size:2rem;">⚠️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#f39c12; margin-top:8px;">Sirih Layu</div>
            <div style="font-size:0.85rem; color:#8ab4c9; margin-top:4px;">Kekurangan air atau nutrisi</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="result-card">
        <h3 style="color:#2ecc71; margin-top:0;">📖 Tentang Aplikasi</h3>
        <p style="color:#b0c8d4; line-height:1.7;">
            <b>BetelVision AI</b> adalah aplikasi deteksi kondisi daun sirih menggunakan model
            <b>MobileNetV2</b> yang dilatih dengan TensorFlow. Aplikasi ini membantu petani dan
            peneliti mendeteksi tanda-tanda awal penyakit atau stres pada tanaman sirih secara otomatis.
        </p>
        <p style="color:#b0c8d4; line-height:1.7; margin-bottom:0;">
            Daun sirih (<i>Piper betle</i>) adalah salah satu tanaman obat yang banyak dibudidayakan
            di Indonesia dan digunakan secara tradisional untuk kesehatan mulut, pengobatan luka,
            dan ramuan herbal lainnya.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card">
        <h3 style="color:#3498db; margin-top:0;">🚀 Cara Penggunaan</h3>
        <ol style="color:#b0c8d4; line-height:2;">
            <li>Pilih menu <b>"Deteksi Gambar"</b> di sidebar kiri</li>
            <li>Upload foto daun sirih (JPG, PNG, atau BMP)</li>
            <li>Tunggu beberapa detik untuk analisis AI</li>
            <li>Lihat hasil deteksi beserta tingkat akurasi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================
# HALAMAN: DETEKSI
# ===========================================================
def page_detect(model, classes):
    st.markdown("""
    <h2 style="color:#e8f4f0; font-weight:700; margin-bottom:4px;">
        🔬 Panel Deteksi Daun Sirih
    </h2>
    <p style="color:#8ab4c9; margin-bottom:24px;">
        Upload foto daun sirih untuk mendeteksi kondisinya secara otomatis.
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih gambar daun sirih",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#4a7a8a;">
            <div style="font-size:4rem;">📂</div>
            <p style="font-size:1.1rem; margin-top:12px;">Belum ada gambar yang diupload.<br>
            Gunakan tombol di atas untuk memilih foto.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    img = Image.open(uploaded).convert("RGB")

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**📷 Gambar yang Diupload**")
        st.image(img, use_column_width=True)

    with col_res:
        with st.spinner("🤖 Menganalisis gambar…"):
            start = time.time()
            result = predict(model, classes, img)
            elapsed = time.time() - start

        _render_result(result, elapsed)

    # --- Top-3 bar chart ---
    st.markdown("---")
    st.markdown("**📊 Probabilitas Semua Kelas**")
    _render_prob_bars(result["all_probs"])


def _render_result(result: dict, elapsed: float):
    """Render kotak hasil prediksi."""
    if result["not_leaf"]:
        st.markdown("""
        <div class="unknown-card">
            <div style="font-size:3rem;">🚫</div>
            <h3 style="color:#aaaaaa; margin-top:12px;">Objek Tidak Dikenali</h3>
            <p style="color:#777; margin-bottom:0;">
                Gambar ini tampak bukan daun sirih atau terlalu tidak jelas untuk dianalisis.
                Coba foto dengan pencahayaan baik dan daun memenuhi sebagian besar frame.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if not result["is_confident"]:
        top_label, top_conf = result["top3"][0]
        st.markdown(f"""
        <div class="warning-card">
            <div style="font-size:2.5rem;">🤔</div>
            <h3 style="color:#f39c12; margin-top:12px;">Tidak Cukup Yakin</h3>
            <p style="color:#c8a060; margin-bottom:8px;">
                Model tidak cukup yakin (akurasi <b>{top_conf:.1f}%</b> < {CONFIDENCE_THRESHOLD:.0f}%).
                Kemungkinan terbaik: <b>{top_label}</b>
            </p>
            <p style="color:#997040; font-size:0.85rem; margin-bottom:0;">
                💡 Tips: Pastikan daun sirih jelas, pencahayaan cukup, dan tidak ada objek lain yang menghalangi.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    label    = result["top_label"]
    conf     = result["top_conf"]
    color    = CLASS_COLORS.get(label, "#8ab4c9")
    icon     = CLASS_ICONS.get(label, "🌿")
    desc     = CLASS_DESCRIPTIONS.get(label, "")

    st.markdown(f"""
    <div class="result-card" style="border-color:{color}; border-width:2px;">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
            <span style="font-size:3rem;">{icon}</span>
            <div>
                <div style="font-size:0.85rem; color:#8ab4c9; text-transform:uppercase; letter-spacing:1px;">Hasil Deteksi</div>
                <div style="font-size:1.8rem; font-weight:800; color:{color};">{label}</div>
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px 16px; margin-bottom:16px;">
            <div style="font-size:0.8rem; color:#8ab4c9; margin-bottom:6px;">TINGKAT AKURASI</div>
            <div style="font-size:2.2rem; font-weight:800; color:{color};">{conf:.1f}%</div>
        </div>
        <p style="color:#b0c8d4; line-height:1.6; margin-bottom:8px;">{desc}</p>
        <div style="font-size:0.8rem; color:#5a7a8a;">⏱ Waktu analisis: {elapsed*1000:.0f} ms</div>
    </div>
    """, unsafe_allow_html=True)


def _render_prob_bars(all_probs: dict):
    label_colors = {
        "Sirih_sehat": "#2ecc71",
        "Sirih_sakit": "#e74c3c",
        "Sirih_layu" : "#f39c12",
    }
    for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        color = label_colors.get(label, "#8ab4c9")
        icon  = CLASS_ICONS.get(label, "🌿")
        col_l, col_p, col_b = st.columns([2, 1, 5])
        with col_l:
            st.markdown(f"<span style='color:{color}; font-weight:600;'>{icon} {label}</span>",
                        unsafe_allow_html=True)
        with col_p:
            st.markdown(f"<span style='color:#e0e0e0; font-weight:700;'>{prob:.1f}%</span>",
                        unsafe_allow_html=True)
        with col_b:
            st.progress(prob / 100.0)


# ===========================================================
# HALAMAN: TENTANG
# ===========================================================
def page_about():
    st.markdown("""
    <h2 style="color:#e8f4f0; font-weight:700; margin-bottom:4px;">
        ℹ️ Tentang BetelVision AI
    </h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#2ecc71; margin-top:0;">🧠 Teknologi</h3>
            <ul style="color:#b0c8d4; line-height:2; margin:0;">
                <li><b>Model:</b> MobileNetV2 (Transfer Learning)</li>
                <li><b>Framework:</b> TensorFlow / Keras</li>
                <li><b>Image Size:</b> 224 × 224 px</li>
                <li><b>Preprocessing:</b> MobileNetV2 standard</li>
                <li><b>Heuristik:</b> HSV color analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#3498db; margin-top:0;">📊 Dataset</h3>
            <ul style="color:#b0c8d4; line-height:2; margin:0;">
                <li><b>Kelas 1:</b> Sirih_sehat — Daun hijau normal</li>
                <li><b>Kelas 2:</b> Sirih_sakit — Terinfeksi penyakit</li>
                <li><b>Kelas 3:</b> Sirih_layu — Kekurangan nutrisi</li>
                <li><b>Augmentasi:</b> Rotasi, flip, brightness</li>
                <li><b>Split:</b> 80% train / 20% validasi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card">
        <h3 style="color:#8e44ad; margin-top:0;">⚙️ Cara Kerja Sistem</h3>
        <p style="color:#b0c8d4; line-height:1.7;">
            Sistem menggunakan dua lapisan validasi untuk mengurangi false positive:
        </p>
        <ol style="color:#b0c8d4; line-height:2;">
            <li><b>Heuristik Warna HSV</b> — Memeriksa apakah gambar mengandung
            warna yang khas daun (hijau, kuning, coklat) sebelum menjalankan model AI.</li>
            <li><b>Model CNN (MobileNetV2)</b> — Jaringan saraf dalam yang dilatih
            khusus pada dataset daun sirih. Hanya menampilkan hasil jika akurasi ≥ {threshold}%
            dengan selisih ≥ {margin}% dari kelas kedua.</li>
        </ol>
        <p style="color:#8ab4c9; font-size:0.85rem; margin-bottom:0;">
            Jika gambar tidak memenuhi syarat, sistem menampilkan "Tidak Dikenali" atau
            "Tidak Cukup Yakin" daripada memberikan hasil yang menyesatkan.
        </p>
    </div>
    """.format(threshold=int(CONFIDENCE_THRESHOLD), margin=int(CONFIDENCE_MARGIN)),
    unsafe_allow_html=True)


# ===========================================================
# MAIN
# ===========================================================
def main():
    st.set_page_config(
        page_title="BetelVision AI — Deteksi Daun Sirih",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # --- Load resources ---
    with st.spinner("Memuat model AI…"):
        model, classes = load_model_and_classes()

    if model is None:
        st.error("❌ Model tidak ditemukan! Pastikan file `model_sirih.h5` ada di direktori yang sama.")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px 10px 10px;">
            <div style="font-size:3rem;">🌿</div>
            <div style="font-size:1.5rem; font-weight:800; color:#2ecc71; margin-top:8px;">
                BetelVision AI
            </div>
            <div style="font-size:0.85rem; color:#8ab4c9; margin-top:4px;">
                Deteksi Daun Sirih
            </div>
        </div>
        <hr style="border-color:#2d4a6b; margin:16px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigasi",
            ["🏠 Beranda", "🔬 Deteksi Gambar", "ℹ️ Tentang"],
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:12px; background:rgba(39,174,96,0.1); border-radius:10px;
                    border:1px solid #2d7a5a;">
            <div style="font-size:0.75rem; color:#2ecc71; font-weight:600;">STATUS MODEL</div>
            <div style="font-size:0.85rem; color:#a8d5b5; margin-top:4px;">✅ Model berhasil dimuat</div>
            <div style="font-size:0.75rem; color:#5a8a6a; margin-top:2px;">
                Kelas: {classes}
            </div>
        </div>
        """.format(classes=", ".join(classes)), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem; color:#3a5a6a; text-align:center;">
            © 2024 BetelVision AI<br>Computer Vision · TensorFlow
        </div>
        """, unsafe_allow_html=True)

    # --- Route ---
    if page == "🏠 Beranda":
        page_home()
    elif page == "🔬 Deteksi Gambar":
        page_detect(model, classes)
    elif page == "ℹ️ Tentang":
        page_about()


if __name__ == "__main__":
    main()
