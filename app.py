import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from deepface import DeepFace
from PIL import Image

# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="Facial Expression Classification System",
    page_icon="😊",
    layout="wide"
)

# ==================================================
# Custom CSS – Minimalis & Akademis
# ==================================================
st.markdown("""
<style>
    body { background-color: #0e1117; color: #fafafa; }
    .title { font-size: 42px; font-weight: 700; }
    .subtitle { font-size: 18px; color: #9ca3af; margin-bottom: 20px; }
    .card { background: #161b22; padding: 24px; border-radius: 18px; margin-bottom: 20px; }
    .metric { font-size: 32px; font-weight: 700; }
    .label { color: #9ca3af; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Models & Encoders
# ==================================================
@st.cache_resource
def load_resources():
    models = {
        "Skenario 1 – DeepFace": load_model("model/model_skenario1_deepface.h5"),
        "Skenario 1 – DeepFace + CNN1D": load_model("model/model_skenario1_deepface_cnn1d.h5"),
        "Skenario 2 – CNN Scratch": load_model("model/model_skenario2_cnn.h5"),
    }
    label_encoder = np.load("model/label_encoder.npy", allow_pickle=True)
    return models, label_encoder

models, label_encoder = load_resources()

# ==================================================
# Header
# ==================================================
st.markdown('<div class="title">Facial Expression Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transfer Learning vs CNN Scratch – DeepFace Integrated</div>', unsafe_allow_html=True)

# ==================================================
# Sidebar – Mode Selection
# ==================================================
st.sidebar.title("📌 Mode Sistem")
mode = st.sidebar.radio(
    "Pilih Mode",
    ["Prediksi Ekspresi", "Evaluasi & Perbandingan Model"]
)

# ==================================================
# MODE 1: PREDIKSI EKSPRESI
# ==================================================
if mode == "Prediksi Ekspresi":

    st.subheader("🧠 Mode Prediksi Ekspresi Wajah")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        selected_model_name = st.selectbox("Pilih Model", list(models.keys()))
        uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])
        predict_btn = st.button("🔍 Prediksi Ekspresi", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Hasil Prediksi")

        if uploaded_file and predict_btn:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

            img = np.array(image)
            img_resized = cv2.resize(img, (224, 224)) / 255.0
            img_input = np.expand_dims(img_resized, axis=0)

            model = models[selected_model_name]
            preds = model.predict(img_input)[0]
            idx = np.argmax(preds)
            label = label_encoder[idx]
            confidence = preds[idx] * 100

            st.markdown(f"<p class='metric'>{label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='label'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

            # Probability chart
            st.subheader("📈 Tingkat Kepercayaan")
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(label_encoder.tolist(), preds * 100)
            bars[idx].set_color('#facc15')
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)")
            st.pyplot(fig)

        else:
            st.info("Upload gambar dan tekan tombol prediksi")

        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# MODE 2: EVALUASI & PERBANDINGAN MODEL
# ==================================================
else:

    st.subheader("📊 Mode Evaluasi & Perbandingan Model")
    st.markdown("Menampilkan grafik hasil training (Accuracy & Loss) untuk analisis performa model.")

    selected_eval_model = st.selectbox("Pilih Model untuk Evaluasi", list(models.keys()))

    try:
        history = np.load(f"history/{selected_eval_model.replace(' ', '_')}_history.npy", allow_pickle=True).item()

        epochs = range(1, len(history['accuracy']) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()

        ax1.plot(epochs, history['accuracy'], label='Accuracy', color='#22c55e')
        ax2.plot(epochs, history['loss'], label='Loss', color='#ef4444', linestyle='--')

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax2.set_ylabel("Loss")

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        st.pyplot(fig)

        st.success("Grafik ini digunakan untuk analisis performa model pada bab eksperimen")

    except:
        st.warning("File history model belum tersedia. Pastikan history training sudah disimpan.")

# ==================================================
# Footer
# ==================================================
st.markdown("---")
st.caption("© 2025 Facial Expression Classification – by Qoid Rif'at")
