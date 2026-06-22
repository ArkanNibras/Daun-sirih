# 🌿 BetelVision AI — Deteksi Daun Sirih

Aplikasi kecerdasan buatan untuk mendeteksi kondisi daun sirih menggunakan **MobileNetV2** dan **TensorFlow**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📸 Fitur Utama

| Fitur | Keterangan |
|---|---|
| 🔬 Deteksi Gambar | Upload foto daun sirih dan dapatkan hasil deteksi instan |
| 📷 Kamera Real-Time | Deteksi langsung melalui webcam (versi desktop) |
| 📊 Probabilitas Kelas | Lihat tingkat akurasi untuk setiap kelas |
| 🛡️ Anti False-Positive | Sistem menolak objek yang bukan daun sirih |

---

## 🧩 Kelas Deteksi

| Kelas | Deskripsi |
|---|---|
| ✅ `Sirih_sehat` | Daun hijau segar tanpa bercak, tekstur normal |
| 🚨 `Sirih_sakit` | Terdeteksi infeksi bakteri atau jamur |
| ⚠️ `Sirih_layu` | Daun layu karena kekurangan air atau nutrisi |

---

## 🚀 Cara Menjalankan

### A. Versi Web (Streamlit)

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Jalankan
streamlit run streamlit_app.py
```

### B. Versi Desktop (CustomTkinter)

```bash
# Install semua dependencies
pip install -r requirements.txt

# Jalankan aplikasi desktop
python app.py
```

---

## 📁 Struktur Proyek

```
Tubes/
├── app.py                  # Aplikasi desktop (CustomTkinter + kamera)
├── streamlit_app.py        # Aplikasi web (Streamlit)
├── train.py                # Script training model
├── predict.py              # Script prediksi CLI
├── model_sirih.h5          # Model terlatih (MobileNetV2)
├── classes.txt             # Label kelas
├── requirements.txt        # Dependencies desktop
├── requirements_streamlit.txt  # Dependencies web/cloud
└── dataset/                # Dataset gambar (tidak di-upload)
    ├── Sirih_sehat/
    ├── Sirih_sakit/
    └── Sirih_layu/
```

---

## ☁️ Deploy ke Streamlit Cloud

1. **Push ke GitHub** repositori ini
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **"New app"**
4. Pilih repositori dan set:
   - **Main file path:** `streamlit_app.py`
   - **Python version:** `3.9`
5. Klik **"Deploy!"**

> ⚠️ **Catatan:** Pastikan file `model_sirih.h5` dan `classes.txt` ikut ter-push ke GitHub (tidak ada di `.gitignore`).

---

## 🧠 Teknologi

- **Model:** MobileNetV2 (Transfer Learning dari ImageNet)
- **Framework:** TensorFlow 2.x / Keras
- **Web UI:** Streamlit
- **Desktop UI:** CustomTkinter
- **Computer Vision:** OpenCV + HSV heuristics
- **Image Processing:** Pillow

---

## 📊 Cara Melatih Ulang Model

```bash
# Siapkan dataset di folder:
# dataset/Sirih_sehat/   ← foto daun sehat
# dataset/Sirih_sakit/   ← foto daun sakit
# dataset/Sirih_layu/    ← foto daun layu

# Jalankan training
python train.py
```

---

## 👨‍💻 Pengembang

Dibuat sebagai proyek mata kuliah **Computer Vision** — Semester 4.

> Model dilatih dengan augmentasi data (rotasi, flip, brightness) untuk meningkatkan generalisasi.
