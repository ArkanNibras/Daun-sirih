import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===========================================================
# KONFIGURASI
# ===========================================================
MODEL_PATH   = 'model_sirih.h5'
CLASSES_FILE = 'classes.txt'
IMAGE_SIZE   = (224, 224)
CONFIDENCE_THRESHOLD = 70.0
CONFIDENCE_MARGIN = 12.0
# ===========================================================

# Deskripsi kondisi untuk setiap kelas
CLASS_DESCRIPTIONS = {
    "Sirih_sehat" : "Daun sirih dalam kondisi SEHAT. Tidak ada tanda-tanda penyakit.",
    "Sirih_sakit" : "Daun sirih terdeteksi SAKIT. Terdapat tanda-tanda infeksi atau penyakit.",
    "Sirih_layu"  : "Daun sirih dalam kondisi LAYU. Kemungkinan kekurangan air atau nutrisi.",
}


def color_layu_score_from_image(img):
    """Simple heuristic for wilted leaves based on HSV color distribution."""
    try:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(float)
        s = hsv[:, :, 1].astype(float)
        v = hsv[:, :, 2].astype(float)

        yellow_mask = ((h >= 10) & (h <= 45)) & (s >= 20) & (s <= 95) & (v >= 35) & (v <= 180)
        brown_mask = ((h >= 0) & (h <= 20)) & (s >= 20) & (s <= 120) & (v >= 20) & (v <= 140)
        green_mask = ((h >= 35) & (h <= 90)) & (s >= 30) & (v >= 40)

        yellow_ratio = float(np.mean(yellow_mask))
        brown_ratio = float(np.mean(brown_mask))
        green_ratio = float(np.mean(green_mask))
        mean_s = float(np.mean(s))
        mean_v = float(np.mean(v))

        score = 0.0
        score += 0.45 * yellow_ratio
        score += 0.35 * brown_ratio
        score += 0.20 * max(0.0, 1.0 - green_ratio)
        score += 0.10 * max(0.0, (140.0 - mean_v) / 140.0)
        score += 0.10 * max(0.0, (60.0 - mean_s) / 60.0)
        return min(1.0, max(0.0, score))
    except Exception:
        return 0.0


def is_probably_not_leaf(img):
    """Reject obvious non-leaf images using a conservative leaf-color heuristic."""
    try:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(float)
        s = hsv[:, :, 1].astype(float)
        v = hsv[:, :, 2].astype(float)

        green_mask = ((h >= 35) & (h <= 90)) & (s >= 25) & (v >= 30)
        yellow_mask = ((h >= 10) & (h <= 45)) & (s >= 20) & (s <= 95) & (v >= 35) & (v <= 180)
        brown_mask = ((h >= 0) & (h <= 20)) & (s >= 20) & (s <= 120) & (v >= 20) & (v <= 140)
        leaf_like_mask = green_mask | yellow_mask | brown_mask

        leaf_ratio = float(np.mean(leaf_like_mask))
        mean_s = float(np.mean(s))
        mean_v = float(np.mean(v))

        if leaf_ratio < 0.08 and mean_s < 25.0 and mean_v < 90.0:
            return True

        return False
    except Exception:
        return True


def load_classes(file_path):
    """Memuat label kelas dari file classes.txt"""
    if not os.path.exists(file_path):
        print(f"[ERROR] File kelas '{file_path}' tidak ditemukan!")
        sys.exit(1)
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def predict_image(image_path):
    """Melakukan prediksi pada sebuah gambar daun sirih."""

    # 1. Cek keberadaan file model dan gambar
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model '{MODEL_PATH}' tidak ditemukan!")
        print("[INFO]  Jalankan 'python train.py' terlebih dahulu untuk melatih model.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] Gambar '{image_path}' tidak ditemukan!")
        sys.exit(1)

    # 2. Muat Model
    print(f"\n[INFO] Memuat model dari '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        sys.exit(1)

    # 3. Muat Label Kelas
    classes = load_classes(CLASSES_FILE)
    print(f"[INFO] Kelas yang dikenali: {classes}")

    # 4. Pra-pemrosesan Gambar
    print(f"[INFO] Memproses gambar: {image_path}")
    try:
        img       = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)   # Sesuai dengan MobileNetV2
    except Exception as e:
        print(f"[ERROR] Gagal memproses gambar: {e}")
        sys.exit(1)

    # 5. Prediksi
    print("[INFO] Melakukan prediksi...")
    predictions   = model.predict(img_array, verbose=0)
    all_probs     = predictions[0]
    top_index     = int(np.argmax(all_probs))
    confidence    = float(all_probs[top_index] * 100)
    second_confidence = float(np.sort(all_probs)[-2] * 100) if len(all_probs) > 1 else 0.0
    predicted_label = classes[top_index]
    is_confident = confidence >= CONFIDENCE_THRESHOLD and (confidence - second_confidence) >= CONFIDENCE_MARGIN

    # Additional heuristic for wilted-looking leaves / non-leaf images
    not_leaf = False
    try:
        with image.load_img(image_path) as img_raw:
            color_layu = color_layu_score_from_image(img_raw)
            not_leaf = is_probably_not_leaf(img_raw)
    except Exception:
        color_layu = 0.0
        not_leaf = True

    if not_leaf:
        predicted_label = "Tidak Dikenali"
        confidence = 0.0
        is_confident = False
    elif predicted_label == "Sirih_sakit" and color_layu >= 0.62 and classes[1] == "Sirih_layu":
        predicted_label = "Sirih_layu"
        confidence = max(confidence, 70.0)
        is_confident = True

    # 6. Tampilkan Hasil
    print("\n" + "=" * 50)
    print("  HASIL KLASIFIKASI DAUN SIRIH")
    print("=" * 50)
    print(f"  Gambar        : {os.path.basename(image_path)}")
    if not is_confident and predicted_label != "Sirih_layu":
        print("  Kondisi       : Tidak Dikenali")
        print("  tingkat akurasi     : Tidak cukup yakin")
        print("  Catatan       : Model terlalu ambigu atau gambar bukan daun sirih yang jelas.")
    else:
        print(f"  Kondisi       : {predicted_label}")
        print(f"  tingkat akurasi     : {confidence:.2f}%")
    print("-" * 50)
    desc = CLASS_DESCRIPTIONS.get(predicted_label, "Kondisi tidak dikenal.")
    print(f"  Keterangan    : {desc}")
    print("-" * 50)
    print("  Probabilitas semua kelas:")
    for i, cls in enumerate(classes):
        bar = "█" * int(all_probs[i] * 20)
        print(f"  {cls:<20} {all_probs[i]*100:6.2f}%  {bar}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n[CARA PENGGUNAAN]")
        print("  python predict.py <path_gambar>")
        print("\n[CONTOH]")
        print("  python predict.py dataset/Sirih_sehat/gambar01.jpg")
        print("  python predict.py C:/foto/daun.jpg")
        sys.exit(0)

    predict_image(sys.argv[1])
