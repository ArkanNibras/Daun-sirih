import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ===========================================================
# KONFIGURASI - Sesuaikan bagian ini sesuai kebutuhan Anda
# ===========================================================
DATASET_PATH = 'dataset'
MODEL_NAME   = 'model_sirih.h5'
CLASSES_FILE = 'classes.txt'
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32  # Dinaikkan untuk kestabilan gradient
EPOCHS       = 50  # Lebih banyak, tapi ada EarlyStopping sebagai rem
# ===========================================================

def train_model():
    print("=" * 50)
    print("  KLASIFIKASI PENYAKIT DAUN SIRIH")
    print("  Proses Training Model CNN (MobileNetV2)")
    print("=" * 50)

    # Pastikan folder dataset ada
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Folder dataset '{DATASET_PATH}' tidak ditemukan!")
        print("Silakan buat folder 'dataset' dan isi dengan subfolder per kelas.")
        print("Contoh struktur:")
        print("  dataset/")
        print("    Sirih_sehat/   (isi dengan gambar)")
        print("    Sirih_sakit/   (isi dengan gambar)")
        print("    Sirih_layu/    (isi dengan gambar)")
        return

    print(f"\n[INFO] Memuat data dari folder: {DATASET_PATH}")

    # ---- Augmentasi & Preprocessing Data ----
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,      # Lebih luas
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,         # Lebih fleksibel untuk jarak jauh/dekat
        brightness_range=[0.7, 1.3], # Mengenali daun di tempat gelap/terang
        horizontal_flip=True,
        vertical_flip=True,     # Daun bisa terbalik
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    print(f"[INFO] Ditemukan {num_classes} kelas: {class_names}")
    
    # Hitung Class Weights (Penting jika jumlah foto tiap folder tidak sama)
    from sklearn.utils import class_weight
    labels = train_generator.classes
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(weights))
    print(f"[INFO] Class Weights: {class_weights}")

    # Simpan daftar kelas ke file
    with open(CLASSES_FILE, 'w') as f:
        for label in class_names:
            f.write(f"{label}\n")
    print(f"[INFO] Label kelas disimpan di '{CLASSES_FILE}'")

    # ---- Membangun Model (Transfer Learning MobileNetV2) ----
    print("\n[INFO] Membangun model MobileNetV2...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Freeze base model
    base_model.trainable = False

    # Tambah lapisan klasifikasi kustom
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ---- Callbacks ----
    callbacks = [
        ModelCheckpoint(
            MODEL_NAME,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # ---- Training Phase 1: Transfer Learning ----
    print("\n[INFO] Memulai Training Tahap 1 (Freeze Base Model)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights # Gunakan bobot kelas
    )

    print(f"\n[INFO] Model tahap 1 selesai.")

    # ---- Fine-tuning (opsional): Unfreeze sebagian layer ----
    print("\n[INFO] Fine-tuning: Membuka lapisan terakhir base model...")
    base_model.trainable = True
    # Hanya latih 30 layer terakhir
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        epochs=20, # Tambah sedikit untuk fine-tuning
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # ---- Plot Grafik ----
    plt.figure(figsize=(14, 5))

    # Gabungkan history
    acc     = history.history['accuracy']     + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss    = history.history['loss']         + history_fine.history['loss']
    val_loss= history.history['val_loss']     + history_fine.history['val_loss']

    epochs_range = range(len(acc))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc,     label='Train Accuracy', color='royalblue')
    plt.plot(epochs_range, val_acc, label='Val Accuracy',   color='darkorange')
    plt.legend()
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss,     label='Train Loss', color='royalblue')
    plt.plot(epochs_range, val_loss, label='Val Loss',   color='darkorange')
    plt.legend()
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_report.png', dpi=150)
    print("[INFO] Grafik training disimpan di 'training_report.png'")

    # Evaluasi akhir
    print("\n" + "=" * 50)
    print("  TRAINING SELESAI!")
    val_loss_final, val_acc_final = model.evaluate(val_generator, verbose=0)
    print(f"  Akurasi Validasi Akhir : {val_acc_final * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
