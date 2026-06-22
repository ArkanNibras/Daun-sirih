import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import shutil
import json
import time
import threading
from collections import deque

# ===========================================================
# KONFIGURASI
# ===========================================================
MODEL_PATH   = 'model_sirih.h5'
CLASSES_FILE = 'classes.txt'
IMAGE_SIZE   = (224, 224)
CONFIDENCE_THRESHOLD = 70.0  # Persentase minimum untuk dianggap valid
CONFIDENCE_MARGIN = 12.0  # Selisih minimum dengan prediksi kedua agar tidak ambigu

# Warna per kelas (BGR untuk OpenCV)
CLASS_COLORS = {
    "Sirih_sehat" : (50, 205, 50),    # Hijau
    "Sirih_sakit" : (0, 60, 255),     # Merah
    "Sirih_layu"  : (0, 165, 255),    # Oranye
}

CLASS_DESCRIPTIONS = {
    "Sirih_sehat" : "Daun SEHAT - Tidak ada tanda penyakit",
    "Sirih_sakit" : "Daun SAKIT - Terdeteksi infeksi/penyakit",
    "Sirih_layu"  : "Daun LAYU - Kekurangan air/nutrisi",
}

# Konfigurasi CustomTkinter
ctk.set_appearance_mode("Light")  # Mengubah menjadi terang (Light Mode)
ctk.set_default_color_theme("green")

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


class SirihApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Daun Sirih")
        self.root.geometry("1100x800")
        self.root.minsize(900, 600)
        
        # Resources
        self.model = None
        self.classes = []
        self.last_prediction_info = None
        self.load_resources()
        
        # Camera state
        self._cam_cap = None
        self._cam_running = False
        self._cam_thread = None
        self._cam_lock = threading.Lock()
        self._pred_buffer = deque(maxlen=8)   # rolling-average smoothing
        self._last_cam_frame = None           # PIL image for capture
        self._cam_result_label = ""          # latest label string
        self._cam_result_conf  = 0.0         # latest confidence
        
        # UI Setup
        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(side="top", fill="both", expand=True)
        
        self.frames = {}
        self.setup_main_menu()
        self.setup_classifier_ui()
        self.setup_camera_ui()
        self.setup_about_ui()
        
        self.show_frame("MainMenu")
        
        # Handle close
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def load_resources(self):
        try:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("Error", f"Model '{MODEL_PATH}' tidak ditemukan!")
                sys.exit(1)
            
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            if not os.path.exists(CLASSES_FILE):
                messagebox.showerror("Error", f"File '{CLASSES_FILE}' tidak ditemukan!")
                sys.exit(1)
                
            with open(CLASSES_FILE, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat sistem: {e}")
            sys.exit(1)

    def show_frame(self, page_name):
        # Stop camera when leaving Camera page
        if page_name != "Camera":
            self._stop_camera()

        frame = self.frames[page_name]
        frame.tkraise()
        
        # Hide/Show frames
        for name, f in self.frames.items():
            if name == page_name:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()

        # Auto-start camera when entering Camera page
        if page_name == "Camera":
            self.root.after(200, self._start_camera)

    def setup_main_menu(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        self.frames["MainMenu"] = frame
        
        # Center container
        center_frame = ctk.CTkFrame(frame, fg_color=("white", "#1e1e24"), corner_radius=20)
        center_frame.pack(pady=60, padx=60, expand=True)
        
        # Split into two columns: Left for Logo, Right for Text & Buttons
        left_frame = ctk.CTkFrame(center_frame, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=(40, 20), pady=40)
        
        right_frame = ctk.CTkFrame(center_frame, fg_color="transparent")
        right_frame.pack(side="right", fill="both", expand=True, padx=(20, 40), pady=40)
        
        # --- Left Side: Big Logo ---
        if os.path.exists("logo.png"):
            logo_img = ctk.CTkImage(light_image=Image.open("logo.png"), size=(280, 280))
            logo_label = ctk.CTkLabel(left_frame, image=logo_img, text="")
            logo_label.pack(expand=True)
        else:
            # Fallback if logo not found
            fallback_label = ctk.CTkLabel(left_frame, text="BetelVision\nLogo", font=ctk.CTkFont(size=30, weight="bold"), text_color="gray")
            fallback_label.pack(expand=True)
        
        # --- Right Side: Title, Subtitle, Buttons ---
        title = ctk.CTkLabel(right_frame, text="Daun Sirih", font=ctk.CTkFont(family="Segoe UI", size=48, weight="bold"), text_color=("#27ae60", "#2ecc71"))
        title.pack(anchor="center", pady=(20, 5))
        
        subtitle = ctk.CTkLabel(right_frame, text="Deteksi Cerdas Penyakit Daun Sirih", font=ctk.CTkFont(family="Segoe UI", size=18), text_color=("gray40", "#a0a0a0"))
        subtitle.pack(anchor="center", pady=(0, 40))
        
        btn_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        btn_frame.pack(expand=True)

        ctk.CTkButton(btn_frame, text="Mulai Klasifikasi", font=ctk.CTkFont(size=16, weight="bold"), width=240, height=50, corner_radius=8, 
                      command=lambda: self.show_frame("Classifier")).pack(pady=10)

        ctk.CTkButton(btn_frame, text="📷  Deteksi Kamera", font=ctk.CTkFont(size=16, weight="bold"), width=240, height=50, corner_radius=8,
                      fg_color="#8e44ad", hover_color="#6c3483",
                      command=lambda: self.show_frame("Camera")).pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="Tentang Aplikasi", font=ctk.CTkFont(size=16, weight="bold"), width=240, height=50, corner_radius=8,
                      fg_color="#3498db", hover_color="#2980b9",
                      command=lambda: self.show_frame("About")).pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="Keluar", font=ctk.CTkFont(size=16, weight="bold"), width=240, height=50, corner_radius=8,
                      fg_color="#e74c3c", hover_color="#c0392b",
                      command=self.on_exit).pack(pady=10)
                      
        # Footer
        footer = ctk.CTkLabel(frame, text="© 2024 BetelVision AI - Computer Vision AI", font=ctk.CTkFont(size=12), text_color=("gray60", "gray40"))
        footer.pack(side="bottom", pady=20)

    def setup_classifier_ui(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        self.frames["Classifier"] = frame

        # Layout: Sidebar (Left), Main Content (Center), Info (Right)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)

        # 1. Sidebar (warna abu-abu sangat terang)
        sidebar = ctk.CTkFrame(frame, width=250, corner_radius=0, fg_color=("gray95", "#1a1a1e"))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.pack_propagate(False)

        # Brand / Logo di sidebar
        brand_label = ctk.CTkLabel(sidebar, text="Daun Sirih", font=ctk.CTkFont(size=24, weight="bold"), text_color=("#27ae60", "#2ecc71"))
        brand_label.pack(pady=(30, 10))
        
        divider = ctk.CTkFrame(sidebar, height=2, fg_color=("gray85", "#2c2c34"))
        divider.pack(fill="x", padx=20, pady=(0, 30))

        # Sidebar Buttons
        self.btn_upload = ctk.CTkButton(sidebar, text="📁  Upload Gambar", font=ctk.CTkFont(size=14, weight="bold"),
                                       height=45, anchor="w", fg_color="transparent", text_color=("gray10", "#e0e0e0"),
                                       hover_color=("gray80", "#2c2c34"), command=self.upload_image)
        self.btn_upload.pack(fill="x", padx=15, pady=5)

        ctk.CTkButton(sidebar, text="📷  Buka Kamera", font=ctk.CTkFont(size=14, weight="bold"),
                      height=45, anchor="w", fg_color="transparent", text_color=("gray10", "#e0e0e0"),
                      hover_color=("gray80", "#2c2c34"), command=lambda: self.show_frame("Camera")).pack(fill="x", padx=15, pady=5)

        # Save result button for debugging/misclassified samples
        self.btn_save = ctk.CTkButton(sidebar, text="💾  Simpan Hasil", font=ctk.CTkFont(size=14, weight="bold"),
                                      height=45, anchor="w", fg_color="transparent", text_color=("gray10", "#e0e0e0"),
                                      hover_color=("gray80", "#2c2c34"), command=self.save_last_result)
        self.btn_save.pack(fill="x", padx=15, pady=5)

        # Spacer
        spacer = ctk.CTkFrame(sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        btn_back = ctk.CTkButton(sidebar, text="←  Kembali", font=ctk.CTkFont(size=14, weight="bold"),
                                 height=45, anchor="w", fg_color="transparent", text_color=("gray10", "#e0e0e0"),
                                 hover_color=("gray80", "#2c2c34"), command=lambda: self.show_frame("MainMenu"))
        btn_back.pack(fill="x", padx=15, pady=20)


        # 2. Main Content
        main_area = ctk.CTkFrame(frame, fg_color="transparent")
        main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Header Info
        self.header_label = ctk.CTkLabel(main_area, text="Panel Diagnostik", font=ctk.CTkFont(size=28, weight="bold"), text_color=("black", "white"))
        self.header_label.pack(anchor="w", pady=(0, 20))

        # Display Area (Camera/Image)
        self.display_container = ctk.CTkFrame(main_area, fg_color=("gray90", "#1e1e24"), corner_radius=15)
        self.display_container.pack(fill="both", expand=True)

        self.display_label = tk.Label(self.display_container, bg="#e5e5e5")
        self.display_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Set default logo inside display area if exists
        if os.path.exists("logo.png"):
            img = Image.open("logo.png").resize((200, 200), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.display_label.imgtk = imgtk
            self.display_label.config(image=imgtk)

        # Status Bar Area (Bottom)
        self.status_frame = ctk.CTkFrame(main_area, height=80, fg_color=("white", "#1e1e24"), corner_radius=15)
        self.status_frame.pack(fill="x", pady=(20, 0))
        self.status_frame.pack_propagate(False)

        self.info_label = ctk.CTkLabel(self.status_frame, text="Siap digunakan. Silakan upload gambar untuk klasifikasi.", 
                                       font=ctk.CTkFont(size=18, weight="bold"), text_color=("gray40", "#a0a0a0"))
        self.info_label.pack(expand=True, fill="both")
        
        # 3. Right Information Panel
        right_panel = ctk.CTkFrame(frame, width=280, fg_color=("white", "#1e1e24"), corner_radius=15)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 20), pady=20)
        right_panel.pack_propagate(False)

        info_header = ctk.CTkLabel(right_panel, text="Panduan Deteksi", font=ctk.CTkFont(size=18, weight="bold"), text_color=("black", "white"))
        info_header.pack(pady=(20, 10), padx=20, anchor="w")
        
        div2 = ctk.CTkFrame(right_panel, height=2, fg_color=("gray85", "#2c2c34"))
        div2.pack(fill="x", padx=20, pady=(0, 20))
        
        # Legend items and progress bars
        self.class_progress_bars = {}
        self.class_progress_labels = {}

        def add_legend(parent, color_hex, title, desc, class_key=None):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=10)
            
            # Header row (Color box + Title + Percentage)
            header_row = ctk.CTkFrame(row, fg_color="transparent")
            header_row.pack(fill="x")
            
            color_box = ctk.CTkFrame(header_row, width=18, height=18, corner_radius=4, fg_color=color_hex)
            color_box.pack(side="left", padx=(0, 10))
            
            ctk.CTkLabel(header_row, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color=("black", "white")).pack(side="left")
            
            if class_key:
                pct_label = ctk.CTkLabel(header_row, text="0.0%", font=ctk.CTkFont(size=14, weight="bold"), text_color=("gray50", "gray50"))
                pct_label.pack(side="right")
                
                # Progress bar row
                progress = ctk.CTkProgressBar(row, height=8, fg_color=("gray85", "#2c2c34"), progress_color=color_hex)
                progress.pack(fill="x", pady=(8, 0))
                progress.set(0)
                
                self.class_progress_bars[class_key] = progress
                self.class_progress_labels[class_key] = pct_label
            else:
                ctk.CTkLabel(row, text=desc, font=ctk.CTkFont(size=12), text_color=("gray50", "gray50"), justify="left", wraplength=180).pack(anchor="w", pady=(4,0))

        add_legend(right_panel, "#2ecc71", "Daun Sehat", "Tidak ada bercak atau penyakit.", "Sirih_sehat")
        add_legend(right_panel, "#e74c3c", "Daun Sakit", "Terdeteksi infeksi bakteri/jamur.", "Sirih_sakit")
        add_legend(right_panel, "#f39c12", "Daun Layu", "Daun layu atau kurang nutrisi.", "Sirih_layu")
        add_legend(right_panel, "#f1c40f", "Tidak Dikenali", "Sistem tidak yakin ini daun sirih.", None)


    # ===========================================================
    # FITUR KAMERA
    # ===========================================================

    def setup_camera_ui(self):
        """Buat frame UI Kamera."""
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        self.frames["Camera"] = frame

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # ── Sidebar kiri ──────────────────────────────────────────
        sidebar = ctk.CTkFrame(frame, width=250, corner_radius=0, fg_color=("gray95", "#1a1a1e"))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Daun Sirih",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=("#8e44ad", "#a855f7")).pack(pady=(30, 10))

        ctk.CTkFrame(sidebar, height=2, fg_color=("gray85", "#2c2c34")).pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(sidebar, text="📷  DETEKSI KAMERA",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=("#8e44ad", "#a855f7")).pack(anchor="w", padx=20, pady=(0, 20))

        # Tombol start / stop
        self._cam_toggle_btn = ctk.CTkButton(
            sidebar, text="⏹  Stop Kamera",
            font=ctk.CTkFont(size=14, weight="bold"), height=45,
            anchor="w", fg_color="#e74c3c", hover_color="#c0392b",
            command=self._toggle_camera)
        self._cam_toggle_btn.pack(fill="x", padx=15, pady=5)

        # Tombol capture
        ctk.CTkButton(
            sidebar, text="📸  Ambil Foto",
            font=ctk.CTkFont(size=14, weight="bold"), height=45,
            anchor="w", fg_color="#27ae60", hover_color="#219150",
            command=self._capture_frame).pack(fill="x", padx=15, pady=5)

        # ── Keterangan kelas ──────────────────────────────────────
        ctk.CTkFrame(sidebar, height=2, fg_color=("gray85", "#2c2c34")).pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(sidebar, text="Keterangan",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=("gray40", "gray60")).pack(anchor="w", padx=20)

        legends = [
            ("#2ecc71", "● Sehat"),
            ("#e74c3c", "● Sakit"),
            ("#f39c12", "● Layu"),
            ("#f1c40f", "● Tidak Dikenali"),
        ]
        for color, text in legends:
            ctk.CTkLabel(sidebar, text=text,
                         font=ctk.CTkFont(size=13),
                         text_color=color).pack(anchor="w", padx=30, pady=2)

        spacer = ctk.CTkFrame(sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        ctk.CTkButton(
            sidebar, text="←  Kembali",
            font=ctk.CTkFont(size=14, weight="bold"), height=45,
            anchor="w", fg_color="transparent",
            text_color=("gray10", "#e0e0e0"),
            hover_color=("gray80", "#2c2c34"),
            command=lambda: self.show_frame("MainMenu")
        ).pack(fill="x", padx=15, pady=20)

        # ── Area utama ────────────────────────────────────────────
        main_area = ctk.CTkFrame(frame, fg_color="transparent")
        main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main_area.rowconfigure(1, weight=1)
        main_area.columnconfigure(0, weight=1)

        # Header
        header_row = ctk.CTkFrame(main_area, fg_color="transparent")
        header_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(header_row, text="Deteksi Real-Time Kamera",
                     font=ctk.CTkFont(size=26, weight="bold"),
                     text_color=("black", "white")).pack(side="left")

        # Status indicator (dot)
        self._cam_status_dot = ctk.CTkLabel(
            header_row, text="⬤  LIVE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e74c3c")
        self._cam_status_dot.pack(side="right", padx=10)

        # Feed container
        feed_container = ctk.CTkFrame(main_area,
                                      fg_color=("#1a1a2e", "#0d0d1a"),
                                      corner_radius=16)
        feed_container.grid(row=1, column=0, sticky="nsew")

        self._cam_video_label = tk.Label(feed_container, bg="#1a1a2e")
        self._cam_video_label.pack(expand=True, fill="both", padx=8, pady=8)

        # Result bar bawah
        result_bar = ctk.CTkFrame(main_area, height=90,
                                  fg_color=("white", "#1e1e24"),
                                  corner_radius=14)
        result_bar.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        result_bar.pack_propagate(False)
        result_bar.columnconfigure(0, weight=1)
        result_bar.columnconfigure(1, weight=0)

        self._cam_result_text = ctk.CTkLabel(
            result_bar,
            text="Menunggu kamera…",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("gray40", "#a0a0a0"))
        self._cam_result_text.pack(side="left", padx=24, expand=True, fill="both")

        conf_frame = ctk.CTkFrame(result_bar, fg_color="transparent")
        conf_frame.pack(side="right", padx=20, pady=10)

        ctk.CTkLabel(conf_frame, text="Akurasi",
                     font=ctk.CTkFont(size=11),
                     text_color=("gray50", "gray60")).pack()

        self._cam_conf_bar = ctk.CTkProgressBar(
            conf_frame, width=160, height=14,
            fg_color=("gray85", "#2c2c34"),
            progress_color="#27ae60")
        self._cam_conf_bar.pack(pady=(4, 0))
        self._cam_conf_bar.set(0)

        self._cam_conf_pct = ctk.CTkLabel(
            conf_frame, text="0.0%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("gray50", "gray60"))
        self._cam_conf_pct.pack()

    # ── Camera engine ──────────────────────────────────────────────

    def _toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self._cam_running:
            return
        # Try multiple camera indices
        cap = None
        for idx in range(3):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if c.isOpened():
                cap = c
                break
            c.release()
        if cap is None:
            messagebox.showerror("Kamera", "Tidak dapat membuka kamera.\nPastikan kamera terhubung dan tidak digunakan aplikasi lain.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)

        with self._cam_lock:
            self._cam_cap = cap
            self._cam_running = True
            self._pred_buffer.clear()

        self._cam_toggle_btn.configure(text="⏹  Stop Kamera",
                                       fg_color="#e74c3c",
                                       hover_color="#c0392b")
        self._cam_status_dot.configure(text="⬤  LIVE", text_color="#2ecc71")

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _stop_camera(self):
        with self._cam_lock:
            self._cam_running = False
            if self._cam_cap:
                self._cam_cap.release()
                self._cam_cap = None
        # Reset UI safely
        try:
            self._cam_toggle_btn.configure(text="▶  Mulai Kamera",
                                           fg_color="#27ae60",
                                           hover_color="#219150")
            self._cam_status_dot.configure(text="⬤  OFFLINE", text_color="#e74c3c")
            self._cam_result_text.configure(text="Kamera dihentikan.",
                                            text_color=("gray40", "#a0a0a0"))
            self._cam_conf_bar.set(0)
            self._cam_conf_pct.configure(text="0.0%")
        except Exception:
            pass

    def _camera_loop(self):
        """Thread: baca frame, prediksi, kirim ke UI."""
        frame_count = 0
        while True:
            with self._cam_lock:
                if not self._cam_running or self._cam_cap is None:
                    break
                cap = self._cam_cap

            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            # Convert for display
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)

            # Run prediction every 3 frames for performance
            if frame_count % 3 == 0:
                label, conf, color_hex = self._predict_frame(bgr)
                self._pred_buffer.append((label, conf, color_hex))

            frame_count += 1

            # Smoothed result from buffer
            if self._pred_buffer:
                smooth_label, smooth_conf, smooth_color = self._smooth_prediction()
            else:
                smooth_label, smooth_conf, smooth_color = "Mendeteksi…", 0.0, "#f1c40f"

            # Draw overlay on frame
            annotated = self._draw_overlay(bgr.copy(), smooth_label, smooth_conf, smooth_color)
            ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            display_pil = Image.fromarray(ann_rgb)

            # Store raw frame for capture
            self._last_cam_frame = pil_frame
            self._cam_result_label = smooth_label
            self._cam_result_conf  = smooth_conf

            # Schedule UI update on main thread
            self.root.after(0, self._update_camera_ui, display_pil,
                            smooth_label, smooth_conf, smooth_color)

            time.sleep(0.025)  # ~40 fps cap

    # ── Strict frame-level leaf validator ──────────────────────────────

    def _is_frame_leaf_like(self, bgr_frame):
        """
        Returns True only when the frame contains enough leaf-like pixels.
        This is the primary guard against false-positives on non-leaf objects
        (bottles, people, backgrounds, etc.).

        Criteria (all must pass):
        - At least 15% of pixels are "leaf-coloured" (green / yellow / brown)
        - At least 8%  of pixels are specifically GREEN  (fresh-leaf hue)
        - The bright-white region (wall, paper, bottle body) is < 55%
        - Overall mean saturation is at least 18 (not a near-greyscale scene)
        """
        try:
            hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV).astype(float)
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # --- leaf colour masks ---
            green_mask  = ((h >= 35) & (h <= 90)) & (s >= 30) & (v >= 40)
            yellow_mask = ((h >= 10) & (h <= 45)) & (s >= 30) & (s <= 160) & (v >= 50)
            brown_mask  = ((h >=  0) & (h <= 22)) & (s >= 35) & (s <= 160) & (v >= 30) & (v <= 170)
            leaf_mask   = green_mask | yellow_mask | brown_mask

            leaf_ratio  = float(np.mean(leaf_mask))
            green_ratio = float(np.mean(green_mask))

            # bright / washed-out region (bottle surface, bright wall)
            white_mask  = (s < 35) & (v > 175)
            white_ratio = float(np.mean(white_mask))

            mean_s = float(np.mean(s))

            # --- decision ---
            if leaf_ratio  < 0.15:   return False   # too few leaf-coloured pixels
            if green_ratio < 0.08:   return False   # no meaningful green present
            if white_ratio > 0.55:   return False   # scene dominated by bright/white areas
            if mean_s      < 18.0:   return False   # nearly greyscale — not a leaf
            return True
        except Exception:
            return False

    # ── Per-frame prediction ────────────────────────────────────────────

    def _predict_frame(self, bgr_frame):
        """Run the Keras model on a single BGR frame (camera-specific thresholds)."""
        # Camera-specific confidence requirements – stricter than the upload path
        CAM_CONFIDENCE = 80.0   # vs 70% for uploads
        CAM_MARGIN     = 18.0   # vs 12% for uploads

        try:
            # 1. Fast colour-based gate: reject non-leaf scenes before hitting the model
            if not self._is_frame_leaf_like(bgr_frame):
                return "Bukan Daun Sirih", 0.0, "#95a5a6"

            rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.LANCZOS)

            # 2. Additional heuristic check (inherited logic)
            if is_probably_not_leaf(pil_img):
                return "Bukan Daun Sirih", 0.0, "#95a5a6"

            color_layu = color_layu_score_from_image(pil_img)

            # 3. Run the model
            arr = np.array(pil_img, dtype=np.float32)
            arr = np.expand_dims(arr, 0)
            arr = preprocess_input(arr)

            preds     = self.model.predict(arr, verbose=0)[0]
            top_idx   = int(np.argmax(preds))
            top_conf  = float(preds[top_idx]  * 100)
            sec_conf  = float(np.sort(preds)[-2] * 100)
            top_label = self.classes[top_idx]

            # 4. Colour-heuristic layu fallback
            if (top_label == "Sirih_sakit" and color_layu >= 0.62 and
                    len(self.classes) > 1 and self.classes[1] == "Sirih_layu"):
                top_label = "Sirih_layu"
                top_conf  = max(top_conf, 70.0)
                sec_conf  = 0.0          # force confident

            # 5. Confidence gate (stricter for camera)
            if top_conf < CAM_CONFIDENCE or (top_conf - sec_conf) < CAM_MARGIN:
                return "Tidak Yakin", top_conf, "#f1c40f"

            # 6. Map to colour
            if "sehat" in top_label.lower():
                color = "#2ecc71"
            elif "sakit" in top_label.lower():
                color = "#e74c3c"
            elif "layu" in top_label.lower():
                color = "#f39c12"
            else:
                color = "#f1c40f"

            return top_label, top_conf, color

        except Exception:
            return "Error", 0.0, "#e74c3c"

    def _smooth_prediction(self):
        """Weighted majority vote over the rolling prediction buffer."""
        if not self._pred_buffer:
            return "Mendeteksi…", 0.0, "#f1c40f"

        vote_conf   = {}
        vote_color  = {}
        vote_weight = {}
        total = len(self._pred_buffer)

        for i, (lbl, conf, col) in enumerate(self._pred_buffer):
            # More recent frames get higher weight
            w = (i + 1) / total
            vote_conf.setdefault(lbl, []).append(conf)
            vote_color[lbl]   = col
            vote_weight[lbl]  = vote_weight.get(lbl, 0.0) + w

        best = max(vote_weight, key=vote_weight.get)
        avg_conf = float(np.mean(vote_conf[best]))
        return best, avg_conf, vote_color[best]

    def _draw_overlay(self, bgr, label, conf, color_hex):
        """Draw confidence bar and label on a BGR frame."""
        h, w = bgr.shape[:2]
        
        # Semi-transparent top banner
        overlay = bgr.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)

        # Hex → BGR
        def hex2bgr(hx):
            hx = hx.lstrip("#")
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            return (b, g, r)

        col_bgr = hex2bgr(color_hex)

        # Label text
        display_text = f"{label}  {conf:.1f}%"
        cv2.putText(bgr, display_text, (14, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, col_bgr, 2, cv2.LINE_AA)

        # Confidence bar at bottom
        bar_y  = h - 18
        bar_w  = int((conf / 100.0) * w)
        cv2.rectangle(bgr, (0, bar_y), (w, h), (30, 30, 30), -1)
        if bar_w > 0:
            cv2.rectangle(bgr, (0, bar_y), (bar_w, h), col_bgr, -1)

        # Corner bracket guides for framing the leaf
        corner_len = 30
        thick = 3
        cx, cy = w // 2, h // 2
        margin = min(w, h) // 3
        x1, y1, x2, y2 = cx - margin, cy - margin, cx + margin, cy + margin
        for (px, py, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(bgr, (px, py), (px + dx * corner_len, py), col_bgr, thick)
            cv2.line(bgr, (px, py), (px, py + dy * corner_len), col_bgr, thick)

        return bgr

    def _update_camera_ui(self, pil_img, label, conf, color_hex):
        """Update camera widgets on the Tk main thread."""
        if not self._cam_running:
            return
        try:
            # Resize to fit the label
            lw = max(self._cam_video_label.winfo_width(),  320)
            lh = max(self._cam_video_label.winfo_height(), 240)
            display = self.resize_for_display(pil_img, lw - 4, lh - 4)
            imgtk = ImageTk.PhotoImage(image=display)
            self._cam_video_label.imgtk = imgtk
            self._cam_video_label.configure(image=imgtk)

            # Result bar
            self._cam_result_text.configure(text=label, text_color=color_hex)
            self._cam_conf_bar.set(min(conf / 100.0, 1.0))
            self._cam_conf_bar.configure(progress_color=color_hex)
            self._cam_conf_pct.configure(text=f"{conf:.1f}%")
        except Exception:
            pass

    def _capture_frame(self):
        """Save the current frame as an image and classify it."""
        frame = self._last_cam_frame
        if frame is None:
            messagebox.showwarning("Kamera", "Tidak ada frame yang tersedia. Pastikan kamera aktif.")
            return

        try:
            out_dir = os.path.join(os.getcwd(), "misclassified")
            os.makedirs(out_dir, exist_ok=True)
            ts  = int(time.time())
            dst = os.path.join(out_dir, f"capture_{ts}.jpg")
            frame.save(dst, "JPEG", quality=95)

            label = self._cam_result_label or "Tidak Diketahui"
            conf  = self._cam_result_conf

            meta = {
                "saved_at":    ts,
                "source":      "camera_capture",
                "final_label": label,
                "confidence":  round(conf, 2)
            }
            with open(os.path.join(out_dir, f"capture_{ts}.json"), "w") as f:
                json.dump(meta, f, indent=2)

            messagebox.showinfo(
                "Foto Tersimpan",
                f"Hasil deteksi: {label} ({conf:.1f}%)\n"
                f"Gambar disimpan di:\n{dst}"
            )
            try:
                os.startfile(out_dir)
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan foto: {e}")

    def setup_about_ui(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        self.frames["About"] = frame

        header = ctk.CTkLabel(frame, text="Tentang BetelVision AI", font=ctk.CTkFont(size=34, weight="bold"), text_color=("#27ae60", "#2ecc71"))
        header.pack(pady=(30, 10), padx=30, anchor="w")

        subtitle = ctk.CTkLabel(frame, text="Informasi lengkap tentang daun sirih, TensorFlow, dan dataset yang digunakan.", 
                                font=ctk.CTkFont(size=16), text_color=("gray30", "gray70"), wraplength=900, justify="left")
        subtitle.pack(pady=(0, 20), padx=30, anchor="w")

        content_frame = ctk.CTkFrame(frame, fg_color=("#f8f9fa", "#1a1a24"), corner_radius=20)
        content_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        about_text = (
            "BetelVision AI adalah aplikasi deteksi kondisi daun sirih menggunakan model pembelajaran mesin. "
            "Aplikasi ini dirancang untuk membantu petani dan peneliti mendeteksi tanda-tanda awal penyakit atau stres pada tanaman sirih.\n\n"
            "Daun sirih (Piper betle) adalah salah satu tanaman obat yang banyak dibudidayakan di Indonesia. "
            "Daun sirih digunakan secara tradisional untuk kesehatan mulut, pengobatan luka, dan ramuan herbal lain. "
            "Daun sirih yang sehat tampak hijau segar tanpa bercak, sedangkan daun yang sakit dapat menunjukkan tanda bercak gelap, jamur, atau perubahan warna.\n\n"
            "TensorFlow adalah pustaka open-source dari Google untuk membangun dan menjalankan model pembelajaran dalam (deep learning). "
            "Dalam aplikasi ini, TensorFlow digunakan untuk memuat model yang telah dilatih pada data daun sirih, mengolah citra, dan memprediksi kondisi daun secara otomatis.\n\n"
            "Dataset yang digunakan terdiri dari gambar daun sirih terbagi dalam tiga kelas utama: 'Sirih_sehat', 'Sirih_sakit', dan 'Sirih_layu'. "
            "Setiap kelas berisi kumpulan gambar yang mewakili kondisi tersebut. Model dilatih dari dataset ini agar dapat membedakan daun sehat, sakit, dan layu. "
            "Semakin banyak variasi gambar dalam dataset, semakin baik model dalam mengenali kondisi daun sirih di dunia nyata.\n\n"
            "Keterangan kelas pada dataset:\n"
            "- Sirih_sehat: Daun hijau dengan tekstur normal dan tanpa lesion.\n"
            "- Sirih_sakit: Daun menunjukkan bercak, kerusakan, atau gejala infeksi.\n"
            "- Sirih_layu: Daun mengerut, berubah warna, atau tampak kekeringan akibat kurang air/nutrisi.\n\n"
            "Jika Anda ingin meningkatkan akurasi, tambahkan lebih banyak gambar dari masing-masing kelas dan pastikan gambar diambil dalam kondisi terang dan fokus. "
            "Dataset yang kuat membantu TensorFlow mempelajari pola visual yang lebih baik, sehingga hasil deteksi menjadi lebih andal."
        )

        content_text = ctk.CTkTextbox(content_frame, width=920, height=260, corner_radius=18,
                                      font=ctk.CTkFont(size=14), text_color=("#333333", "#dddddd"),
                                      fg_color=("#ffffff", "#1f1f24"), border_width=0)
        content_text.insert("0.0", about_text)
        content_text.configure(state="disabled")
        content_text.pack(padx=20, pady=(20, 10), anchor="center")

        sample_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        sample_frame.pack(fill="both", expand=True, pady=(0, 20), padx=20)

        self.setup_about_samples(sample_frame)

        btn_back = ctk.CTkButton(frame, text="← Kembali ke Menu", font=ctk.CTkFont(size=16, weight="bold"), width=220, height=45,
                                 fg_color="#27ae60", hover_color="#219150",
                                 command=lambda: self.show_frame("MainMenu"))
        btn_back.pack(pady=(0, 30))

    def setup_about_samples(self, parent):
        samples = [
            ("Daun Sehat", "Daun hijau segar tanpa bercak, tekstur normal, dan warna seragam.", "dataset/Sirih_sehat/20260501_075733.jpg"),
            ("Daun Sakit", "Daun dengan bercak gelap atau bagian yang terinfeksi/terluka.", "dataset/Sirih_sakit/20260505_191209.jpg"),
            ("Daun Layu", "Daun mengerut, kusam, atau tampak kering karena kekurangan air/nutrisi.", "dataset/Sirih_layu/20260419_131019.jpg"),
        ]

        self.about_sample_images = []
        for title, desc, img_path in samples:
            card = ctk.CTkFrame(parent, fg_color=("#ffffff", "#2a2a2e"), corner_radius=16)
            card.pack(side="left", expand=True, fill="both", padx=8, pady=8)
            card.pack_propagate(False)
            card.configure(height=280)

            title_label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=16, weight="bold"), text_color=("#2c3e50", "#f5f6fa"))
            title_label.pack(pady=(16, 8), padx=10)

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((260, 160), Image.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.about_sample_images.append(imgtk)
                    img_label = ctk.CTkLabel(card, image=imgtk, text="")
                    img_label.pack(padx=10)
                except Exception:
                    ctk.CTkLabel(card, text="Gambar tidak dapat dimuat.", font=ctk.CTkFont(size=12), text_color=("#7f8c8d", "#bdc3c7"), wraplength=240, justify="center").pack(padx=10, pady=20)
            else:
                ctk.CTkLabel(card, text="Contoh gambar tidak tersedia.", font=ctk.CTkFont(size=12), text_color=("#7f8c8d", "#bdc3c7"), wraplength=240, justify="center").pack(padx=10, pady=20)

            ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=13), text_color=("#34495e", "#dcdde1"), wraplength=260, justify="center").pack(padx=10, pady=(12, 16))

    def resize_for_display(self, img, max_w, max_h):
        # Resize image retaining aspect ratio
        img_w, img_h = img.size
        ratio = min(max_w/img_w, max_h/img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        try:
            # Predict
            img_ai = keras_image.load_img(file_path, target_size=IMAGE_SIZE)
            img_arr = keras_image.img_to_array(img_ai)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)
            
            preds = self.model.predict(img_arr, verbose=0)[0]
            # Top-3 predictions untuk diagnostik lebih baik
            top_k = np.argsort(preds)[-3:][::-1]
            top_items = [(self.classes[i], preds[i] * 100) for i in top_k]

            # Update progress bars
            if hasattr(self, 'class_progress_bars'):
                for i, class_name in enumerate(self.classes):
                    if class_name in self.class_progress_bars:
                        prob = float(preds[i])
                        self.class_progress_bars[class_name].set(prob)
                        self.class_progress_labels[class_name].configure(text=f"{prob*100:.1f}%")

            # Hasil utama berdasarkan probabilitas tertinggi
            top_label, top_conf = top_items[0]
            second_label, second_conf = top_items[1] if len(top_items) > 1 else (top_label, 0.0)
            is_confident = top_conf >= CONFIDENCE_THRESHOLD and (top_conf - second_conf) >= CONFIDENCE_MARGIN

            # Fallback berbasis warna untuk daun yang tampak layu
            color_layu = 0.0
            not_leaf = False
            try:
                with Image.open(file_path) as img_raw:
                    color_layu = color_layu_score_from_image(img_raw)
                    not_leaf = is_probably_not_leaf(img_raw)
            except Exception:
                color_layu = 0.0
                not_leaf = True

            final_label = None
            final_color = None
            if not_leaf:
                final_label = "Objek Tidak Dikenali"
                final_color = ("#d35400", "#f1c40f")
                msg_body = "Gambar ini tampak bukan daun sirih atau terlalu tidak jelas untuk diprediksi.\n\n"
            elif top_label == "Sirih_sakit" and color_layu >= 0.62 and second_label == "Sirih_layu":
                final_label = "Sirih_layu"
                final_color = ("#f39c12", "#f39c12")
                desc = CLASS_DESCRIPTIONS.get(final_label, "")
                msg_body = f"Kondisi: {final_label}\ntingkat akurasi: {top_conf:.2f}% (fallback warna)\n\n{desc}\n\n"
            elif not is_confident:
                final_label = "Objek Tidak Dikenali"
                final_color = ("#d35400", "#f1c40f")
                msg_body = "Model tidak yakin karena gambar terlalu ambigu, bukan daun sirih yang jelas, atau terlalu mirip dengan kelas lain.\n\n"
            else:
                final_label = f"{top_label} ({top_conf:.1f}%)"
                final_color = ("#27ae60", "#2ecc71") if "sehat" in top_label.lower() else ("#c0392b", "#e74c3c")
                desc = CLASS_DESCRIPTIONS.get(top_label, "")
                msg_body = f"Kondisi: {top_label}\ntingkat akurasi: {top_conf:.2f}%\n\n{desc}\n\n"

            # Tambahkan informasi Top-3 ke body pesan
            msg_body += "Top-3 prediksi:\n"
            for name, p in top_items:
                msg_body += f" - {name}: {p:.2f}%\n"

            # Simpan info terakhir untuk kemungkinan penyimpanan/debug
            try:
                self.last_prediction_info = {
                    'file_path': file_path,
                    'top3': [{'label': n, 'prob': float(p)} for n, p in top_items],
                    'final_label': final_label,
                    'timestamp': time.time()
                }
            except Exception:
                self.last_prediction_info = None

            # Show in GUI
            display_img = Image.open(file_path)
            max_w = self.display_container.winfo_width()
            max_h = self.display_container.winfo_height()
            if max_w > 10 and max_h > 10:
                display_img = self.resize_for_display(display_img, max_w - 20, max_h - 20)

            imgtk = ImageTk.PhotoImage(image=display_img)
            self.display_label.imgtk = imgtk
            self.display_label.configure(image=imgtk)
            
            self.info_label.configure(text=final_label, text_color=final_color)
            messagebox.showinfo("Hasil Klasifikasi", msg_body)
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memproses gambar: {e}")

    def save_last_result(self):
        """Save the last prediction and image to 'misclassified/' for analysis."""
        info = getattr(self, 'last_prediction_info', None)
        if not info:
            messagebox.showwarning("Tidak ada data", "Tidak ada hasil prediksi yang dapat disimpan.")
            return

        try:
            out_dir = os.path.join(os.getcwd(), 'misclassified')
            os.makedirs(out_dir, exist_ok=True)

            src = info.get('file_path')
            ts = int(info.get('timestamp', time.time()))
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)
            dst_img = os.path.join(out_dir, f"{name}_{ts}{ext}")
            shutil.copy(src, dst_img)

            meta = {
                'saved_at': ts,
                'original_path': src,
                'final_label': info.get('final_label'),
                'top3': info.get('top3')
            }
            dst_meta = os.path.join(out_dir, f"{name}_{ts}.json")
            with open(dst_meta, 'w') as f:
                json.dump(meta, f, indent=2)

            messagebox.showinfo("Tersimpan", f"Hasil dan gambar disimpan di: {out_dir}")
            # Open folder (Windows)
            try:
                os.startfile(out_dir)
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan: {e}")

    def on_exit(self):
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = ctk.CTk()
    app = SirihApp(root)
    root.mainloop()
