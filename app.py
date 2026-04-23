import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
from PIL import Image
from datetime import datetime

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

confidence_threshold = 0.15

st.title("🚧 Deteksi Jalan Berlubang & Retak (YOLOv8)")
st.write("Upload gambar untuk melakukan deteksi")

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # =========================
    # SIMPAN FILE SEMENTARA
    # =========================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # =========================
    # DETEKSI (PAKAI PATH)
    # =========================
    results = model.predict(source=temp_path, conf=confidence_threshold)
    result = results[0]

    # =========================
    # BACA GAMBAR (SAMA PERSIS)
    # =========================
    img = cv2.imread(temp_path)

    if img is None:
        st.error("Gagal membaca gambar")
    else:
        if result.boxes is not None and len(result.boxes) > 0:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(cls)
                class_name = model.names[class_id]
                confidence = float(conf)

                # WARNA (SAMA PERSIS)
                if class_id == 0:
                    color = (0, 0, 255)
                elif class_id == 1:
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)

                label = f"{class_name.lower()} {confidence:.2f}"

                # BOX
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

                # TEXT
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            4)

            st.success("Deteksi selesai!")

        else:
            st.warning("Tidak ada objek terdeteksi")

        # =========================
        # TAMPILKAN (BGR → RGB)
        # =========================
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Hasil Deteksi", use_column_width=True)

    # Hapus file sementara
    os.remove(temp_path)