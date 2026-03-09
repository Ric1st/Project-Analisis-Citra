import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="Disaster Detector AI", layout="centered")

# 1. LOAD LABELS
with open('class_labels.json', 'r') as f:
    class_names = json.load(f)

# Tentukan mana yang aman/bahaya (Sesuaikan dengan nama folder di datasetmu)
DANGEROUS_CLASSES = ["Water_Disaster", "Infrastructure", "Earthquake",
                     "Human_Damage", "Urban_Fire", "Wild_Fire", "Land_Slide", "Drought"]

# 2. LOAD MODEL ARCHITECTURE


@st.cache_resource  # Agar model tidak di-load berulang kali (biar cepat)
def load_model():
    model = models.resnet50(weights=None)  # Tidak perlu download weights lagi
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 12)
    )
    # Load weights yang sudah kamu latih di Kaggle
    model.load_state_dict(torch.load(
        'model_final_resnet50_bencana.pth', map_location='cpu'))
    model.eval()
    return model


model = load_model()

# 3. IMAGE PREPROCESSING (Harus sama dengan saat training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI STREAMLIT ---
st.title("🚨 Disaster & Sensitive Content Detector")
st.write("Unggah gambar untuk mendeteksi jenis bencana dan tingkat keamanan.")

uploaded_file = st.file_uploader(
    "Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Proses Prediksi
    st.write("🔍 Sedang menganalisis...")
    img_tensor = transform(image).unsqueeze(0)  # Tambah dimensi batch

    with torch.no_grad():
        outputs = model(img_tensor)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        _, predicted = outputs.max(1)

        label = class_names[predicted.item()]
        confidence = percentage[predicted.item()].item()

    # Tampilkan Hasil
    st.divider()
    st.subheader(f"Prediksi: **{label}**")
    st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    # Logika Aman/Berbahaya
    if label in DANGEROUS_CLASSES:
        st.error("⚠️ STATUS: KONTEN BERBAHAYA / SENSITIF")
        st.write(
            "Gambar ini terdeteksi mengandung unsur bencana atau kerusakan manusia.")
    else:
        st.success("✅ STATUS: AMAN")
        st.write("Gambar ini tergolong aman untuk ditampilkan.")
