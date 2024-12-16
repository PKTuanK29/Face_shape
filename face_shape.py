import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Thay đổi tiêu đề tab trình duyệt và favicon
st.set_page_config(page_title="FaceShape", layout="wide")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .block-container {
        padding: 1rem 2rem;
        max-width: 900px;
        margin: auto;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'MyModel.keras')

model = load_model()
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
label_translation = {
    'Heart': 'Mặt trái tim', 'Oblong': 'Mặt thon dài',
    'Oval': 'Mặt trái xoan', 'Round': 'Mặt tròn', 'Square': 'Mặt vuông'
}

def preprocess_image(image_file):
    img = Image.open(image_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image_file, model, class_labels):
    img_array = preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    predicted_prob = np.max(predictions)
    return predictions, predicted_label, predicted_prob

def suggest_hairstyles(face_shape):
    base_url = "https://raw.githubusercontent.com/tkieuvt/face_shape/main/images/"
    suggestions = {
        'Heart': [("heart1.jpg", "Tóc dài xoăn lơi"), ("heart2.jpg", "Tóc layer ngắn với mái thưa bay"), ("heart3.webp", "Tóc đuôi ngựa với mái bay")],
        'Oblong': [("oblong1.webp", "Tóc búi thấp với mái bay"), ("oblong2.jpg", "Tóc dài uốn gợn sóng"), ("oblong3.webp", "Tóc ngang vai với mái bay")],
        'Oval': [("oval1.jpg", "Tóc dài xoăn sóng nhẹ"), ("oval2.png", "Tóc ngắn uốn cụp, mái bay"), ("oval3.png", "Tóc layer thẳng dài")],
        'Round': [("round1.jpg", "Tóc dài uốn sóng lơi với mái thưa"), ("round2.jpg", "Tóc hippie ngắn với mái thưa"), ("round3.jpg", "Tóc bob ngang vai với mái thưa")],
        'Square': [("square1.jpg", "Tóc layer dài với phần mái dài"), ("square2.jpg", "Tóc hippie dài"), ("square3.jpg", "Tóc bob ngắn")]
    }
    return suggestions.get(face_shape, [])

def plot_predictions(predictions):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#5A4FCF', '#7A6FE1', '#A19BE8', '#C0BBF2', '#E4E2F7']
    bars = ax.barh(class_labels, predictions[0], color=colors)
    
    for bar, value in zip(bars, predictions[0]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f'{value * 100:.2f}%', va='center', ha='left', fontsize=10)
    
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks(range(len(class_labels)))
    ax.set_title("Dự đoán xác suất của từng lớp", fontsize=14, fontweight='bold', pad=10)
    
    st.pyplot(fig)

def display_hairstyles(face_shape):
    hairstyle_images = suggest_hairstyles(face_shape)
    for i in range(0, len(hairstyle_images), 3):
        cols = st.columns(3)
        for col, (hairstyle_url, hairstyle_name) in zip(cols, hairstyle_images[i:i + 3]):
            with col:
                st.image(hairstyle_url, caption=hairstyle_name, use_column_width=True)

st.title("Dự đoán Hình Dạng Khuôn Mặt")
input_method = st.radio("Chọn phương thức đầu vào", ("Tải ảnh từ máy tính", "Chụp ảnh từ camera"))

if input_method == "Tải ảnh từ máy tính":
    uploaded_file = st.file_uploader("Tải ảnh của bạn lên", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh đã tải lên", use_container_width=True)
        
        predictions, predicted_label, predicted_prob = predict_image(uploaded_file, model, class_labels)
        predicted_label_vn = label_translation.get(predicted_label, "Không xác định")
        predicted_prob_percent = predicted_prob * 100

        st.markdown(f"<h2 style='text-align: center;'>Dự đoán: <b>{predicted_label_vn}</b></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Xác suất: <b>{predicted_prob_percent:.2f}%</b></p>", unsafe_allow_html=True)
        plot_predictions(predictions)
        display_hairstyles(predicted_label)

elif input_method == "Chụp ảnh từ camera":
    camera_input = st.camera_input("Chụp ảnh từ camera")

    if camera_input:
        st.image(camera_input, caption="Ảnh chụp từ camera", use_container_width=True)

        predictions, predicted_label, predicted_prob = predict_image(camera_input, model, class_labels)
        predicted_label_vn = label_translation.get(predicted_label, "Không xác định")
        predicted_prob_percent = predicted_prob * 100

        st.markdown(f"<h2 style='text-align: center;'>Dự đoán: <b>{predicted_label_vn}</b></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Xác suất: <b>{predicted_prob_percent:.2f}%</b></p>", unsafe_allow_html=True)
        plot_predictions(predictions)
        display_hairstyles(predicted_label)
