import streamlit as st
import os
import zipfile
import kaggle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import random
from PIL import Image

# ======================
# CONFIG
# ======================
DATASET = "Plantvillage"
DATASET_ZIP = "Plantvillage.zip"
MODEL_PATH = "plant_disease_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

# ======================
# STREAMLIT HEADER
# ======================
st.title("🌱 Plant Disease Detector")
st.write("Upload a leaf image to detect disease and get fertilizer recommendations.")


# ======================
# FUNCTION: Download dataset from Kaggle
# ======================
@st.cache_data(show_spinner=False)
def download_dataset():
    data_dir = Path(DATASET)
    if not data_dir.exists():
        st.info("📦 Downloading dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("emmarex/plantdisease", path=".", unzip=True)
        # If it's zipped, extract it
        if os.path.exists(DATASET_ZIP):
            with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(DATASET_ZIP)
    return Path("PlantVillage")


# ======================
# FUNCTION: Load and preprocess dataset
# ======================
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    return train_gen, val_gen


# ======================
# FUNCTION: Build Model
# ======================
def build_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# ======================
# FUNCTION: Train model (cached)
# ======================
@st.cache_resource(show_spinner=False)
def train_or_load_model():
    data_dir = download_dataset()
    train_gen, val_gen = load_data(data_dir)

    model = None

    if os.path.exists(MODEL_PATH):
        st.success("✅ Loaded cached model.")
        model = tf.keras.models.load_model(MODEL_PATH)
        history = None
    else:
        st.info("🚀 Training model...")
        model = build_model(num_classes=len(train_gen.class_indices))
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
        model.save(MODEL_PATH)
    return model, train_gen, val_gen, history


# ======================
# FUNCTION: Plot training curves
# ======================
def plot_training_curves(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history["accuracy"], label="Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axs[0].set_title("Accuracy")
    axs[0].legend()

    axs[1].plot(history.history["loss"], label="Train Loss")
    axs[1].plot(history.history["val_loss"], label="Val Loss")
    axs[1].set_title("Loss")
    axs[1].legend()
    st.pyplot(fig)


# ======================
# FUNCTION: Plot confusion matrix
# ======================
def plot_conf_matrix(model, val_gen):
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        xticklabels=val_gen.class_indices.keys(),
        yticklabels=val_gen.class_indices.keys(),
        cmap="Blues",
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    st.pyplot(fig)


# Fertilizer recommendation
recommendations = {
    "Apple___Apple_scab": "Use fungicide containing Captan or Mancozeb. Add compost to improve soil.",
    "Apple___Black_rot": "Apply copper-based fungicide. Avoid overhead irrigation.",
    "Apple___Cedar_apple_rust": "Use sulfur spray and maintain plant spacing.",
    "Apple___healthy": "No disease detected. Use balanced NPK fertilizer.",
    "Blueberry___healthy": "Use composted mulch and slow-release NPK fertilizer.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicide and improve airflow.",
    "Cherry_(including_sour)___healthy": "Use organic mulch and potassium-rich fertilizer.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use fungicides like azoxystrobin and improve nitrogen balance.",
    "Corn_(maize)___Common_rust_": "Apply fungicide containing strobilurins. Avoid dense planting.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use fungicide like propiconazole. Rotate crops.",
    "Corn_(maize)___healthy": "Use nitrogen-rich fertilizers like urea.",
    "Grape___Black_rot": "Use fungicide with mancozeb or captan. Remove infected leaves.",
    "Grape___Esca_(Black_Measles)": "Remove infected canes. Use balanced fertilizer.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper fungicide and improve air flow.",
    "Grape___healthy": "Use compost tea and low-nitrogen fertilizer.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees. Apply micronutrient sprays.",
    "Peach___Bacterial_spot": "Apply copper sprays before flowering. Ensure proper drainage.",
    "Peach___healthy": "Use slow-release balanced fertilizer.",
    "Pepper__bell___Bacterial_spot": "Use disease-free seeds and copper-based fungicide.",
    "Pepper__bell___healthy": "Apply compost and calcium nitrate.",
    "Potato___Early_blight": "Use a copper-based fungicide. Add compost to soil.",
    "Potato___Late_blight": "Use fungicides like chlorothalonil. Improve drainage.",
    "Potato___healthy": "No disease. Use potassium-rich fertilizer.",
    "Raspberry___healthy": "Use aged compost and mulch with straw.",
    "Soybean___healthy": "Use inoculants containing Rhizobium and apply potash.",
    "Squash___Powdery_mildew": "Use neem oil or potassium bicarbonate spray.",
    "Strawberry___Leaf_scorch": "Use mulch and avoid overhead watering. Apply nitrogen-rich fertilizer.",
    "Strawberry___healthy": "Use well-rotted manure and phosphate-rich fertilizer.",
    "Tomato_Bacterial_spot": "Use copper-based bactericides. Ensure crop rotation.",
    "Tomato_Early_blight": "Use chlorothalonil spray. Improve air flow.",
    "Tomato_Late_blight": "Use metalaxyl fungicide. Avoid wet foliage.",
    "Tomato_Leaf_Mold": "Use sulfur fungicide. Improve air circulation.",
    "Tomato_Septoria_leaf_spot": "Use fungicides like chlorothalonil or mancozeb.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticide spray. Keep the area weed-free.",
    "Tomato__Target_Spot": "Apply chlorothalonil fungicide. Improve air flow.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Use resistant varieties. Apply neem oil spray.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants. Use certified seeds.",
    "Tomato_healthy": "Use compost and balanced NPK fertilizer.",
}


# ======================
# MAIN APP
# ======================
model, train_gen, val_gen, history = train_or_load_model()

_, img_height, img_width, _ = model.input_shape
st.write(f"Model expects images of size: {img_height}x{img_width}")

if history:
    st.subheader("📈 Training Curves")
    plot_training_curves(history)

    st.subheader("🔍 Confusion Matrix")
    plot_conf_matrix(model, val_gen)

# Upload image for prediction
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    img_resized = image.resize((img_width, img_height))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image for model
    # img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    # Show result
    st.subheader("Prediction Result:")
    pred_class = list(train_gen.class_indices.keys())[np.argmax(predictions)]
    st.write(f"**Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.info(
        f"💡 Recommendation: {recommendations.get(pred_class, 'No specific recommendation available.')}"
    )
