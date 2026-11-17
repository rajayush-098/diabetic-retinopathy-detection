import streamlit as st
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mobilenetv2_independent_finetuned_best.keras"
)

IMG_SIZE = 224

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# =====================================================
# 2. LOAD TRAINED MODEL AND GRAD-CAM MODEL
# =====================================================

@st.cache_resource
def load_models():

    model = tf.keras.models.load_model(MODEL_PATH)

    # Locate MobileNetV2 backbone
    base_model = next(
        (
            layer for layer in model.layers
            if isinstance(layer, tf.keras.Model)
        ),
        None
    )

    if base_model is None:
        raise ValueError("MobileNetV2 backbone not found.")

    # Final convolutional activation layer
    last_conv_layer = base_model.get_layer("out_relu")

    # Create feature extractor
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Locate classification head
    base_index = model.layers.index(base_model)

    classifier_layers = model.layers[base_index + 1:]

    # Create Grad-CAM model
    grad_input = tf.keras.Input(
        shape=(224, 224, 3)
    )

    conv_outputs = feature_extractor(
        grad_input,
        training=False
    )

    x = conv_outputs

    for layer in classifier_layers:
        x = layer(x)

    grad_model = tf.keras.Model(
        inputs=grad_input,
        outputs=[conv_outputs, x]
    )

    return model, grad_model


# =====================================================
# 3. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image_array = np.array(image)

    tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.uint8
    )

    # Resize while maintaining aspect ratio
    tensor = tf.image.resize_with_pad(
        tensor,
        IMG_SIZE,
        IMG_SIZE,
        antialias=True
    )

    tensor = tf.cast(tensor, tf.float32)

    # Original image for display
    display_image = tf.clip_by_value(
        tensor / 255.0,
        0,
        1
    ).numpy()

    # MobileNetV2 preprocessing
    processed_image = preprocess_input(tensor)

    img_array = tf.expand_dims(
        processed_image,
        axis=0
    )

    return display_image, img_array


# =====================================================
# 4. PREDICTION
# =====================================================

def predict_dr(model, img_array):

    probabilities = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    return predicted_class, probabilities


# =====================================================
# 5. GRAD-CAM FUNCTION
# =====================================================

def generate_gradcam(img_array, grad_model):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients.")

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    feature_maps = conv_outputs[0]

    weights = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + 1e-8),
        tf.zeros_like(heatmap)
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()

    return heatmap


# =====================================================
# 6. APPLICATION INTERFACE
# =====================================================

st.title("👁️ Diabetic Retinopathy Detection")

st.subheader(
    "Deep Learning-Based Retinal Image Classification"
)

st.write(
    """
    Upload a retinal fundus image to obtain a model-generated
    prediction across five diabetic retinopathy categories.
    """
)

st.warning(
    """
    Educational research prototype only. This application is
    not clinically validated and must not be used for medical
    diagnosis or treatment decisions.
    """
)


# =====================================================
# 7. IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    try:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        model, grad_model = load_models()

        display_image, img_array = preprocess_image(
            original_image
        )

        with st.spinner("Analyzing retinal image..."):

            predicted_class, probabilities = predict_dr(
                model,
                img_array
            )

            heatmap = generate_gradcam(
                img_array,
                grad_model
            )

        st.success("Image analysis completed!")

        # =============================================
        # 8. PREDICTION RESULTS
        # =============================================

        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted DR Grade",
                CLASS_NAMES[predicted_class]
            )

        with col2:

            st.metric(
                "Model Softmax Score",
                f"{probabilities[predicted_class] * 100:.2f}%"
            )

        st.caption(
            "Softmax scores are model outputs, not calibrated "
            "medical probabilities or diagnostic certainty."
        )

        # =============================================
        # 9. CLASS PROBABILITIES
        # =============================================

        st.subheader("Class-wise Prediction Scores")

        probability_data = {
            "DR Grade": CLASS_NAMES,
            "Model Score (%)": (
                probabilities * 100
            ).round(2)
        }

        st.dataframe(
            probability_data,
            use_container_width=True,
            hide_index=True
        )

        # =============================================
        # 10. IMAGE AND GRAD-CAM VISUALIZATION
        # =============================================

        st.header("Grad-CAM Visualization")

        heatmap_color = plt.get_cmap("jet")(
            heatmap
        )[:, :, :3]

        overlay = (
            0.6 * display_image
            + 0.4 * heatmap_color
        )

        overlay = np.clip(
            overlay,
            0,
            1
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                display_image,
                caption="Original Retinal Image",
                use_container_width=True
            )

        with col2:

            st.image(
    heatmap_color,
    caption="Grad-CAM Heatmap",
    use_container_width=True
)

        with col3:

            st.image(
                overlay,
                caption="Grad-CAM Overlay",
                clamp=True,
                use_container_width=True
            )

        st.info(
            """
            Grad-CAM highlights image regions that contribute
            positively to the selected class prediction.
            It does not identify or confirm retinal lesions.
            """
        )

    except Exception as error:

        st.error(
            f"An error occurred during image analysis: {error}"
        )

else:

    st.info(
        "Please upload a retinal fundus image to begin."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "Diabetic Retinopathy Detection Using Deep Learning "
    "and MobileNetV2 | Academic Research Project"
)