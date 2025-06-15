import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load SavedModel folder
model = tf.keras.models.load_model("my_model")  # Make sure this folder contains 'saved_model.pb'

# Update this with your actual class labels
class_names = ["Perfect", "Abnormal"]

# Prediction function
def predict(image):
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    label = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return f"{label} ({confidence*100:.2f}%)"

# Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Inspectors Ally: Fabric Defect Detection",
    description="Upload an image to detect defects in fabric using a model trained on Teachable Machine."
).launch()
