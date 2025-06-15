import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load SavedModel from the folder
model = tf.keras.models.load_model("keras_Model")

# Define your class labels
class_names = ["Perfect", "Abnormal"]  # Update if you have more

def predict(image):
    image = image.resize((224, 224))  # Match model input size
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    label = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return f"{label} ({confidence*100:.2f}%)"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Inspectors Ally: Fabric Defect Detection",
    description="Upload an image to detect fabric defects using a Teachable Machine model."
).launch()
