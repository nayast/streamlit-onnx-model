import streamlit as st

import onnxruntime as ort
from transformers import ViTImageProcessor
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def preprocess_img(image, processor):
    image = processor(images=image, return_tensors="pt")
    return image

def load_model(model):
    ort_session = ort.InferenceSession(model)
    return ort_session

def model_inference(ort_session, image, labels):
    ort_inputs = {ort_session.get_inputs()[0].name: image["pixel_values"].numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs

def main():
    ort_outputs = None
    st.set_page_config(layout="wide")
    col1, col2 = st.columns([0.3, 0.7])
    processor = ViTImageProcessor.from_pretrained("model/preprocessor_config.json")
    with open("model/labels.json", "r") as f:
        labels = json.load(f)
    ort_session = load_model("model/plant-disease.onnx")
    with col1:
        file_image = st.file_uploader(label="Select image", type=["jpg", "jpeg"])
        if file_image is not None:
            image = Image.open(file_image)
            image_processed = preprocess_img(image, processor)
            ort_outputs = model_inference(ort_session, image_processed, labels)
            st.image(image, caption="Uploaded image")

    with col2:
        if ort_outputs is not None:
            x_axis = [float(prob) for prob in softmax(ort_outputs)[0][0]]
            y_axis = [labels[key] for key in labels]
            plt.figure(figsize=(2, 1))
            plt.barh(y_axis, x_axis)

            st.pyplot(plt.gcf())
            st.markdown(f"### Predicted label: {labels[str(ort_outputs[0].argmax(axis=-1)[0])]}")


if __name__ == '__main__':
    main()
