import os
import gdown
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from custom_layers import TextPreprocessingLayer, WeightedBinaryCrossentropy
from feature_extractor import extract_num_features

logger = logging.getLogger(__name__)

_model = None

def download_model():
    file_id = "1M2_QYW9ETpkkyufW4kg7OBhJV9KPVXKI"
    url = f"https://drive.google.com/uc?id={file_id}"

    os.makedirs("model", exist_ok=True)
    output = "model/smishing_model_best.keras"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    return output

def get_model():
    global _model
    if _model is None:

        model_path = download_model()

        _model = load_model(
            model_path,
            custom_objects={
                "TextPreprocessingLayer": TextPreprocessingLayer,
                "WeightedBinaryCrossentropy": WeightedBinaryCrossentropy,
            },
            compile=False,
        )

        logger.info("Model loaded dari %s", model_path)

    return _model


THRESHOLD = 0.3794


def predict_teks(teks: str) -> dict:
    model = get_model()
    teks_input = tf.constant([[teks]], dtype=tf.string)
    num_input = extract_num_features(teks)

    prob = float(model.predict([teks_input, num_input], verbose=0)[0][0])

    phishing_score = round(prob * 100, 2)
    label = "PHISHING" if prob >= THRESHOLD else "NORMAL"
    confidence = phishing_score if label == "PHISHING" else round((1 - prob) * 100, 2)

    return {
        "label": label,
        "phishing_score": phishing_score,
        "confidence": confidence,
    }