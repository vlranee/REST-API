import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

from services.predictor import predict_teks
from groq_service import generate_rekomendasi

app = Flask(__name__)
CORS(app)

MAX_TEXT_LENGTH = 2000

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Smishing Detection API aktif",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predictions"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status":"healthy"}),200


@app.route("/api/v1/predictions", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "teks" not in data:
        return jsonify({
            "error": "Field 'teks' wajib diisi",
            "code": "MISSING_FIELD"
        }), 400

    teks = data["teks"].strip()

    if not teks:
        return jsonify({
            "error": "Field 'teks' tidak boleh kosong",
            "code": "EMPTY_FIELD"
        }), 400

    if len(teks) > MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"Teks melebihi {MAX_TEXT_LENGTH} karakter",
            "code": "TEXT_TOO_LONG"
        }), 422

    try:
        result = predict_teks(teks)
    except Exception as e:
        logger.error("Model prediction failed: %s", e, exc_info=True)
        return jsonify({
            "error": "Prediksi gagal, coba lagi",
            "code": "PREDICTION_ERROR"
        }), 500

    rekomendasi = None
    if result["label"] == "PHISHING":
        try:
            rekomendasi = generate_rekomendasi(
                teks,
                result["phishing_score"],
                result["label"],
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            rekomendasi = "Analisis detail tidak tersedia saat ini."

    return jsonify({
        "teks": teks,
        "label": result["label"],
        "is_phishing": result["label"] == "PHISHING",
        "phishing_score": result["phishing_score"],
        "normal_score": round(100 - result["phishing_score"], 2),
        "confidence": result["confidence"],
        "rekomendasi": rekomendasi 
    }), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)