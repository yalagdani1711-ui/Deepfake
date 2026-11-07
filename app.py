import os
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import pipeline

app = Flask(__name__)
CORS(app)

print("Loading deepfake detection AI model into memory...")
pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
print("Model loaded successfully!")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Server active"}), 200

@app.route("/check-image", methods=["POST"])
def check_image():
    data = request.json or {}
    image_url = data.get("url")
    if not image_url:
        return jsonify({"isDeepfake": False, "confidence": 0})
    try:
        resp = requests.get(image_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        results = pipe(img)
        fake_score = 0.0
        for res in results:
            if "fake" in res["label"].lower():
                fake_score = res["score"]
        is_fake = fake_score >= 0.70
        return jsonify({"isDeepfake": is_fake, "confidence": fake_score})
    except Exception as e:
        print(f"Error evaluating image: {e}")
        return jsonify({"isDeepfake": False, "confidence": 0})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
