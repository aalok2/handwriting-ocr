import os
import re
import tempfile

# Skip model-source connectivity check for faster startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR

app = Flask(__name__)

# Initialize PaddleOCR once (downloads models on first run)
ocr = PaddleOCR(use_textline_orientation=True, lang='en')


def extract_numbers_and_sum(image_path: str) -> dict:
    """Run OCR on the image, extract numbers, and compute their sum."""
    all_detections = []

    for result in ocr.predict(image_path):
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polys = result.get("rec_polys", [])

        for i, (text, score) in enumerate(zip(texts, scores)):
            # Compute bounding-box area to gauge text prominence
            area = 0
            if i < len(polys):
                p = polys[i]
                # Shoelace formula on the polygon (4 corners)
                n = len(p)
                for j in range(n):
                    x1, y1 = int(p[j][0]), int(p[j][1])
                    x2, y2 = int(p[(j + 1) % n][0]), int(p[(j + 1) % n][1])
                    area += x1 * y2 - x2 * y1
                area = abs(area) / 2

            all_detections.append({
                "text": text,
                "confidence": round(score, 3),
                "area": area,
            })

    # ── Filter 1: confidence threshold ──
    MIN_CONFIDENCE = 0.5
    filtered = [d for d in all_detections if d["confidence"] >= MIN_CONFIDENCE]

    # ── Filter 2: keep only the larger text regions ──
    # (discards small, faint background scribbles)
    if filtered:
        areas = [d["area"] for d in filtered]
        median_area = sorted(areas)[len(areas) // 2]
        # Keep detections whose area is at least 30% of the median
        filtered = [d for d in filtered if d["area"] >= median_area * 0.3]

    # Combine remaining text fragments
    full_text = " ".join([d["text"] for d in filtered])

    # Extract whole numbers (skip decimals like 000.02 by consuming them)
    # First remove decimal numbers so they don't become fragments
    cleaned = re.sub(r'\d+\.\d+', ' ', full_text)
    numbers = [int(n) for n in re.findall(r'-?\d+', cleaned)]

    return {
        "raw_text": " ".join([d["text"] for d in all_detections]),
        "filtered_text": full_text,
        "detections": all_detections,
        "numbers": numbers,
        "sum": sum(numbers) if numbers else 0,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to a temp file
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = extract_numbers_and_sum(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
