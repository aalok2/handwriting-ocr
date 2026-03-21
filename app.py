import os
import re
import json
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Skip model-source connectivity check for faster startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
from litellm import completion, RateLimitError

app = Flask(__name__)

# Initialize PaddleOCR once (downloads models on first run)
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# LLM Router Configuration
# You can switch the model here. Examples:
# "gpt-4o-mini" (OpenAI)
# "claude-3-haiku-20240307" (Anthropic)
# "gemini/gemini-2.0-flash" (Google)
# "groq/llama3-8b-8192" (Groq/OSS)
# "huggingface/microsoft/Phi-3-mini-4k-instruct" (HuggingFace)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def extract_numbers_and_sum(image_path: str) -> dict:
    """Run OCR on the image, then use an LLM agent to extract prices and compute their sum."""
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

    # Combine all text fragments for the LLM
    full_text = " ".join([d["text"] for d in all_detections])

    extracted_numbers, calculated_sum = call_llm_agent(full_text)

    return {
        "raw_text": full_text,
        "filtered_text": full_text,
        "detections": all_detections,
        "numbers": extracted_numbers,
        "sum": calculated_sum,
    }

def call_llm_agent(full_text: str) -> tuple[list[float], float]:
    # --- LLM Agent Layer ---
    system_prompt = """You are an intelligent receipt and handwriting parsing assistant.
Your task is to extract individual prices from the provided OCR text and calculate their sum.
CRITICAL RULES:
1. ONLY extract prices. Ignore quantities (e.g., '3kg', '4 pcs', '2x').
2. Ignore any pre-calculated total sums (e.g., if the text says 'Total: 150', do NOT include 150 in your list of prices).
3. You MUST call the `calculate_sum` tool with the list of extracted prices.
"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sum",
                "description": "Calculates the sum of a list of prices extracted from the receipt/image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prices": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of individual prices to sum up."
                        }
                    },
                    "required": ["prices"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the OCR text:\n\n{full_text}"}
    ]

    max_retries = 2
    extracted_numbers = []
    calculated_sum = 0
    
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "calculate_sum"}}
            )
            
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "calculate_sum":
                    args = json.loads(tool_call.function.arguments)
                    extracted_numbers = args.get("prices", [])
                    calculated_sum = sum(extracted_numbers)
                    break
            
            # If no tool call or wrong tool call, we can retry
            messages.append({"role": "assistant", "content": "You must call the `calculate_sum` tool with the extracted prices."})
        except RateLimitError as e:
            print(f"Rate limit / quota exceeded — not retrying: {e}")
            break
        except Exception as e:
            if attempt == max_retries:
                print(f"LLM failed after {max_retries} retries: {e}")
            else:
                print(f"LLM attempt {attempt + 1} failed: {e}. Retrying...")

    return extracted_numbers, calculated_sum


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
    app.run(debug=True, host="0.0.0.0", port=5001)
