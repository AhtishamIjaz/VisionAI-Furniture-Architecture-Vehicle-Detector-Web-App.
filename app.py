import os
import cv2
import base64
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from detector import VisionDetector

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLES_DIR = BASE_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max upload

# Initialize VisionDetector
print("[*] Initializing VisionAI Detector Engine...")
detector = VisionDetector(models_dir=str(MODELS_DIR), default_model="coco")
print("[OK] VisionAI Detector Engine Ready!")

def image_to_base64(img_bgr, quality=92):
    """Converts OpenCV BGR image to base64 JPEG string."""
    success, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(str(SAMPLES_DIR), filename)

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns list of bundled sample images."""
    samples = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for f in SAMPLES_DIR.glob(ext):
            samples.append({
                "filename": f.name,
                "url": f"/samples/{f.name}",
                "title": f.stem.replace("_", " ").title()
            })
    return jsonify({"samples": samples})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        
        conf_thresh = float(request.form.get("conf", 0.25))
        model_choice = request.form.get("model", "coco")
        sample_name = request.form.get("sample", None)

        detector.switch_model(model_choice)

        img_bgr = None

        # 1. Check if user selected a sample image
        if sample_name:
            sample_path = SAMPLES_DIR / secure_filename(sample_name)
            if sample_path.exists():
                img_bgr = cv2.imread(str(sample_path))

        # 2. Check if user uploaded a picture file
        if img_bgr is None and "file" in request.files:
            file = request.files["file"]
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                save_path = UPLOAD_DIR / f"{int(time.time())}_{filename}"
                file.save(str(save_path))
                img_bgr = cv2.imread(str(save_path))

        if img_bgr is None:
            return jsonify({
                "success": False,
                "error": "No valid image provided. Please upload a picture or select a sample image."
            }), 400

        # Run inference
        result = detector.predict(img_bgr, conf_thresh=conf_thresh)
        elapsed = round((time.time() - start_time) * 1000, 1)

        # Convert images to base64 for direct browser rendering
        orig_b64 = image_to_base64(img_bgr)
        annotated_b64 = image_to_base64(result["annotated_image"])

        return jsonify({
            "success": True,
            "inference_time_ms": elapsed,
            "model_used": result["model_used"],
            "total_objects": result["total_objects"],
            "category_counts": result["category_counts"],
            "detections": result["detections"],
            "original_image": orig_b64,
            "annotated_image": annotated_b64,
            "image_dims": result["image_dims"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
