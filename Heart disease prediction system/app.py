"""
Heart Disease Prediction System - Web Application Server
Flask web application integrating the complete 5-stage medical image pipeline:
1. Load Input Image
2. Preprocessing
3. Feature Extraction
4. ML Training
5. Performance Evaluation & Diagnostic Prediction
+ Interactive Code Studio for editing, compiling, and running code live.
"""

import os
import io
import time
import json
import base64
import subprocess
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

from preprocess import ImagePreprocessor
from feature_extraction import FeatureExtractor
from train_model import HeartDiseaseMLTrainer, CLASSES, FEATURE_NAMES

app = Flask(__name__)
app.config["SECRET_KEY"] = "cardio-ai-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
STATIC_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_SAMPLES_DIR, exist_ok=True)

# Initialize core pipeline singletons
preprocessor = ImagePreprocessor(target_size=(256, 256))
feature_extractor = FeatureExtractor(glcm_levels=16)
ml_trainer = HeartDiseaseMLTrainer()

# Ensure default model is trained on startup
try:
    eval_json_path = os.path.join(os.path.dirname(__file__), "models", "latest_evaluation.json")
    if not os.path.exists(eval_json_path):
        ml_trainer.train(algorithm="Random Forest")
except Exception as e:
    print(f"Initial model training notice: {e}")


# Pre-defined sample patient metadata
PATIENT_SAMPLES = [
    {
        "id": "sample-1",
        "title": "Normal Sinus Rhythm",
        "category": "Healthy Patient",
        "file": "normal_sinus_rhythm.png",
        "description": "Standard 12-lead ECG Lead II showing uniform P waves, sharp QRS complexes, and normal T waves. No ischemia or arrhythmia.",
        "patient": {"age": 42, "gender": "Female", "bp": "118/76 mmHg", "hr": "72 bpm"},
        "expected_class": "Normal Healthy"
    },
    {
        "id": "sample-2",
        "title": "Acute STEMI Infarction",
        "category": "High Risk / Critical",
        "file": "myocardial_infarction.png",
        "description": "Acute Anterior Myocardial Infarction showing significant ST-segment elevation (>2mm) and deep pathological Q wave. Acute coronary occlusion.",
        "patient": {"age": 61, "gender": "Male", "bp": "158/98 mmHg", "hr": "105 bpm"},
        "expected_class": "Myocardial Infarction"
    },
    {
        "id": "sample-3",
        "title": "Atrial Fibrillation",
        "category": "Arrhythmia / Moderate",
        "file": "arrhythmia_atrial_fib.png",
        "description": "Disorganized baseline fibrillatory oscillations with irregularly irregular ventricular response intervals (R-R variability). Absent distinct P waves.",
        "patient": {"age": 68, "gender": "Male", "bp": "136/84 mmHg", "hr": "138 bpm"},
        "expected_class": "Arrhythmia"
    },
    {
        "id": "sample-4",
        "title": "Coronary Ischemia (CAD)",
        "category": "Moderate Risk",
        "file": "coronary_artery_disease.png",
        "description": "Symmetrical deep T-wave inversions and horizontal ST-segment depression indicative of subendocardial ischemia from coronary arterial narrowing.",
        "patient": {"age": 55, "gender": "Male", "bp": "144/90 mmHg", "hr": "88 bpm"},
        "expected_class": "Coronary Artery Disease"
    },
    {
        "id": "sample-5",
        "title": "Cardiomegaly Scan",
        "category": "Severe / Chronic",
        "file": "cardiomegaly_heart_failure.png",
        "description": "Chest Radiograph silhouette displaying severe cardiothoracic enlargement (CTR > 0.65) consistent with congestive heart failure and dilated cardiomyopathy.",
        "patient": {"age": 73, "gender": "Female", "bp": "150/92 mmHg", "hr": "82 bpm"},
        "expected_class": "Cardiomegaly / Heart Failure"
    }
]


@app.route("/")
def index():
    """Renders the main single-frame unified dashboard application."""
    return render_template("index.html", samples=PATIENT_SAMPLES)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns the catalog of preloaded clinical cardiac samples."""
    return jsonify({"success": True, "samples": PATIENT_SAMPLES})


@app.route("/download/report-docx", methods=["GET"])
def download_docx_report():
    """Endpoint to download the generated project Word document."""
    docx_path = os.path.join(os.path.dirname(__file__), "Heart_Disease_Prediction_Project_Report.docx")
    if os.path.exists(docx_path):
        return send_file(docx_path, as_attachment=True, download_name="Heart_Disease_Prediction_Project_Report.docx")
    return jsonify({"success": False, "error": "Document not found"}), 404


@app.route("/download/report-pdf", methods=["GET"])
def download_pdf_report():
    """Endpoint to download the generated project PDF report."""
    pdf_path = os.path.join(os.path.dirname(__file__), "Heart_Disease_Prediction_Project_Report.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name="Heart_Disease_Prediction_Project_Report.pdf")
    return jsonify({"success": False, "error": "PDF not found"}), 404


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handles medical image upload or sample selection."""
    try:
        data = request.json or {}
        sample_file = data.get("sample_file")

        if sample_file:
            # Load from sample directory
            sample_path = os.path.join(STATIC_SAMPLES_DIR, secure_filename(sample_file))
            if not os.path.exists(sample_path):
                return jsonify({"success": False, "error": "Sample file not found"}), 404

            with open(sample_path, "rb") as f:
                img_bytes = f.read()

            base64_str = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            return jsonify({
                "success": True,
                "image_data": base64_str,
                "source": "sample",
                "filename": sample_file
            })

        # Check for uploaded file in multipart form
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400

            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)

            with open(save_path, "rb") as f:
                img_bytes = f.read()

            base64_str = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            return jsonify({
                "success": True,
                "image_data": base64_str,
                "source": "upload",
                "filename": filename
            })

        return jsonify({"success": False, "error": "No image data or file provided"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def run_preprocess():
    """Executes the multi-stage image preprocessing pipeline."""
    try:
        data = request.json
        image_data = data.get("image_data")
        contrast_factor = float(data.get("contrast", 1.6))
        noise_radius = float(data.get("noise_radius", 1.2))
        threshold_val = int(data.get("threshold", 130))

        if not image_data:
            return jsonify({"success": False, "error": "Image data missing"}), 400

        result = preprocessor.process_full_pipeline(
            input_source=image_data,
            contrast_factor=contrast_factor,
            noise_radius=noise_radius,
            threshold_val=threshold_val
        )

        return jsonify({
            "success": True,
            "stages": {
                "original": result["original"],
                "grayscale": result["grayscale"],
                "denoised": result["denoised"],
                "contrast_enhanced": result["contrast_enhanced"],
                "thresholded": result["thresholded"],
                "edges": result["edges"]
            },
            "dimensions": result["dimensions"],
            "histogram": result["histogram"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/features", methods=["POST"])
def run_feature_extraction():
    """Extracts GLCM texture, statistical moments, and cardiac wave descriptors."""
    try:
        data = request.json
        image_data = data.get("image_data")

        if not image_data:
            return jsonify({"success": False, "error": "Image data missing"}), 400

        # Load image via preprocessor
        img = preprocessor.load_image(image_data)
        gray = preprocessor.to_grayscale(img)

        # Extract all features
        features_result = feature_extractor.extract_all_features(gray)

        return jsonify({
            "success": True,
            "features": features_result["features"],
            "feature_names": features_result["feature_names"],
            "feature_vector": features_result["feature_vector"],
            "glcm_matrix_preview": features_result["glcm_matrix_preview"],
            "summary": features_result["summary"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def run_model_training():
    """Trains a chosen Machine Learning model and returns performance metrics."""
    try:
        data = request.json or {}
        algorithm = data.get("algorithm", "Random Forest")
        test_size = float(data.get("test_size", 0.2))
        params = data.get("params", {})

        metrics = ml_trainer.train(algorithm=algorithm, test_size=test_size, params=params)

        return jsonify({
            "success": True,
            "metrics": metrics
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def run_prediction():
    """Runs ML inference on extracted feature vector and returns clinical diagnosis."""
    try:
        data = request.json
        feature_vector = data.get("feature_vector")

        if not feature_vector:
            return jsonify({"success": False, "error": "Feature vector required"}), 400

        prediction_result = ml_trainer.predict_from_features(feature_vector)

        return jsonify({
            "success": True,
            "diagnosis": prediction_result
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/code_templates", methods=["GET"])
def get_code_templates():
    """Returns available Python scripts so the user can load, inspect, and run them."""
    base_dir = os.path.dirname(__file__)
    files = {
        "train_model.py": "train_model.py",
        "preprocess.py": "preprocess.py",
        "feature_extraction.py": "feature_extraction.py",
        "test_predict.py": "test_predict.py",
        "app.py": "app.py"
    }

    templates = {}
    for key, filename in files.items():
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                templates[key] = f.read()

    return jsonify({"success": True, "templates": templates})


@app.route("/api/run_code", methods=["POST"])
def run_code():
    """
    Executes user-typed Python code or script directly on the backend
    and streams terminal output (stdout/stderr) back to the web console.
    """
    try:
        data = request.json or {}
        code = data.get("code", "")
        script_name = data.get("script_name", "custom_script.py")

        if not code.strip():
            return jsonify({"success": False, "error": "No code provided to execute."}), 400

        # Save to temporary scratch script in workspace
        scratch_dir = os.path.join(os.path.dirname(__file__), "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        safe_script_path = os.path.join(scratch_dir, "run_temp.py")

        with open(safe_script_path, "w", encoding="utf-8") as f:
            f.write(code)

        start_time = time.time()
        # Execute in python environment with timeout protection
        proc = subprocess.run(
            ["python", safe_script_path],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=25
        )
        elapsed = round((time.time() - start_time) * 1000, 1)

        output = proc.stdout
        if proc.stderr:
            output += ("\n--- STDERR ---\n" + proc.stderr)

        return jsonify({
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "output": output if output.strip() else "[Process finished with no console output]",
            "execution_time_ms": elapsed
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "returncode": -1,
            "output": "Error: Execution timed out (limit: 25 seconds).",
            "execution_time_ms": 25000
        }), 408
    except Exception as e:
        return jsonify({
            "success": False,
            "returncode": 1,
            "output": f"Execution error: {str(e)}",
            "execution_time_ms": 0
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" HEART DISEASE PREDICTION SYSTEM - WEB APPLICATION")
    print("=" * 65)
    print(" Server running on http://127.0.0.1:5000")
    print(" Open the above URL in your browser to access the application.")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
