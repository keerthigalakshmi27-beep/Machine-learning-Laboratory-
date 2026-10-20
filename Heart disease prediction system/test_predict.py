"""
Heart Disease Prediction System - Standalone CLI Prediction Script
Loads an input image, runs preprocessing, extracts features, and outputs ML prediction.
Usage:
    python test_predict.py
    python test_predict.py --image path/to/ecg_or_cardiac.png
"""

import sys
import os
import argparse
from preprocess import ImagePreprocessor
from feature_extraction import FeatureExtractor
from train_model import HeartDiseaseMLTrainer


def run_prediction_cli(image_path=None):
    if not image_path:
        # Default sample image
        default_sample = os.path.join(os.path.dirname(__file__), "static", "samples", "myocardial_infarction.png")
        if os.path.exists(default_sample):
            image_path = default_sample
        else:
            print("No image provided and default sample not found.")
            return

    print("=" * 60)
    print("HEART DISEASE PREDICTION SYSTEM - CLI INFERENCE PIPELINE")
    print("=" * 60)
    print(f"[*] Step 1: Loading Input Image: {image_path}")

    preprocessor = ImagePreprocessor(target_size=(256, 256))
    pre_res = preprocessor.process_full_pipeline(image_path)
    print(f"[+] Image loaded & preprocessed successfully! Dimensions: {pre_res['dimensions']}")

    print("[*] Step 2: Extracting Texture, GLCM, & Morphological Features...")
    extractor = FeatureExtractor(glcm_levels=16)
    feat_res = extractor.extract_all_features(pre_res["raw_gray_image"])
    print(f"[+] Extracted {len(feat_res['feature_vector'])} clinical features:")
    for name, val in zip(feat_res["feature_names"], feat_res["feature_vector"]):
        print(f"    - {name:<22}: {val}")

    print("\n[*] Step 3: Running Machine Learning Prediction...")
    trainer = HeartDiseaseMLTrainer()
    pred = trainer.predict_from_features(feat_res["feature_vector"])

    print("=" * 60)
    print("DIAGNOSTIC REPORT SUMMARY")
    print("=" * 60)
    print(f"Diagnostic Outcome  : {pred['prediction']}")
    print(f"Confidence Score    : {pred['confidence']}%")
    print(f"Risk Stratification : {pred['risk_level']} (Severity Index: {pred['severity_index']}/100)")
    print(f"Model In Use        : {pred['model_used']}")
    print(f"Clinical Summary    : {pred['clinical_summary']}")
    print(f"Action Recommendation: {pred['recommendation']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Prediction CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to cardiac/ECG image")
    args = parser.parse_args()
    run_prediction_cli(args.image)
