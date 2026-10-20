"""
Heart Disease Prediction System - Image Preprocessor Module
Stages:
1. Load Image (from File Path or Base64 / Bytes)
2. Grayscale Conversion
3. Gaussian / Median Noise Filtering
4. Contrast Enhancement / Histogram Equalization
5. Binarization & Adaptive Thresholding
6. Edge Detection (Sobel / Gradient Filter)
7. Region of Interest (ROI) Resizing & Normalization
"""

import io
import math
import base64
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class ImagePreprocessor:
    """Handles the multi-stage image preprocessing pipeline for cardiac & ECG images."""

    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def load_image(self, input_source):
        """Loads an image from filepath, raw bytes, or base64 string."""
        if isinstance(input_source, Image.Image):
            return input_source.convert("RGB")

        if isinstance(input_source, str):
            if input_source.startswith("data:image"):
                # Base64 data URL
                header, encoded = input_source.split(",", 1)
                data = base64.b64decode(encoded)
                return Image.open(io.BytesIO(data)).convert("RGB")
            else:
                # Regular file path
                return Image.open(input_source).convert("RGB")

        if isinstance(input_source, (bytes, bytearray)):
            return Image.open(io.BytesIO(input_source)).convert("RGB")

        raise ValueError("Unsupported input format for loading image.")

    def resize_image(self, img, size=None):
        """Resize image to standard dimensions for uniform ML processing."""
        target = size or self.target_size
        return img.resize(target, Image.Resampling.LANCZOS)

    def to_grayscale(self, img):
        """Converts an RGB image to single-channel 8-bit Grayscale."""
        return img.convert("L")

    def apply_gaussian_filter(self, img, radius=1.5):
        """Applies Gaussian smoothing to suppress high-frequency baseline noise."""
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_median_filter(self, img, size=3):
        """Applies Median filtering to eliminate salt-and-pepper noise."""
        # PIL MedianFilter supports size 3 or 5
        k_size = 3 if size <= 3 else 5
        return img.filter(ImageFilter.MedianFilter(size=k_size))

    def enhance_contrast(self, img, factor=1.5):
        """Applies adaptive contrast enhancement to accentuate heart chambers or ECG waveforms."""
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(factor)
        # Apply Histogram Equalization
        if enhanced.mode == "L":
            return ImageOps.equalize(enhanced)
        return enhanced

    def apply_threshold(self, gray_img, threshold=128):
        """Binarizes grayscale image using a configurable threshold cutoff."""
        if gray_img.mode != "L":
            gray_img = gray_img.convert("L")
        return gray_img.point(lambda p: 255 if p > threshold else 0, mode="1").convert("L")

    def apply_edge_detection(self, gray_img):
        """Detects boundaries using Sobel/FIND_EDGES gradient operator."""
        if gray_img.mode != "L":
            gray_img = gray_img.convert("L")
        return gray_img.filter(ImageFilter.FIND_EDGES)

    def process_full_pipeline(self, input_source, contrast_factor=1.6, noise_radius=1.2, threshold_val=130):
        """
        Executes the complete step-by-step preprocessor pipeline and returns
        images and metadata at every single stage for visual inspection.
        """
        raw_img = self.load_image(input_source)
        resized_img = self.resize_image(raw_img)

        # 1. Grayscale
        gray_img = self.to_grayscale(resized_img)

        # 2. Denoised / Gaussian Filtered
        denoised_img = self.apply_gaussian_filter(gray_img, radius=noise_radius)

        # 3. Contrast Enhanced / Equalized
        contrast_img = self.enhance_contrast(denoised_img, factor=contrast_factor)

        # 4. Thresholding / Segmented
        threshold_img = self.apply_threshold(contrast_img, threshold=threshold_val)

        # 5. Edge Detected
        edge_img = self.apply_edge_detection(contrast_img)

        # Compute Pixel Histogram for Visualization (256 bins)
        hist = gray_img.histogram()

        return {
            "original": self.image_to_base64(resized_img),
            "grayscale": self.image_to_base64(gray_img),
            "denoised": self.image_to_base64(denoised_img),
            "contrast_enhanced": self.image_to_base64(contrast_img),
            "thresholded": self.image_to_base64(threshold_img),
            "edges": self.image_to_base64(edge_img),
            "histogram": hist,
            "dimensions": {"width": resized_img.width, "height": resized_img.height},
            "raw_gray_image": gray_img,
            "raw_contrast_image": contrast_img,
            "raw_edges_image": edge_img
        }

    @staticmethod
    def image_to_base64(img, format="PNG"):
        """Encodes a PIL Image into a base64 Data URL string for web rendering."""
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"


if __name__ == "__main__":
    # Test script standalone execution
    print("Testing Preprocessor Module...")
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    test_img = Image.new("RGB", (128, 128), color=(255, 120, 100))
    result = preprocessor.process_full_pipeline(test_img)
    print("Preprocessor initialized successfully!")
    print(f"Pipeline stages produced: {list(result.keys())}")
