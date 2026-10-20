"""
Heart Disease Prediction System - Medical Sample Image Generator
Generates realistic clinical ECG traces and cardiac image samples for the 5 classes:
1. Normal Sinus Rhythm
2. Coronary Artery Disease (Ischemia)
3. Acute Myocardial Infarction (STEMI)
4. Arrhythmia (Atrial Fibrillation)
5. Cardiomegaly / Congestive Heart Failure
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

STATIC_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")
os.makedirs(STATIC_SAMPLES_DIR, exist_ok=True)


def draw_ecg_grid(draw, width, height, grid_color=(235, 245, 238), major_color=(210, 230, 215)):
    """Draws standard medical millimeter ECG grid background (small squares 10px, major squares 50px)."""
    # Small grid lines
    for x in range(0, width, 10):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 10):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Major grid lines (every 50px)
    for x in range(0, width, 50):
        draw.line([(x, 0), (x, height)], fill=major_color, width=1)
    for y in range(0, height, 50):
        draw.line([(0, y), (width, y)], fill=major_color, width=1)


def generate_normal_ecg(width=500, height=260):
    """Generates standard Normal Sinus Rhythm (Lead II)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_ecg_grid(draw, width, height)

    baseline = height // 2
    period = 110  # regular R-R interval (~72 BPM)
    points = []

    for x in range(width):
        phase = x % period
        y = baseline

        # P wave (small upward deflection)
        if 20 <= phase < 35:
            p_rel = (phase - 20) / 15.0
            y -= math.sin(p_rel * math.pi) * 12
        # PR segment
        elif 35 <= phase < 45:
            y = baseline
        # Q wave (small downward deflection)
        elif 45 <= phase < 49:
            y += (phase - 45) * 4
        # R wave (tall sharp spike)
        elif 49 <= phase < 54:
            y -= 85 - (phase - 49) * 16
        # S wave (sharp downward deflection)
        elif 54 <= phase < 59:
            y += 20 - (59 - phase) * 4
        # ST segment (flat)
        elif 59 <= phase < 70:
            y = baseline
        # T wave (rounded upward deflection)
        elif 70 <= phase < 95:
            t_rel = (phase - 70) / 25.0
            y -= math.sin(t_rel * math.pi) * 22

        points.append((x, y))

    # Draw continuous ECG trace with anti-aliased look
    draw.line(points, fill=(15, 23, 42), width=2)

    # Header label
    draw.rectangle([(10, 10), (220, 34)], fill=(240, 253, 244), outline=(187, 247, 208))
    draw.text((16, 14), "LEAD II - NORMAL SINUS RHYTHM", fill=(22, 101, 52))

    return img


def generate_mi_ecg(width=500, height=260):
    """Generates Acute Myocardial Infarction (STEMI - ST Elevation & Pathological Q)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_ecg_grid(draw, width, height, grid_color=(254, 242, 242), major_color=(254, 202, 202))

    baseline = height // 2
    period = 90  # faster heart rate (~105 BPM)
    points = []

    for x in range(width):
        phase = x % period
        y = baseline

        # P wave
        if 15 <= phase < 28:
            p_rel = (phase - 15) / 13.0
            y -= math.sin(p_rel * math.pi) * 10
        # Deep Pathological Q-wave
        elif 35 <= phase < 42:
            y += 35
        # R wave
        elif 42 <= phase < 48:
            y -= 75
        # ST Elevation (Tombstone pattern - ST segment severely elevated)
        elif 48 <= phase < 72:
            st_rel = (phase - 48) / 24.0
            # Elevated ST segment that merges with hyperacute T wave
            y -= 45 + math.sin(st_rel * math.pi) * 20
        elif 72 <= phase < 80:
            y -= (80 - phase) * 4

        points.append((x, y))

    draw.line(points, fill=(185, 28, 28), width=2)
    draw.rectangle([(10, 10), (280, 34)], fill=(254, 226, 226), outline=(239, 68, 68))
    draw.text((16, 14), "CRITICAL: ACUTE STEMI (ST-ELEVATION)", fill=(153, 27, 27))

    return img


def generate_arrhythmia_ecg(width=500, height=260):
    """Generates Atrial Fibrillation / Arrhythmia (Irregular R-R intervals, fibrillatory waves)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_ecg_grid(draw, width, height, grid_color=(250, 245, 255), major_color=(233, 213, 255))

    baseline = height // 2
    # Irregular R-peak timings
    r_peaks = [35, 80, 150, 210, 245, 330, 380, 440, 485]
    points = []

    for x in range(width):
        y = baseline
        # Baseline chaotic fibrillatory 'f' waves
        f_wave = math.sin(x * 0.4) * 4 + math.cos(x * 0.9) * 3
        y += f_wave

        # Check proximity to any R peak
        for r_x in r_peaks:
            dist = x - r_x
            if -3 <= dist <= 3:
                # Sharp QRS spike
                y -= (100 - abs(dist) * 28)
            elif 4 <= dist <= 8:
                # S wave
                y += 18 - (dist - 4) * 3

        points.append((x, y))

    draw.line(points, fill=(126, 34, 206), width=2)
    draw.rectangle([(10, 10), (290, 34)], fill=(243, 232, 255), outline=(192, 132, 252))
    draw.text((16, 14), "ATRIAL FIBRILLATION - IRREGULAR R-R", fill=(88, 28, 135))

    return img


def generate_cad_ecg(width=500, height=260):
    """Generates Coronary Artery Disease / Myocardial Ischemia (T-wave inversion, ST depression)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_ecg_grid(draw, width, height, grid_color=(254, 249, 195), major_color=(253, 224, 71))

    baseline = height // 2
    period = 100
    points = []

    for x in range(width):
        phase = x % period
        y = baseline

        # P wave
        if 20 <= phase < 34:
            p_rel = (phase - 20) / 14.0
            y -= math.sin(p_rel * math.pi) * 11
        # QRS complex
        elif 45 <= phase < 48:
            y += 8
        elif 48 <= phase < 53:
            y -= 80
        elif 53 <= phase < 57:
            y += 24
        # ST depression
        elif 57 <= phase < 68:
            y += 18  # Sub-baseline ST depression
        # Deep Symmetrical Inverted T-wave (Classic CAD ischemia sign)
        elif 68 <= phase < 90:
            t_rel = (phase - 68) / 22.0
            y += math.sin(t_rel * math.pi) * 28  # Inverted below baseline

        points.append((x, y))

    draw.line(points, fill=(180, 83, 9), width=2)
    draw.rectangle([(10, 10), (280, 34)], fill=(254, 240, 138), outline=(234, 179, 8))
    draw.text((16, 14), "ISCHEMIA - T-WAVE INVERSION (CAD)", fill=(113, 63, 18))

    return img


def generate_cardiomegaly_scan(width=500, height=260):
    """Generates Chest Radiography silhouette with enlarged cardiothoracic ratio (>0.65)."""
    img = Image.new("L", (width, height), color=25)
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Draw Ribcage / Thoracic Cavity contours
    draw.ellipse([cx - 210, cy - 110, cx + 210, cy + 120], fill=65, outline=110, width=2)

    # Lungs (dark radiolucent fields)
    draw.ellipse([cx - 190, cy - 85, cx - 40, cy + 90], fill=30)
    draw.ellipse([cx + 40, cy - 85, cx + 190, cy + 90], fill=30)

    # Enlarged Heart Silhouette (Severe Cardiomegaly - apex shifts far left)
    # Left ventricular apex enlarged beyond mid-clavicular line
    heart_bbox = [cx - 130, cy - 50, cx + 155, cy + 85]
    draw.ellipse(heart_bbox, fill=165, outline=220, width=2)

    # Aortic knuckle and pulmonary trunk
    draw.ellipse([cx - 35, cy - 70, cx + 25, cy - 30], fill=145)

    # Convert to RGB for UI consistency
    img_rgb = img.convert("RGB")
    draw_rgb = ImageDraw.Draw(img_rgb)
    draw_rgb.rectangle([(10, 10), (280, 34)], fill=(255, 237, 213), outline=(249, 115, 22))
    draw_rgb.text((16, 14), "CHEST SCAN: CARDIOMEGALY (CTR > 0.65)", fill=(154, 52, 18))

    return img_rgb


def create_all_samples():
    """Builds and saves the full sample gallery into static/samples/."""
    samples = {
        "normal_sinus_rhythm.png": generate_normal_ecg(),
        "myocardial_infarction.png": generate_mi_ecg(),
        "arrhythmia_atrial_fib.png": generate_arrhythmia_ecg(),
        "coronary_artery_disease.png": generate_cad_ecg(),
        "cardiomegaly_heart_failure.png": generate_cardiomegaly_scan(),
    }

    for filename, img in samples.items():
        save_path = os.path.join(STATIC_SAMPLES_DIR, filename)
        img.save(save_path)
        print(f"Generated sample image: {save_path}")

    return list(samples.keys())


if __name__ == "__main__":
    print("Generating medical sample images...")
    created = create_all_samples()
    print(f"Successfully generated {len(created)} sample medical images!")
