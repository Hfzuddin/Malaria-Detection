import os
import sys
import joblib
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime

# ============================================================================
# 0. CONFIGURATION & SETUP
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

MODEL_FILENAME = 'model_70_128D_30_3Class.pkl' 
MODEL_PAYLOAD_PATH = os.path.join(BASE_DIR, 'model', MODEL_FILENAME)
FEATURE_DIMENSION = 128 

PROCESSED_FOLDER = os.path.join(PROJECT_ROOT, 'processed_images')
IMG_SIZE = (224, 224)

MAX_MB = 500
MAX_FILES = 500

MIN_CELL_AREA = 500
MAX_CELL_AREA = 50000
MIN_CIRCULARITY = 0.3

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024 
app.config['SECRET_KEY'] = 'malaria_secret_key'

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

svm_model = None
scaler = None
cnn_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOADED_METRICS = {}
CLASS_NAMES = []
CLASS_TO_IDX = {}
analysis_results = []

# ============================================================================
# 1. MODEL ARCHITECTURE (Updated for 3-class)
# ============================================================================
class HybridEfficientNet(nn.Module):
    def __init__(self, feature_dim=128, num_classes=3):
        super(HybridEfficientNet, self).__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.backbone = models.efficientnet_b3(weights=weights)
        self.backbone.classifier = nn.Identity()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            in_features = out.shape[1]

        self.custom_head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.45),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.custom_head(self.backbone(x))
        return features

# ============================================================================
# 2. LOAD MODEL
# ============================================================================
try:
    if not os.path.exists(MODEL_PAYLOAD_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PAYLOAD_PATH}")

    # Monkey-patch torch.load to force CPU loading for GPU-trained models
    _original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _original_torch_load(
        *args, **{**kwargs, 'map_location': 'cpu', 'weights_only': False}
    )
    try:
        payload = joblib.load(MODEL_PAYLOAD_PATH)
    finally:
        torch.load = _original_torch_load  # Always restore original

    svm_model = payload['svm_model']
    scaler = payload['scaler']
    cnn_state_dict = payload['cnn_state_dict']
        
    LOADED_METRICS = payload.get('metrics', {})
    CLASS_NAMES = payload.get('class_names', ['Non-bloodCell', 'Parasitized', 'Uninfected'])
    CLASS_TO_IDX = payload.get('class_to_idx', {})
    
    saved_config = payload.get('config', {})
    dim_to_use = saved_config.get('feature_dim', FEATURE_DIMENSION)
    num_classes = saved_config.get('num_classes', 3)
    
    cnn_model = HybridEfficientNet(feature_dim=dim_to_use, num_classes=num_classes)
    cnn_model.load_state_dict(cnn_state_dict, strict=False)
    cnn_model.to(device)
    cnn_model.eval()

except Exception as e:
    print(f"FATAL ERROR: {e}")
    sys.exit(1)

# ============================================================================
# 3. CORE AI FUNCTIONS
# ============================================================================

def preprocess_single_cell(img_cv2):
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()
    return img_tensor.unsqueeze(0).to(device)

def predict_logic(img_cv2):
    try:
        img_tensor = preprocess_single_cell(img_cv2)
        with torch.no_grad():
            features = cnn_model(img_tensor).cpu().numpy()
            
        features_scaled = scaler.transform(features)
        probs = svm_model.predict_proba(features_scaled)[0]
        
        # 3-class prediction
        # probs[0] = Non-bloodCell
        # probs[1] = Parasitized (INFECTED)
        # probs[2] = Uninfected
        
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        if predicted_class == 0:  # Non-bloodCell
            return "NON_BLOOD", confidence, False
        elif predicted_class == 1:  # Parasitized
            return "INFECTED", confidence, True
        else:  # Uninfected
            return "UNINFECTED", confidence, False
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return "ERROR", 0.0, False

def highlight_parasite(img_bgr):
    """
    Detects and highlights malaria parasites with red color.
    Parasites appear as dark purple/blue spots inside blood cells.
    """
    img_highlighted = img_bgr.copy()
    height, width = img_bgr.shape[:2]
    img_area = height * width
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Create cell mask (exclude black background)
    # Black background has very low value in all channels
    mask_not_black = cv2.inRange(img_bgr, np.array([10, 10, 10]), np.array([255, 255, 255]))
    
    # Also exclude pure black using grayscale
    _, mask_not_dark_bg = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    # Combine to get cell region only
    cell_mask = cv2.bitwise_and(mask_not_black, mask_not_dark_bg)
    
    # Clean up cell mask
    kernel = np.ones((5, 5), np.uint8)
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 2: Detect parasite colors (dark purple/violet spots)
    # Parasites are darker purple compared to the light pink/purple cell
    mask_dark_purple = cv2.inRange(hsv, np.array([100, 40, 20]), np.array([160, 255, 150]))
    
    # Step 3: Detect dark spots within cell using adaptive threshold
    # This catches the darker stained regions
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 15, 5)
    
    # Step 4: Combine parasite detection methods
    parasite_mask = cv2.bitwise_or(mask_dark_purple, adaptive_thresh)
    
    # Step 5: IMPORTANT - Only keep parasites INSIDE the cell (exclude background)
    parasite_mask = cv2.bitwise_and(parasite_mask, cell_mask)
    
    # Step 6: Clean up noise
    kernel_small = np.ones((3, 3), np.uint8)
    parasite_mask = cv2.morphologyEx(parasite_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    parasite_mask = cv2.morphologyEx(parasite_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Step 7: Filter by size - remove tiny noise and huge regions
    contours, _ = cv2.findContours(parasite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_mask = np.zeros_like(parasite_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Keep spots that are reasonable parasite size
        # Not too small (noise) and not too big (whole cell)
        if (img_area * 0.002) < area < (img_area * 0.25):
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    
    # Step 8: Create red highlight overlay
    red_overlay = img_highlighted.copy()
    red_overlay[clean_mask > 0] = [0, 0, 255]  # BGR: Red color
    
    # Blend original with red overlay (semi-transparent)
    alpha = 0.5
    img_highlighted = cv2.addWeighted(img_bgr, 1 - alpha, red_overlay, alpha, 0)
    
    # Draw red contour outline for clarity
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_highlighted, contours, -1, (0, 0, 255), 2)
    
    return img_highlighted, clean_mask

def is_full_slide_image(img_bgr):
    height, width = img_bgr.shape[:2]
    img_area = height * width
    
    if img_area < 100000:
        return False
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cell_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CELL_AREA < area < MAX_CELL_AREA:
            cell_count += 1
    
    return cell_count >= 3

def extract_cells_from_slide(img_bgr):
    height, width = img_bgr.shape[:2]
    cells = []
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CELL_AREA or area > MAX_CELL_AREA:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if circularity < MIN_CIRCULARITY:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        cell_img = img_bgr[y1:y2, x1:x2].copy()
        if cell_img.size > 0:
            cells.append(cell_img)
    
    return cells

# ============================================================================
# 4. FULL SLIDE ANALYZER
# ============================================================================

def analyze_full_slide(img_orig, filename, start_time):
    height, width = img_orig.shape[:2]
    
    cells = extract_cells_from_slide(img_orig)
    
    if len(cells) == 0:
        return {
            'filename': filename,
            'diagnosis': "INVALID INPUT",
            'error_msg': "No valid blood cells detected. Please ensure image shows stained blood cells."
        }
    
    infected_count = 0
    uninfected_count = 0
    non_blood_count = 0
    total_confidence = 0.0
    valid_cells = 0
    
    for cell_img in cells:
        label, conf, is_infected = predict_logic(cell_img)
        if label == "INFECTED":
            infected_count += 1
            valid_cells += 1
            total_confidence += conf
        elif label == "UNINFECTED":
            uninfected_count += 1
            valid_cells += 1
            total_confidence += conf
        elif label == "NON_BLOOD":
            non_blood_count += 1
    
    if valid_cells == 0:
        return {
            'filename': filename,
            'diagnosis': "INVALID INPUT",
            'error_msg': "No valid blood cells detected. Please ensure image shows stained blood cells."
        }
    
    avg_confidence = (total_confidence / valid_cells) * 100
    
    if infected_count > 0:
        diagnosis = "INFECTED"
        is_infected = True
    else:
        diagnosis = "UNINFECTED"
        is_infected = False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = os.path.splitext(secure_filename(filename))[0]
    orig_name = f"orig_{timestamp}_{base_name}.png"
    proc_name = f"proc_{timestamp}_{base_name}.png"
    
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], orig_name), img_orig)
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], proc_name), img_orig)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'filename': filename,
        'diagnosis': diagnosis,
        'confidence': round(avg_confidence, 1),
        'is_infected': is_infected,
        'processing_time': round(processing_time, 2),
        'image_dimensions': f"{width}x{height}",
        'original_image': orig_name,
        'preprocessed_image': proc_name
    }

# ============================================================================
# 5. SINGLE CELL ANALYZER
# ============================================================================

def analyze_single_cell_image(file_stream, filename):
    start_time = datetime.now()

    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_orig is None:
        return {'filename': filename, 'diagnosis': "INVALID INPUT", 'error_msg': "No valid blood cells detected. Please ensure image shows stained blood cells."}

    height, width = img_orig.shape[:2]

    if is_full_slide_image(img_orig):
        return analyze_full_slide(img_orig, filename, start_time)

    final_label, conf_float, final_infected = predict_logic(img_orig)
    final_conf = conf_float * 100

    # Handle Non-bloodCell detection from model
    if final_label == "NON_BLOOD":
        return {
            'filename': filename, 
            'diagnosis': "INVALID INPUT", 
            'confidence': round(final_conf, 1),
            'is_infected': False, 
            'error_msg': "No valid blood cells detected. Please ensure image shows stained blood cells.",
            'original_image': "", 
            'preprocessed_image': ""
        }

    if final_label == "ERROR":
        return {
            'filename': filename, 
            'diagnosis': "INVALID INPUT", 
            'confidence': 0.0,
            'is_infected': False, 
            'error_msg': "Prediction error occurred",
            'original_image': "", 
            'preprocessed_image': ""
        }

    # Create highlighted image if infected
    if final_infected:
        img_display, _ = highlight_parasite(img_orig)
    else:
        img_display = img_orig.copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = os.path.splitext(secure_filename(filename))[0]
    orig_name = f"orig_{timestamp}_{base_name}.png"
    proc_name = f"proc_{timestamp}_{base_name}.png"
    
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], orig_name), img_orig)
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], proc_name), img_display)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'filename': filename,
        'diagnosis': final_label,
        'confidence': round(final_conf, 1),
        'is_infected': final_infected,
        'processing_time': round(processing_time, 2),
        'image_dimensions': f"{width}x{height}",
        'original_image': orig_name,
        'preprocessed_image': proc_name
    }

# ============================================================================
# 6. ROUTES
# ============================================================================
@app.route('/')
@app.route('/gallery')
@app.route('/history')
@app.route('/result')
def index(): 
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_images_route():
    global analysis_results

    
    if 'images' not in request.files: return jsonify({'error': 'No files'}), 400
    files = request.files.getlist('images')
    
    if len(files) > MAX_FILES:
        return jsonify({'error': f'Max {MAX_FILES} files allowed.'}), 400

    for file in files:
        if file and '.' in file.filename:
            file.seek(0)
            res = analyze_single_cell_image(file, file.filename)
            analysis_results.append(res)
        
    return jsonify({
        'success': True, 
        'total_analyzed': len(analysis_results),
        'results': analysis_results 
    })


@app.route('/api/results')
def get_results(): return jsonify(analysis_results)

@app.route('/api/model_metrics')
def get_metrics():
    global LOADED_METRICS
    return jsonify({
        'model_name': MODEL_FILENAME,
        'accuracy': LOADED_METRICS.get('accuracy', 'N/A'),
        'precision': LOADED_METRICS.get('precision', 'N/A'),
        'recall': LOADED_METRICS.get('recall', 'N/A'),
        'f1': LOADED_METRICS.get('f1', 'N/A')
    })

@app.route('/processed_images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/results_images/<filename>')
def serve_result_image(filename):
    results_folder = os.path.join(BASE_DIR, 'results')
    return send_from_directory(results_folder, filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File too large! Max {MAX_MB}MB.'}), 413

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" MALARIA DETECTION SYSTEM")
    print("="*50)
    print(f" Model Used  : {MODEL_FILENAME}")
    print(f" Features    : {dim_to_use}D")
    print("-"*50)
    print(f" Accuracy    : {LOADED_METRICS.get('accuracy', 0)*100:.2f}%")
    print(f" Precision   : {LOADED_METRICS.get('precision', 0)*100:.2f}%")
    print(f" Recall      : {LOADED_METRICS.get('recall', 0)*100:.2f}%")
    print(f" F1-Score    : {LOADED_METRICS.get('f1', 0)*100:.2f}%")
    print("="*50)
    print(" Server      : http://127.0.0.1:5500")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', debug=True, port=5500)