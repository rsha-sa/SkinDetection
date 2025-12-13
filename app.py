import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
from src.model_downloader import ensure_models

from src.model_loader import ModelManager

app = Flask(__name__)

# Initialize AI Core
print("Initializing AI Core...")
try:
    ensure_models()  
    model_manager = ModelManager()
    print("✅ AI Core Ready.")
except Exception as e:
    print(f"❌ Critical Error Initializing ModelManager: {e}")
    model_manager = None

def encode_image(image):
    """Helper to convert PIL Image to Base64 string for HTML display"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model_manager:
        return jsonify({'error': 'AI Model not initialized. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load Image
        image = Image.open(file.stream).convert('RGB')
        
        # 1. Run Classification
        fusion_result = model_manager.predict_fusion_classification(image)
        
        # 2. Run Detection
        det_results_raw = model_manager.predict_multi_detection(image)
        det_response = {k: encode_image(v) for k, v in det_results_raw.items()}

        # 3. Run Segmentation
        seg_results_raw = model_manager.predict_multi_segmentation(image)
        seg_response = {k: encode_image(v) for k, v in seg_results_raw.items()}

        # Construct JSON Response
        response_data = {
            'classification': fusion_result, 
            'detection': det_response,     
            'segmentation': seg_response   
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc() 
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)