import os
import base64
import gc
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
from src.model_downloader import ensure_models
from src.model_loader import ModelManager

app = Flask(__name__)

global_model_manager = None

def get_model_manager():
    """Loads the model manager structure only when needed."""
    global global_model_manager
    if global_model_manager is None:
        print("⏳ Loading AI Model Manager...")
        try:
            ensure_models()
            global_model_manager = ModelManager()
            print("✅ AI Core Ready.")
        except Exception as e:
            print(f"❌ Critical Error Initializing ModelManager: {e}")
            return None
    return global_model_manager

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
    manager = get_model_manager()
    
    if not manager:
        return jsonify({'error': 'AI Model failed to initialize. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')

        fusion_result = manager.predict_fusion_classification(image)
        
        det_results_raw = manager.predict_multi_detection(image)
        det_response = {k: encode_image(v) for k, v in det_results_raw.items()}

        seg_results_raw = manager.predict_multi_segmentation(image)
        seg_response = {k: encode_image(v) for k, v in seg_results_raw.items()}

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
    port = int(os.environ.get("PORT", 5000))  
    app.run(debug=True, host='0.0.0.0', port=port)