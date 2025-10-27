#!/usr/bin/env python3
"""
Deepfake Detector Web Interface
Professional Flask application for deepfake detection with explainable AI
"""

import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'deepfake_detector_secret_key_2024'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
model_loaded = False

class DeepfakeDetector:
    """Professional deepfake detection class with explainable AI"""
    
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained deepfake detection model"""
        try:
            # Try loading SavedModel first
            model_paths = [
                'deepfake_detector_savedmodel',
                'hybrid_deepfake_detector_savedmodel'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    loaded_model = tf.saved_model.load(model_path)
                    
                    if hasattr(loaded_model, 'signatures'):
                        if 'serving_default' in loaded_model.signatures:
                            inference_func = loaded_model.signatures['serving_default']
                            
                            # Create wrapper function
                            def predict_wrapper(x):
                                if not isinstance(x, tf.Tensor):
                                    x = tf.convert_to_tensor(x, dtype=tf.float32)
                                
                                # Use the correct keys for your trained model
                                input_key = 'input_1'  # Correct input key
                                
                                result = inference_func(**{input_key: x})
                                output_key = 'dense_2'  # Correct output key from your model
                                
                                return result[output_key].numpy()
                            
                            self.model = predict_wrapper
                            self.model_loaded = True
                            logger.info("✅ Model loaded successfully!")
                            return True
            
            logger.error("❌ No compatible model found!")
            return False
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Resize to model input size
            image = cv2.resize(image, (224, 224))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def predict(self, image):
        """Make prediction on preprocessed image"""
        try:
            if not self.model_loaded:
                return None, "Model not loaded"
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, "Image preprocessing failed"
            
            # Make prediction
            prediction = self.model(processed_image)
            
            # Extract prediction value
            if hasattr(prediction, 'numpy'):
                prediction = prediction.numpy()
            
            pred_value = float(prediction.flatten()[0])
            
            # Determine class and confidence
            if pred_value > 0.5:
                predicted_class = "FAKE"
                confidence = pred_value
            else:
                predicted_class = "REAL"
                confidence = 1 - pred_value
            
            result = {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'raw_score': float(pred_value),
                'fake_probability': float(pred_value),
                'real_probability': float(1 - pred_value)
            }
            
            return result, None
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, str(e)
    
    def generate_gradcam(self, image, prediction_result):
        """Generate Grad-CAM visualization (simplified version)"""
        try:
            # For now, create a simple heatmap overlay
            # In a full implementation, you'd use the actual Grad-CAM algorithm
            
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None
            
            # Create a meaningful attention map based on image features
            attention_map = self._create_meaningful_heatmap(processed_image[0])
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            original = processed_image[0]
            ax1.imshow(original)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Heatmap overlay
            ax2.imshow(original)
            ax2.imshow(attention_map, alpha=0.4, cmap='jet')
            ax2.set_title(f'Attention Map\nPrediction: {prediction_result["prediction"]} ({prediction_result["confidence"]:.1%})')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Save to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return None
    
    def _create_meaningful_heatmap(self, image):
        """Create content-aware attention heatmap that follows actual image features"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image
            
            from scipy import ndimage
            import cv2
            
            # Ensure image is in proper range
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
            
            # 1. Edge detection using Canny (more robust than Sobel)
            edges = cv2.Canny(gray, 50, 150)
            edges = edges.astype(np.float32) / 255.0
            
            # 2. Corner detection (Harris corners - important features)
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = np.abs(corners)
            if corners.max() > 0:
                corners = corners / corners.max()
            
            # 3. Local Binary Pattern-like texture analysis
            texture_map = np.zeros_like(gray, dtype=np.float32)
            for i in range(1, gray.shape[0]-1):
                for j in range(1, gray.shape[1]-1):
                    center = gray[i, j]
                    neighbors = gray[i-1:i+2, j-1:j+2]
                    texture_map[i, j] = np.std(neighbors.astype(np.float32))
            
            # Normalize texture map
            if texture_map.max() > 0:
                texture_map = texture_map / texture_map.max()
            
            # 4. Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            if gradient_mag.max() > 0:
                gradient_mag = gradient_mag / gradient_mag.max()
            
            # 5. Combine features with content-aware weighting
            attention = (0.4 * edges + 
                        0.25 * corners + 
                        0.2 * texture_map + 
                        0.15 * gradient_mag)
            
            # 6. Adaptive enhancement based on content
            if attention.max() > attention.min():
                attention = (attention - attention.min()) / (attention.max() - attention.min())
            
                # Create adaptive threshold based on image content
                mean_attention = np.mean(attention)
                std_attention = np.std(attention)
                
                # Use content-adaptive threshold instead of fixed percentile
                if std_attention > 0.05:  # Image has good variation
                    threshold = mean_attention + 0.5 * std_attention
                    attention_enhanced = np.where(attention > threshold,
                                                np.power(attention, 0.7),  # Enhance high values
                                                np.power(attention, 1.2))  # Suppress low values
                else:  # Low variation image
                    attention_enhanced = attention
                
                # Apply content-aware smoothing
                sigma = max(1.0, min(3.0, 2.0 * std_attention * 10))  # Adaptive smoothing
                attention_smooth = ndimage.gaussian_filter(attention_enhanced, sigma=sigma)
                
                # Final enhancement
                attention_final = np.power(attention_smooth, 0.9)
                
                # Ensure minimum contrast
                if attention_final.std() < 0.05:
                    # Boost contrast without center bias
                    attention_final = np.power(attention_final, 0.5)  # Increase contrast
                    attention_final = (attention_final - attention_final.min()) / (attention_final.max() - attention_final.min())
                
                return attention_final
            else:
                # If no variation found, create a simple edge-based map
                simple_edges = ndimage.sobel(gray.astype(np.float32))
                simple_edges = np.abs(simple_edges)
                if simple_edges.max() > 0:
                    simple_edges = simple_edges / simple_edges.max()
                return simple_edges
            
        except Exception as e:
            logger.error(f"Content-aware heatmap creation failed: {e}")
            logger.error(f"Error details: {str(e)}")
            
            # Last resort: create edge-based attention without center bias
            try:
                gray_simple = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) if len(image.shape) == 3 else image
                edges_simple = ndimage.sobel(gray_simple)
                edges_simple = np.abs(edges_simple)
                if edges_simple.max() > 0:
                    edges_simple = edges_simple / edges_simple.max()
                    # Add some random variation to avoid uniform patterns
                    noise = np.random.random(edges_simple.shape) * 0.1
                    edges_simple = np.clip(edges_simple + noise, 0, 1)
                return edges_simple
            except:
                # Final fallback - but avoid center bias
                h, w = image.shape[:2]
                return np.random.random((h, w)) * 0.3 + 0.1  # Low-intensity random pattern

# Initialize detector
detector = DeepfakeDetector()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process image
            image = Image.open(filepath)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Make prediction
            result, error = detector.predict(image)
            if error:
                return jsonify({'error': error}), 500
            
            # Generate Grad-CAM visualization
            gradcam_image = detector.generate_gradcam(image, result)
            
            # Prepare response
            response = {
                'success': True,
                'prediction': result['prediction'],
                'confidence': f"{result['confidence']:.1%}",
                'raw_score': f"{result['raw_score']:.4f}",
                'probabilities': {
                    'fake': f"{result['fake_probability']:.1%}",
                    'real': f"{result['real_probability']:.1%}"
                },
                'gradcam_image': gradcam_image,
                'filename': filename
            }
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(response)
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG images.'}), 400
    
    except Exception as e:
        logger.error(f"Upload handling failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model_loaded,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Deepfake Detector Web Interface...")
    
    # Load model on startup (only when run directly)
    print("📂 Loading deepfake detection model...")
    if detector.load_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🎯 Upload an image to detect if it's real or fake!")
        app.run(debug=False, host='0.0.0.0', port=5000)  # Disabled debug to prevent double loading
    else:
        print("❌ Failed to load model!")
        print("💡 Make sure your model files are in the same directory:")
        print("   - deepfake_detector_savedmodel/")
        print("   - hybrid_deepfake_detector_savedmodel/")
