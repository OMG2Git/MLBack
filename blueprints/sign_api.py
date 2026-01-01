"""
Sign Language Translator API Blueprint
Real-time ASL (American Sign Language) recognition using CNN
Model: Convolutional Neural Network | Accuracy: 98.93%
"""

from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import pickle
import base64
import os
from io import BytesIO
from PIL import Image

# Create Blueprint
sign_bp = Blueprint('sign', __name__)

# Model paths
MODEL_PATH = os.path.join('models', 'sign_language_model.h5')
LABEL_MAP_PATH = os.path.join('models', 'label_map.pkl')

# Load model and label map
print("👋 Loading Sign Language model...")
try:
    from tensorflow.keras.models import load_model
    
    model = load_model(MODEL_PATH)
    
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    
    print(f"✅ Sign Language model loaded successfully!")
    print(f"✅ Label map loaded: {len(label_map)} classes")
    print(f"✅ Available letters: {', '.join([label_map[i] for i in sorted(label_map.keys())])}")
    
except FileNotFoundError as e:
    print(f"❌ Error loading sign language model: {e}")
    print(f"⚠️  Make sure '{MODEL_PATH}' and '{LABEL_MAP_PATH}' exist")
    model = None
    label_map = {}
except Exception as e:
    print(f"❌ Unexpected error loading model: {e}")
    model = None
    label_map = {}


def extract_hand_region(img_array):
    """Extract hand region using simple thresholding"""
    try:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Apply binary threshold to isolate hand
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            
            # Crop to hand region
            hand_region = img_array[y:y+h, x:x+w]
            return hand_region
        
        return img_array
    except Exception as e:
        print(f"Hand extraction warning: {e}")
        return img_array


def preprocess_image(image_data):
    """Preprocess image for CNN prediction"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to numpy array (RGB)
        img_array = np.array(img)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Optional: Extract hand region (uncomment if needed)
        # img_gray = extract_hand_region(img_gray)
        
        # Resize to 28x28 (model input size)
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Reshape for model input: (batch_size, height, width, channels)
        img_final = img_normalized.reshape(1, 28, 28, 1)
        
        return img_final
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


@sign_bp.route('/')
def home():
    """Sign Language API info"""
    return jsonify({
        'status': 'running',
        'service': 'Sign Language Translator API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'model_details': {
            'type': 'CNN (Convolutional Neural Network)',
            'accuracy': '98.93%',
            'input_size': '28x28 grayscale',
            'total_classes': len(label_map),
            'excluded_letters': ['J', 'Z']  # J and Z require motion
        },
        'endpoints': {
            '/api/sign/': 'GET - API info',
            '/api/sign/predict': 'POST - Predict sign language letter',
            '/api/sign/letters': 'GET - List available letters',
            '/api/sign/health': 'GET - Health check',
            '/api/sign/info': 'GET - Model details'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/api/sign/predict',
            'body': {
                'image': 'base64_encoded_image_data'
            }
        },
        'tips': [
            'Use plain, light-colored background',
            'Ensure good lighting',
            'Center hand in frame',
            'Keep hand still for 1-2 seconds',
            'Show clear hand gesture'
        ]
    })


@sign_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'service': 'sign_language_translator',
        'model_loaded': model is not None,
        'label_map_loaded': len(label_map) > 0,
        'classes': len(label_map)
    })


@sign_bp.route('/letters', methods=['GET'])
def get_letters():
    """Get all available letters"""
    if not label_map:
        return jsonify({
            'error': 'Label map not loaded',
            'status': 'service_unavailable'
        }), 503
    
    letters = [label_map[i] for i in sorted(label_map.keys())]
    
    return jsonify({
        'success': True,
        'letters': letters,
        'total': len(letters),
        'excluded': ['J', 'Z'],
        'reason': 'J and Z require motion (not static gestures)',
        'alphabet': 'American Sign Language (ASL)'
    })


@sign_bp.route('/predict', methods=['POST'])
def predict():
    """Predict sign language letter from image"""
    
    # Check if model is loaded
    if model is None or not label_map:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'status': 'service_unavailable'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'Please provide image in JSON body: {"image": "base64_data"}',
                'example': {
                    'image': 'data:image/jpeg;base64,/9j/4AAQSkZJRg...'
                }
            }), 400
        
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({
                'error': 'Image data is empty',
                'tip': 'Make sure to send base64 encoded image'
            }), 400
        
        # Preprocess image
        try:
            processed_img = preprocess_image(image_data)
        except Exception as e:
            return jsonify({
                'error': 'Failed to process image',
                'details': str(e),
                'tip': 'Ensure image is valid and properly base64 encoded'
            }), 400
        
        # Predict
        predictions = model.predict(processed_img, verbose=0)[0]
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        
        results = []
        for idx in top_5_indices:
            if idx in label_map:
                confidence = float(predictions[idx] * 100)
                results.append({
                    'letter': label_map[idx],
                    'confidence': round(confidence, 2)
                })
        
        # Get top prediction
        top_prediction = results[0] if results else None
        
        # Debug logging
        if top_prediction:
            print(f"✅ Predicted: {top_prediction['letter']} ({top_prediction['confidence']:.1f}%)")
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': top_prediction,
            'confidence_level': get_confidence_level(top_prediction['confidence']) if top_prediction else 'Unknown',
            'metadata': {
                'total_predictions': len(results),
                'model_accuracy': '98.93%'
            }
        })
        
    except Exception as e:
        print(f"❌ Sign prediction error: {str(e)}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e),
            'status': 'internal_error'
        }), 500


@sign_bp.route('/info')
def info():
    """Get detailed model information"""
    return jsonify({
        'model': {
            'name': 'ASL Sign Language Recognizer',
            'algorithm': 'CNN (Convolutional Neural Network)',
            'framework': 'TensorFlow/Keras',
            'accuracy': '98.93%',
            'training_samples': 27455,
            'test_accuracy': '98.93%',
            'parameters': '~162K'
        },
        'architecture': {
            'input_shape': '28x28x1 (grayscale)',
            'layers': [
                'Conv2D + MaxPooling',
                'Conv2D + MaxPooling',
                'Dropout',
                'Flatten',
                'Dense (128)',
                'Output (24 classes)'
            ],
            'activation': 'ReLU (hidden), Softmax (output)'
        },
        'preprocessing': {
            'steps': [
                'Convert to grayscale',
                'Resize to 28x28',
                'Normalize to [0, 1]',
                'Optional hand segmentation'
            ]
        },
        'dataset': {
            'source': 'Sign Language MNIST',
            'format': 'Static hand gestures',
            'classes': 24,
            'excluded': ['J', 'Z'],
            'alphabet': 'American Sign Language (ASL)'
        },
        'usage_tips': [
            'Use plain background (preferably light colored)',
            'Ensure good, even lighting',
            'Center hand in camera frame',
            'Make clear, distinct gestures',
            'Wait 1-2 seconds for stable prediction',
            'Avoid shadows and cluttered backgrounds'
        ]
    })


@sign_bp.route('/examples')
def examples():
    """Get example letters and their descriptions"""
    examples = {
        'A': 'Fist with thumb on the side',
        'B': 'Flat hand with fingers together, thumb across palm',
        'C': 'Curved hand forming C shape',
        'D': 'Index finger up, other fingers touching thumb',
        'E': 'Fingers curled, thumb across fingertips',
        'F': 'Index and middle finger up, thumb touches middle finger',
        'G': 'Index finger and thumb pointing horizontally',
        'H': 'Index and middle finger pointing horizontally',
        'I': 'Pinky finger up, other fingers down',
        'K': 'Index and middle finger up in V, thumb touches middle',
        'L': 'L-shape with thumb and index finger',
        'M': 'Thumb under three fingers',
        'N': 'Thumb under two fingers',
        'O': 'Fingers and thumb form O shape',
        'P': 'Like K but pointing down',
        'Q': 'Like G but pointing down',
        'R': 'Index and middle finger crossed',
        'S': 'Fist with thumb over fingers',
        'T': 'Thumb between index and middle',
        'U': 'Index and middle finger up together',
        'V': 'Index and middle finger up in V shape',
        'W': 'Index, middle, and ring finger up',
        'X': 'Index finger crooked',
        'Y': 'Thumb and pinky extended (shaka)'
    }
    
    return jsonify({
        'examples': examples,
        'total': len(examples),
        'note': 'Letters J and Z require motion and are not supported in static gesture recognition'
    })


def get_confidence_level(confidence):
    """Get human-readable confidence level"""
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    elif confidence >= 40:
        return "Low"
    else:
        return "Very Low"
