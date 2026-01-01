from flask import Blueprint, request, jsonify
import pickle
import re
import string
import os

news_bp = Blueprint('news', __name__)

# Global variables for model
model = None
vectorizer = None

def preprocess_text(text):
    """
    EXACT preprocessing from training code
    Must match training exactly!
    """
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Reuters prefix (specific to ISOT dataset)
    text = re.sub(r'\(reuters\)\s*-\s*', '', text)
    text = re.sub(r'^[A-Z\s]+\(reuters\)', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#','', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_models():
    """Load model and vectorizer"""
    global model, vectorizer
    
    try:
        print("📰 Loading Fake News Detection model and vectorizer...")
        
        # Load model
        model_path = os.path.join('models', 'fake_news_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print(f"✅ Fake News Detection model loaded successfully!")
        print(f"✅ Vectorizer features: {len(vectorizer.vocabulary_)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        vectorizer = None

# Load models when blueprint is registered
load_models()

@news_bp.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'service': 'Fake News Detection API',
        'model': 'Logistic Regression + TF-IDF',
        'accuracy': '~94%',
        'endpoints': {
            'predict': {
                'url': '/predict',
                'method': 'POST',
                'body': {
                    'text': 'News article text to analyze'
                },
                'example': {
                    'text': 'Breaking: Scientists discover aliens on Mars!'
                }
            },
            'health': {
                'url': '/health',
                'method': 'GET'
            }
        }
    })

@news_bp.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'fake_news_detection',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

@news_bp.route('/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    try:
        # Check if models are loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the service.'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        # Get text
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Missing "text" field or empty text'}), 400
        
        # Preprocess text (MUST match training!)
        cleaned_text = preprocess_text(text)
        
        # Transform using vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get confidence
        confidence = float(max(probabilities))
        
        # Result
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'fake': round(float(probabilities[0]) * 100, 2),
                'real': round(float(probabilities[1]) * 100, 2)
            },
            'text_preview': text[:100] + ('...' if len(text) > 100 else '')
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500
