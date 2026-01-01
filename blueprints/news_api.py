"""
Fake News Detection API Blueprint
Detects whether a news article is real or fake using ML
"""

from flask import Blueprint, request, jsonify
import pickle
import re
import string
import os

# Create Blueprint
news_bp = Blueprint('news', __name__)

# Model paths (relative to project root)
MODEL_PATH = os.path.join('models', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load the trained model and vectorizer
print("📰 Loading Fake News Detection model and vectorizer...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✅ Fake News Detection model loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Make sure models are in the 'models/' folder")
    model = None
    vectorizer = None
except Exception as e:
    print(f"❌ Unexpected error loading model: {e}")
    model = None
    vectorizer = None


def preprocess_text(text):
    """Clean and preprocess text data (same as training)"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


@news_bp.route('/')
def home():
    """Home route - API status"""
    return jsonify({
        'status': 'running',
        'service': 'Fake News Detection API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'endpoints': {
            '/api/news/': 'GET - API info',
            '/api/news/predict': 'POST - Detect fake news',
            '/api/news/health': 'GET - Health check'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/api/news/predict',
            'body': {
                'text': 'Your news article text here...'
            }
        }
    })


@news_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'service': 'fake_news_detection',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


@news_bp.route('/predict', methods=['POST'])
def predict():
    """Predict if news article is fake or real"""
    
    # Check if model is loaded
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'status': 'service_unavailable'
        }), 503
    
    try:
        # Get the article text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text in JSON body: {"text": "your article..."}',
                'example': {
                    'text': 'Breaking news: Scientists discover new planet...'
                }
            }), 400
        
        article_text = data['text']
        
        # Check if text is too short
        if len(article_text.strip()) < 10:
            return jsonify({
                'error': 'Article text too short. Please provide at least 10 characters.',
                'received_length': len(article_text.strip())
            }), 400
        
        # Preprocess the text
        cleaned_text = preprocess_text(article_text)
        
        # Check if cleaned text is empty
        if len(cleaned_text.strip()) == 0:
            return jsonify({
                'error': 'Article contains no meaningful text after preprocessing.',
                'tip': 'Make sure your text contains actual words and sentences.'
            }), 400
        
        # Vectorize the text
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        # Prepare response
        result = {
            'success': True,
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'label': int(prediction),
            'confidence': {
                'fake': round(float(prediction_proba[0]) * 100, 2),
                'real': round(float(prediction_proba[1]) * 100, 2)
            },
            'metadata': {
                'text_length': len(article_text),
                'cleaned_text_length': len(cleaned_text),
                'cleaned_text_preview': cleaned_text[:100] + '...' if len(cleaned_text) > 100 else cleaned_text
            },
            'verdict': get_verdict(prediction_proba[prediction])
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e),
            'status': 'internal_error'
        }), 500


def get_verdict(confidence):
    """Get human-readable verdict based on confidence"""
    confidence_percent = confidence * 100
    
    if confidence_percent >= 90:
        return "Very Confident"
    elif confidence_percent >= 75:
        return "Confident"
    elif confidence_percent >= 60:
        return "Moderately Confident"
    else:
        return "Low Confidence"


# Optional: Additional utility endpoint
@news_bp.route('/info')
def info():
    """Get model information"""
    return jsonify({
        'model': {
            'type': 'Logistic Regression',
            'features': 'TF-IDF Vectorization',
            'training_data': 'News articles dataset',
            'accuracy': '~94%'
        },
        'preprocessing': {
            'steps': [
                'Convert to lowercase',
                'Remove URLs',
                'Remove mentions and hashtags',
                'Remove punctuation',
                'Remove extra whitespace'
            ]
        },
        'classes': {
            0: 'Fake News',
            1: 'Real News'
        }
    })
