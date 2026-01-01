from flask import Blueprint, request, jsonify
import pickle
import re
import string
import os

news_bp = Blueprint('news', __name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def preprocess_text(text):
    """
    Clean and preprocess text data (EXACT SAME AS TRAINING)
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
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
        
        print("✅ Fake News Detection model loaded successfully!")
        print(f"✅ Vectorizer features: {len(vectorizer.vocabulary_)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        vectorizer = None

# Load models when blueprint is registered
load_models()

@news_bp.route('/', methods=['GET'])
def home():
    """Home route - API status"""
    return jsonify({
        'status': 'running',
        'message': 'Fake News Detection API',
        'model': 'Logistic Regression + TF-IDF',
        'accuracy': '~94%',
        'endpoints': {
            '/predict': 'POST - Detect fake news',
            '/health': 'GET - Health check'
        }
    })

@news_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

@news_bp.route('/predict', methods=['POST'])
def predict():
    """Predict if news article is fake or real"""
    try:
        # Check if models are loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the service.'
            }), 500
        
        # Get the article text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text in JSON body: {"text": "your article..."}'
            }), 400
        
        article_text = data['text']
        
        # Check if text is too short
        if len(article_text.strip()) < 10:
            return jsonify({
                'error': 'Article text too short. Please provide at least 10 characters.'
            }), 400
        
        # Preprocess the text (EXACT SAME AS LOCAL)
        cleaned_text = preprocess_text(article_text)
        
        # Vectorize the text
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        # Prepare response (EXACT SAME AS LOCAL)
        result = {
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'label': int(prediction),
            'confidence': {
                'fake': round(float(prediction_proba[0]) * 100, 2),
                'real': round(float(prediction_proba[1]) * 100, 2)
            },
            'text_length': len(article_text),
            'cleaned_text_preview': cleaned_text[:100] + '...' if len(cleaned_text) > 100 else cleaned_text
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
