from flask import Blueprint, request, jsonify
import pickle
import re
import string
import os

mental_bp = Blueprint('mental', __name__)

# Global variables
model = None
vectorizer = None
emotions_list = []

def preprocess_text(text):
    """
    EXACT preprocessing from your training code
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def load_models():
    """Load emotion detection model and vectorizer"""
    global model, vectorizer, emotions_list
    
    try:
        print("🧠 Loading Mental Health emotion detection model...")
        
        # Load model
        model_path = os.path.join('models', 'emotion_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer (mental health uses separate vectorizer)
        vectorizer_path = os.path.join('models', 'mental_health_vectorizer.pkl')
        
        # If mental health vectorizer doesn't exist, try the general one
        if not os.path.exists(vectorizer_path):
            print("⚠️ mental_health_vectorizer.pkl not found, using tfidf_vectorizer.pkl")
            vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Get emotion classes from model
        if hasattr(model, 'classes_'):
            emotions_list = model.classes_.tolist()
        
        print("✅ Mental Health emotion model loaded successfully!")
        print(f"✅ Emotions: {emotions_list}")
        print(f"✅ Vectorizer features: {len(vectorizer.vocabulary_)}")
        
    except Exception as e:
        print(f"❌ Error loading emotion model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        vectorizer = None

# Load models when blueprint is registered
load_models()

@mental_bp.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'service': 'Mental Health Emotion Detection API',
        'model': 'Random Forest Classifier (200 trees, balanced)',
        'algorithm': 'TF-IDF + Random Forest',
        'accuracy': '73.70%',
        'emotions_supported': emotions_list if emotions_list else [
            'joy', 'sadness', 'anger', 'fear', 'love', 'surprise'
        ],
        'features': {
            'max_features': 5000,
            'ngrams': '1-3 (unigrams, bigrams, trigrams)',
            'balanced': 'Yes - undersampled to equal distribution'
        },
        'endpoints': {
            'predict': {
                'url': '/predict',
                'method': 'POST',
                'description': 'Detect emotion from text',
                'body': {
                    'text': 'Your message here'
                },
                'example': {
                    'text': 'I am so happy and excited about this!'
                },
                'response': {
                    'emotion': 'joy',
                    'confidence': 85.6,
                    'all_emotions': {
                        'joy': 85.6,
                        'sadness': 5.2,
                        'anger': 3.1
                    }
                }
            },
            'health': {
                'url': '/health',
                'method': 'GET',
                'description': 'Check API health status'
            },
            'resources': {
                'url': '/resources',
                'method': 'GET',
                'description': 'Get mental health support resources'
            }
        },
        'training_info': {
            'dataset': 'Emotion detection dataset (balanced)',
            'samples_per_emotion': 'Equal distribution',
            'preprocessing': 'Lowercase, remove URLs, remove punctuation',
            'vectorization': 'TF-IDF (5000 features, trigrams)'
        }
    })

@mental_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'service': 'mental_health_emotion_detection',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'emotions_available': len(emotions_list) if emotions_list else 0,
        'emotions': emotions_list if emotions_list else []
    })

@mental_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from text
    
    Request body:
    {
        "text": "I am feeling great today!"
    }
    
    Response:
    {
        "emotion": "joy",
        "confidence": 85.6,
        "all_emotions": {...},
        "text_preview": "I am feeling great..."
    }
    """
    try:
        # Check if models are loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the service.',
                'model_status': model is not None,
                'vectorizer_status': vectorizer is not None
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data received',
                'usage': 'Send POST request with {"text": "your message"}'
            }), 400
        
        # Get text
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Missing or empty "text" field',
                'example': {'text': 'I am feeling happy today'}
            }), 400
        
        # Validate text length
        if len(text) < 3:
            return jsonify({
                'error': 'Text too short. Please provide at least 3 characters.'
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                'error': 'Text too long. Maximum 5000 characters allowed.'
            }), 400
        
        # Preprocess text (MUST match training!)
        cleaned_text = preprocess_text(text)
        
        if not cleaned_text:
            return jsonify({
                'error': 'Text became empty after preprocessing. Please use meaningful text.'
            }), 400
        
        # Transform using vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict emotion
        emotion = model.predict(text_vectorized)[0]
        
        # Get probability scores for all emotions
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities))
        
        # Build emotion probability dict
        emotions_proba = {}
        if hasattr(model, 'classes_'):
            for idx, emotion_class in enumerate(model.classes_):
                emotions_proba[emotion_class] = round(float(probabilities[idx]) * 100, 2)
        
        # Sort emotions by probability (descending)
        sorted_emotions = dict(sorted(emotions_proba.items(), key=lambda x: x[1], reverse=True))
        
        # Get top 3 emotions
        top_emotions = dict(list(sorted_emotions.items())[:3])
        
        # Build response
        result = {
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'all_emotions': sorted_emotions,
            'top_3_emotions': top_emotions,
            'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
            'analysis': {
                'primary_emotion': emotion,
                'secondary_emotions': list(top_emotions.keys())[1:3] if len(top_emotions) > 1 else [],
                'emotion_diversity': len([e for e in emotions_proba.values() if e > 5.0])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'type': type(e).__name__
        }), 500

@mental_bp.route('/resources', methods=['GET'])
def resources():
    """
    Mental health support resources and helplines
    """
    return jsonify({
        'message': 'If you or someone you know needs help, please reach out to these resources:',
        'disclaimer': 'This API is for informational purposes only and not a substitute for professional mental health care.',
        'helplines': {
            'india': {
                'vandrevala_foundation': {
                    'number': '1860-2662-345 / 1800-2333-330',
                    'available': '24/7',
                    'type': 'Mental health support'
                },
                'aasra': {
                    'number': '91-22-27546669',
                    'available': '24/7',
                    'type': 'Suicide prevention'
                },
                'sumaitri': {
                    'number': '011-23389090',
                    'available': '10 AM - 10 PM IST',
                    'type': 'Crisis helpline'
                },
                'nimhans': {
                    'number': '080-46110007',
                    'available': 'Mon-Sat, 9 AM - 5 PM',
                    'type': 'Mental health institute'
                },
                'icall': {
                    'number': '9152987821',
                    'email': 'icall@tiss.edu',
                    'available': 'Mon-Sat, 10 AM - 8 PM',
                    'type': 'Psychosocial helpline'
                }
            },
            'international': {
                'usa': {
                    'suicide_prevention': '988',
                    'crisis_text_line': 'Text HOME to 741741',
                    'samhsa': '1-800-662-4357'
                },
                'uk': {
                    'samaritans': '116 123',
                    'shout': 'Text SHOUT to 85258'
                },
                'australia': {
                    'lifeline': '13 11 14',
                    'beyond_blue': '1300 22 4636'
                }
            }
        },
        'online_resources': {
            'therapy_platforms': [
                'BetterHelp.com',
                'Talkspace.com',
                'YourDOST.com (India)'
            ],
            'meditation_apps': [
                'Headspace',
                'Calm',
                'Insight Timer'
            ],
            'educational': [
                'NIMH - nimh.nih.gov',
                'Mental Health Foundation - mentalhealth.org.uk',
                'Mind - mind.org.uk'
            ]
        },
        'self_care_tips': [
            'Practice mindfulness and meditation',
            'Maintain regular sleep schedule',
            'Exercise regularly',
            'Connect with friends and family',
            'Limit social media and news consumption',
            'Practice gratitude journaling',
            'Seek professional help when needed'
        ],
        'emergency': {
            'note': 'If you are in immediate danger or crisis:',
            'actions': [
                'Call emergency services (911 in US, 112 in India)',
                'Go to nearest emergency room',
                'Contact crisis helpline immediately'
            ]
        }
    })

@mental_bp.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    return jsonify({
        'emotions': emotions_list if emotions_list else [],
        'total': len(emotions_list) if emotions_list else 0,
        'model_type': 'Random Forest Classifier',
        'balanced': True
    })

@mental_bp.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple texts at once
    
    Request:
    {
        "texts": ["text1", "text2", "text3"]
    }
    """
    try:
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" array'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': '"texts" must be an array'}), 400
        
        if len(texts) == 0:
            return jsonify({'error': 'Empty texts array'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts allowed per request'}), 400
        
        results = []
        for i, text in enumerate(texts):
            try:
                cleaned = preprocess_text(str(text))
                if not cleaned:
                    results.append({'index': i, 'error': 'Empty after preprocessing'})
                    continue
                
                vectorized = vectorizer.transform([cleaned])
                emotion = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                confidence = float(max(proba))
                
                results.append({
                    'index': i,
                    'emotion': emotion,
                    'confidence': round(confidence * 100, 2),
                    'text_preview': text[:50] + ('...' if len(text) > 50 else '')
                })
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
        
        return jsonify({
            'total': len(texts),
            'successful': len([r for r in results if 'emotion' in r]),
            'failed': len([r for r in results if 'error' in r]),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
