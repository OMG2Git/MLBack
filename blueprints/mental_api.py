"""
Mental Health Emotion Detection API Blueprint
Detects emotions from text and provides supportive responses
Model: Random Forest Classifier | Accuracy: 73.70%
"""

from flask import Blueprint, request, jsonify
import pickle
import re
import string
import os
import random

# Create Blueprint
mental_bp = Blueprint('mental', __name__)

# Model paths (relative to project root)
MODEL_PATH = os.path.join('models', 'emotion_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load the trained model and vectorizer
print("🧠 Loading Mental Health emotion detection model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✅ Mental Health emotion model loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading emotion model: {e}")
    print("⚠️  Make sure 'emotion_model.pkl' and 'tfidf_vectorizer.pkl' are in 'models/' folder")
    model = None
    vectorizer = None
except Exception as e:
    print(f"❌ Unexpected error loading emotion model: {e}")
    model = None
    vectorizer = None


# Emotion responses and resources
EMOTION_RESPONSES = {
    'anger': [
        "I can sense you're feeling angry. That's a valid emotion. Would you like to talk about what's causing this anger?",
        "It's okay to feel angry. Try taking a few deep breaths. What's triggering these feelings?",
        "Anger can be intense. Remember, it's okay to feel this way. Want to share what's bothering you?"
    ],
    'fear': [
        "I hear that you're feeling fearful or anxious. That must be difficult. Can you tell me more about what's worrying you?",
        "Fear is a natural emotion. You're safe here. What's making you feel this way?",
        "It sounds like you're experiencing anxiety or fear. Let's talk through this together. What's on your mind?"
    ],
    'joy': [
        "I'm so glad you're feeling happy! That's wonderful. What's bringing you joy today?",
        "It's beautiful to hear positivity from you! What made you smile?",
        "Your joy is contagious! Tell me more about what's making you feel good."
    ],
    'love': [
        "It sounds like you're experiencing love or gratitude. That's a beautiful feeling! Want to share more?",
        "Love and connection are so important. I'm happy you're feeling this way. Tell me more!",
        "What a lovely emotion to experience! Would you like to talk about it?"
    ],
    'sadness': [
        "I'm sorry you're feeling sad. It's okay to feel this way. I'm here to listen. What's making you feel down?",
        "Sadness can be heavy. Thank you for sharing with me. Want to talk about what's bothering you?",
        "I hear your sadness. You're not alone in this. Would you like to tell me more about how you're feeling?"
    ],
    'surprise': [
        "It seems something unexpected happened! How are you processing this?",
        "Surprise can bring mixed emotions. Want to share what happened?",
        "That sounds unexpected! How are you feeling about this?"
    ]
}


RESOURCES = {
    'anger': {
        'tips': [
            "Try the 4-7-8 breathing technique: Breathe in for 4, hold for 7, exhale for 8",
            "Physical exercise can help release anger - try a quick walk",
            "Write down what's making you angry without filtering"
        ]
    },
    'fear': {
        'tips': [
            "Ground yourself: Name 5 things you can see, 4 you can touch, 3 you can hear",
            "Remember: Anxiety lies. This feeling will pass",
            "Try progressive muscle relaxation - tense and release each muscle group"
        ],
        'helplines': [
            "🇮🇳 AASRA: 91-22-27546669",
            "🇮🇳 Vandrevala Foundation: 1860-2662-345",
            "🇮🇳 iCall: 022-25521111"
        ]
    },
    'sadness': {
        'tips': [
            "Be gentle with yourself - it's okay to have difficult days",
            "Reach out to someone you trust, even if it's just to say hi",
            "Small steps count: Getting out of bed is an achievement"
        ],
        'helplines': [
            "🇮🇳 AASRA: 91-22-27546669",
            "🇮🇳 Vandrevala Foundation: 1860-2662-345",
            "🇮🇳 iCall: 022-25521111",
            "🌍 International: Befrienders.org"
        ]
    },
    'joy': {
        'tips': [
            "Savor this moment! Write down what made you happy",
            "Share your joy with someone - happiness multiplies when shared",
            "Remember this feeling for difficult times ahead"
        ]
    },
    'love': {
        'tips': [
            "Express gratitude to those you care about",
            "Take time to appreciate the connections in your life",
            "Love includes self-love - be kind to yourself too"
        ]
    },
    'surprise': {
        'tips': [
            "Take a moment to process what happened",
            "It's okay to need time to adjust to unexpected news",
            "Talk to someone about your feelings"
        ]
    }
}


def preprocess_text(text):
    """Clean and preprocess text (same as training)"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


@mental_bp.route('/')
def home():
    """Home route - API status"""
    return jsonify({
        'status': 'running',
        'service': 'Mental Health Emotion Detection API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'model_details': {
            'type': 'Random Forest Classifier',
            'accuracy': '73.70%',
            'emotions': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
        },
        'endpoints': {
            '/api/mental/': 'GET - API info',
            '/api/mental/detect': 'POST - Detect emotion from text',
            '/api/mental/health': 'GET - Health check',
            '/api/mental/emotions': 'GET - List supported emotions'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/api/mental/detect',
            'body': {
                'message': 'Your message here...'
            }
        }
    })


@mental_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'service': 'mental_health_detection',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


@mental_bp.route('/emotions')
def emotions():
    """Get list of supported emotions"""
    return jsonify({
        'emotions': list(EMOTION_RESPONSES.keys()),
        'count': len(EMOTION_RESPONSES),
        'details': {
            emotion: {
                'has_tips': 'tips' in RESOURCES.get(emotion, {}),
                'has_helplines': 'helplines' in RESOURCES.get(emotion, {}),
                'response_variants': len(responses)
            }
            for emotion, responses in EMOTION_RESPONSES.items()
        }
    })


@mental_bp.route('/detect', methods=['POST'])
def detect_emotion():
    """Detect emotion from user message"""
    
    # Check if model is loaded
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'status': 'service_unavailable'
        }), 503
    
    try:
        # Get message from request
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Please provide message in JSON body: {"message": "your text..."}',
                'example': {
                    'message': 'I am feeling really happy today!'
                }
            }), 400
        
        message = data['message']
        
        # Validate message length
        if len(message.strip()) < 3:
            return jsonify({
                'error': 'Message too short. Please provide at least 3 characters.',
                'received_length': len(message.strip())
            }), 400
        
        # Preprocess the text
        cleaned_text = preprocess_text(message)
        
        # Check if cleaned text is empty
        if len(cleaned_text.strip()) == 0:
            return jsonify({
                'error': 'Message contains no meaningful text after preprocessing.',
                'tip': 'Please use actual words to express your feelings.'
            }), 400
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict emotion
        emotion = model.predict(text_vector)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]
        emotion_classes = model.classes_
        
        # Create confidence dict
        confidence = {}
        for em, prob in zip(emotion_classes, probabilities):
            confidence[em] = round(float(prob) * 100, 2)
        
        # Get top confidence
        top_confidence = round(max(probabilities) * 100, 2)
        
        # Get response for detected emotion
        response_text = random.choice(EMOTION_RESPONSES.get(emotion, ["I'm here to listen. Tell me more."]))
        
        # Get resources
        resources = RESOURCES.get(emotion, {})
        
        # Determine if helpline should be shown (high confidence negative emotions)
        show_helpline = emotion in ['fear', 'sadness', 'anger'] and top_confidence > 60
        
        # Prepare result
        result = {
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'top_confidence': top_confidence,
            'response': response_text,
            'resources': resources if resources else None,
            'show_helpline': show_helpline,
            'metadata': {
                'message_length': len(message),
                'cleaned_length': len(cleaned_text),
                'confidence_level': get_confidence_level(top_confidence)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Emotion detection error: {str(e)}")
        return jsonify({
            'error': 'An error occurred during emotion detection',
            'details': str(e),
            'status': 'internal_error'
        }), 500


def get_confidence_level(confidence):
    """Get human-readable confidence level"""
    if confidence >= 80:
        return "Very High"
    elif confidence >= 60:
        return "High"
    elif confidence >= 40:
        return "Moderate"
    else:
        return "Low"


# Optional: Crisis detection endpoint
@mental_bp.route('/crisis-resources')
def crisis_resources():
    """Get mental health crisis resources"""
    return jsonify({
        'helplines': {
            'india': {
                'AASRA': {
                    'phone': '91-22-27546669',
                    'hours': '24x7',
                    'languages': ['English', 'Hindi']
                },
                'Vandrevala Foundation': {
                    'phone': '1860-2662-345',
                    'hours': '24x7',
                    'languages': ['English', 'Hindi', 'Multiple']
                },
                'iCall': {
                    'phone': '022-25521111',
                    'hours': 'Mon-Sat, 8 AM - 10 PM',
                    'email': 'icall@tiss.edu'
                }
            },
            'international': {
                'Befrienders Worldwide': {
                    'website': 'https://www.befrienders.org',
                    'description': 'Global network of crisis centers'
                }
            }
        },
        'emergency': {
            'india': '112 (National Emergency Number)',
            'message': 'If you are in immediate danger, please call emergency services'
        },
        'tips': [
            "You are not alone - help is available",
            "Talking to someone can make a difference",
            "It's okay to ask for help",
            "Crisis feelings are temporary"
        ]
    })


# Optional: Conversation history tracking
@mental_bp.route('/info')
def info():
    """Get model information"""
    return jsonify({
        'model': {
            'name': 'Mental Health Emotion Detector',
            'algorithm': 'Random Forest Classifier',
            'accuracy': '73.70%',
            'training_data': 'Emotion-labeled text dataset',
            'features': 'TF-IDF Vectorization'
        },
        'emotions': {
            'positive': ['joy', 'love', 'surprise'],
            'negative': ['anger', 'fear', 'sadness'],
            'total': 6
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
        'support': {
            'provides_helplines': True,
            'provides_tips': True,
            'languages': ['English']
        }
    })
