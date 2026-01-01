"""
AI Unified API - Main Application
Combines 5 ML APIs: News Detection, Mental Health, Resume Screener, Traffic, Sign Language
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os

# Import all blueprints
from blueprints.news_api import news_bp
from blueprints.mental_api import mental_bp
from blueprints.resume_api import resume_bp
from blueprints.traffic_api import traffic_bp
from blueprints.sign_api import sign_bp

# Create Flask app
app = Flask(__name__)

# Configure CORS (allow all origins for now - customize in production)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# App configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['JSON_SORT_KEYS'] = False

# Register blueprints with URL prefixes
app.register_blueprint(news_bp, url_prefix='/api/news')
app.register_blueprint(mental_bp, url_prefix='/api/mental')
app.register_blueprint(resume_bp, url_prefix='/api/resume')
app.register_blueprint(traffic_bp, url_prefix='/api/traffic')
app.register_blueprint(sign_bp, url_prefix='/api/sign')


@app.route('/')
def home():
    """API Gateway - Main landing page"""
    return jsonify({
        'status': 'running',
        'service': 'AI Unified API Gateway',
        'version': '1.0.0',
        'description': '5 AI-powered APIs in one unified platform',
        'apis': {
            'fake_news_detection': {
                'endpoint': '/api/news',
                'description': 'Detect fake news articles',
                'model': 'Logistic Regression + TF-IDF',
                'accuracy': '~94%'
            },
            'mental_health_chatbot': {
                'endpoint': '/api/mental',
                'description': 'Emotion detection and mental health support',
                'model': 'Random Forest Classifier',
                'accuracy': '73.70%'
            },
            'resume_screener': {
                'endpoint': '/api/resume',
                'description': 'ATS resume analysis and job matching',
                'model': 'TF-IDF + Cosine Similarity',
                'features': 'Skills matching, keyword analysis'
            },
            'traffic_predictor': {
                'endpoint': '/api/traffic',
                'description': 'Multi-junction traffic congestion forecasting',
                'model': 'LSTM Neural Network',
                'accuracy': '94-98%'
            },
            'sign_language_translator': {
                'endpoint': '/api/sign',
                'description': 'ASL sign language recognition',
                'model': 'CNN',
                'accuracy': '98.93%'
            }
        },
        'documentation': {
            'news': '/api/news/',
            'mental': '/api/mental/',
            'resume': '/api/resume/',
            'traffic': '/api/traffic/',
            'sign': '/api/sign/'
        },
        'health_checks': {
            'news': '/api/news/health',
            'mental': '/api/mental/health',
            'resume': '/api/resume/health',
            'traffic': '/api/traffic/health',
            'sign': '/api/sign/health'
        }
    })


@app.route('/health')
def health():
    """Overall system health check"""
    
    # Import here to avoid circular imports
    from blueprints.news_api import model as news_model
    from blueprints.mental_api import model as mental_model
    from blueprints.traffic_api import models as traffic_models
    from blueprints.sign_api import model as sign_model
    
    health_status = {
        'status': 'healthy',
        'apis': {
            'news': news_model is not None,
            'mental': mental_model is not None,
            'resume': True,  # Resume doesn't load models at startup
            'traffic': len(traffic_models) > 0,
            'sign': sign_model is not None
        }
    }
    
    # Determine overall status
    all_healthy = all(health_status['apis'].values())
    health_status['status'] = 'healthy' if all_healthy else 'degraded'
    health_status['apis_loaded'] = sum(health_status['apis'].values())
    health_status['total_apis'] = 5
    
    return jsonify(health_status)


@app.route('/apis')
def list_apis():
    """List all available APIs with endpoints"""
    return jsonify({
        'total': 5,
        'apis': [
            {
                'name': 'Fake News Detection',
                'prefix': '/api/news',
                'endpoints': {
                    'info': 'GET /api/news/',
                    'predict': 'POST /api/news/predict',
                    'health': 'GET /api/news/health'
                }
            },
            {
                'name': 'Mental Health Chatbot',
                'prefix': '/api/mental',
                'endpoints': {
                    'info': 'GET /api/mental/',
                    'detect': 'POST /api/mental/detect',
                    'health': 'GET /api/mental/health',
                    'emotions': 'GET /api/mental/emotions'
                }
            },
            {
                'name': 'Resume Screener',
                'prefix': '/api/resume',
                'endpoints': {
                    'info': 'GET /api/resume/',
                    'analyze': 'POST /api/resume/analyze',
                    'health': 'GET /api/resume/health',
                    'skills': 'GET /api/resume/skills'
                }
            },
            {
                'name': 'Traffic Predictor',
                'prefix': '/api/traffic',
                'endpoints': {
                    'info': 'GET /api/traffic/',
                    'predict': 'POST /api/traffic/predict',
                    'junctions': 'GET /api/traffic/junctions',
                    'history': 'POST /api/traffic/history',
                    'health': 'GET /api/traffic/health'
                }
            },
            {
                'name': 'Sign Language Translator',
                'prefix': '/api/sign',
                'endpoints': {
                    'info': 'GET /api/sign/',
                    'predict': 'POST /api/sign/predict',
                    'letters': 'GET /api/sign/letters',
                    'health': 'GET /api/sign/health'
                }
            }
        ]
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_apis': '/apis',
        'documentation': '/'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'tip': 'Check server logs for details'
    }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'error': 'File too large',
        'message': 'Maximum file size is 16 MB',
        'max_size': '16 MB'
    }), 413


# Run the app
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AI UNIFIED API - STARTING")
    print("="*70)
    print("\n📦 Available APIs:")
    print("  1. 📰 Fake News Detection    → /api/news")
    print("  2. 🧠 Mental Health Chatbot  → /api/mental")
    print("  3. 📄 Resume Screener        → /api/resume")
    print("  4. 🚗 Traffic Predictor      → /api/traffic")
    print("  5. 👋 Sign Language          → /api/sign")
    print("\n🌐 Server running at: http://localhost:5000")
    print("📚 API Docs: http://localhost:5000/")
    print("💚 Health Check: http://localhost:5000/health")
    print("\n" + "="*70 + "\n")
    
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Run app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=False  # Set to False in production
    )
