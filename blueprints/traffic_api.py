"""
Traffic Predictor API Blueprint
LSTM-powered traffic congestion forecasting for 4 junctions
Model: LSTM Neural Network | Time-series prediction
"""

from flask import Blueprint, request, jsonify
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import pandas as pd

# Create Blueprint
traffic_bp = Blueprint('traffic', __name__)

# Global storage for models and scalers
models = {}
scalers = {}
traffic_data = None

# Model paths
MODELS_DIR = 'models'
DATA_PATH = os.path.join('data', 'traffic.csv')

# Model accuracy stats (from training)
MODEL_STATS = {
    1: {'mae': 3.12, 'accuracy': 96.88},
    2: {'mae': 5.62, 'accuracy': 94.38},
    3: {'mae': 1.81, 'accuracy': 98.19},
    4: {'mae': 5.91, 'accuracy': 94.09}
}


def load_all_models():
    """Load all trained LSTM models and scalers"""
    global traffic_data
    
    print("🚗 Loading Traffic Prediction models...")
    
    # Load traffic data
    try:
        if os.path.exists(DATA_PATH):
            traffic_data = pd.read_csv(DATA_PATH)
            traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])
            print(f"✅ Traffic data loaded: {len(traffic_data)} records")
        else:
            print(f"⚠️  Traffic data not found at: {DATA_PATH}")
            traffic_data = None
    except Exception as e:
        print(f"❌ Error loading traffic data: {e}")
        traffic_data = None
    
    # Load models and scalers for each junction
    for junction in [1, 2, 3, 4]:
        try:
            # Import tensorflow here to avoid unnecessary loading
            from tensorflow.keras.models import load_model
            
            model_path = os.path.join(MODELS_DIR, f'traffic_model_junction_{junction}.h5')
            scaler_path = os.path.join(MODELS_DIR, f'scaler_junction_{junction}.pkl')
            
            models[junction] = load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                scalers[junction] = pickle.load(f)
            
            print(f"✅ Junction {junction} model loaded (MAE: {MODEL_STATS[junction]['mae']}%)")
            
        except FileNotFoundError:
            print(f"⚠️  Model files not found for Junction {junction}")
        except Exception as e:
            print(f"❌ Error loading Junction {junction} model: {e}")


# Load models on module import
load_all_models()


@traffic_bp.route('/')
def home():
    """Traffic API info"""
    return jsonify({
        'status': 'running',
        'service': 'Traffic Prediction API',
        'version': '1.0.0',
        'model': 'LSTM Neural Network',
        'models_loaded': len(models),
        'junctions': list(models.keys()),
        'features': ['Multi-hour forecasting', 'Congestion level analysis', 'Historical data access'],
        'endpoints': {
            '/api/traffic/': 'GET - API info',
            '/api/traffic/predict': 'POST - Predict traffic',
            '/api/traffic/junctions': 'GET - List junctions',
            '/api/traffic/history': 'POST - Get historical data',
            '/api/traffic/health': 'GET - Health check',
            '/api/traffic/stats': 'GET - Model statistics'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/api/traffic/predict',
            'body': {
                'junction': 1,
                'hours_ahead': 6
            }
        }
    })


@traffic_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if len(models) > 0 else 'degraded',
        'service': 'traffic_predictor',
        'models_loaded': len(models),
        'available_junctions': list(models.keys()),
        'data_loaded': traffic_data is not None,
        'data_records': len(traffic_data) if traffic_data is not None else 0
    })


@traffic_bp.route('/junctions', methods=['GET'])
def get_junctions():
    """Get available junctions with statistics"""
    if traffic_data is None:
        return jsonify({'error': 'Traffic data not available'}), 503
    
    junctions_info = []
    
    for junction in [1, 2, 3, 4]:
        try:
            junction_data = traffic_data[traffic_data['Junction'] == junction]
            
            if len(junction_data) > 0:
                junctions_info.append({
                    'id': junction,
                    'name': f'Junction {junction}',
                    'avg_vehicles': int(junction_data['Vehicles'].mean()),
                    'max_vehicles': int(junction_data['Vehicles'].max()),
                    'min_vehicles': int(junction_data['Vehicles'].min()),
                    'records': len(junction_data),
                    'model_loaded': junction in models,
                    'model_accuracy': MODEL_STATS.get(junction, {}).get('accuracy', 0)
                })
        except Exception as e:
            print(f"Error processing junction {junction}: {e}")
    
    return jsonify({
        'junctions': junctions_info,
        'total_junctions': len(junctions_info)
    })


@traffic_bp.route('/stats')
def get_stats():
    """Get model statistics and performance metrics"""
    return jsonify({
        'models': {
            junction: {
                'mae': stats['mae'],
                'accuracy': stats['accuracy'],
                'loaded': junction in models
            }
            for junction, stats in MODEL_STATS.items()
        },
        'overall': {
            'avg_accuracy': sum(s['accuracy'] for s in MODEL_STATS.values()) / len(MODEL_STATS),
            'avg_mae': sum(s['mae'] for s in MODEL_STATS.values()) / len(MODEL_STATS),
            'total_models': len(models)
        },
        'architecture': {
            'type': 'LSTM (Long Short-Term Memory)',
            'input_sequence': 24,
            'prediction_horizon': 'Multi-step (1-12 hours)',
            'features': 'Univariate time-series (vehicle count)'
        }
    })


@traffic_bp.route('/predict', methods=['POST'])
def predict():
    """Predict traffic for next N hours"""
    
    # Check if models are loaded
    if len(models) == 0:
        return jsonify({
            'error': 'No models loaded. Please check server logs.',
            'status': 'service_unavailable'
        }), 503
    
    if traffic_data is None:
        return jsonify({
            'error': 'Traffic data not available',
            'status': 'service_unavailable'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Please provide JSON body with junction and hours_ahead',
                'example': {'junction': 1, 'hours_ahead': 6}
            }), 400
        
        junction = data.get('junction', 1)
        hours_ahead = data.get('hours_ahead', 1)
        
        # Validate inputs
        if junction not in [1, 2, 3, 4]:
            return jsonify({
                'error': f'Invalid junction: {junction}. Must be 1, 2, 3, or 4'
            }), 400
        
        if junction not in models:
            return jsonify({
                'error': f'Model for junction {junction} not available',
                'available_junctions': list(models.keys())
            }), 400
        
        if hours_ahead < 1 or hours_ahead > 12:
            return jsonify({
                'error': 'hours_ahead must be between 1 and 12'
            }), 400
        
        # Get last 24 hours of data for this junction
        junction_data = traffic_data[traffic_data['Junction'] == junction].sort_values('DateTime')
        
        if len(junction_data) < 24:
            return jsonify({
                'error': f'Insufficient data for junction {junction}. Need at least 24 hours.'
            }), 400
        
        last_24_values = junction_data['Vehicles'].tail(24).values.reshape(-1, 1)
        
        # Normalize using the junction's scaler
        scaler = scalers[junction]
        scaled_values = scaler.transform(last_24_values)
        
        # Predict future hours
        predictions = []
        current_sequence = scaled_values.copy()
        
        for _ in range(hours_ahead):
            # Prepare input (last 24 values)
            X = current_sequence[-24:].reshape(1, 24, 1)
            
            # Predict next hour
            pred_scaled = models[junction].predict(X, verbose=0)[0][0]
            predictions.append(pred_scaled)
            
            # Update sequence for next prediction (rolling window)
            current_sequence = np.append(current_sequence, [[pred_scaled]], axis=0)
        
        # Inverse transform predictions to original scale
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_real = scaler.inverse_transform(predictions_array).flatten()
        
        # Get last datetime and create future timestamps
        last_datetime = junction_data['DateTime'].iloc[-1]
        future_times = [last_datetime + timedelta(hours=i+1) for i in range(hours_ahead)]
        
        # Get historical statistics for context
        historical_avg = float(junction_data['Vehicles'].mean())
        historical_max = float(junction_data['Vehicles'].max())
        current_traffic = int(junction_data['Vehicles'].iloc[-1])
        
        # Prepare result with congestion levels
        result = {
            'success': True,
            'junction': junction,
            'predictions': [
                {
                    'datetime': time.strftime('%Y-%m-%d %H:%M'),
                    'vehicles': max(0, int(pred)),  # Ensure non-negative
                    'congestion_level': min(100, max(0, int((pred / historical_max) * 100)))
                }
                for time, pred in zip(future_times, predictions_real)
            ],
            'historical_average': int(historical_avg),
            'current_traffic': current_traffic,
            'metadata': {
                'model_accuracy': MODEL_STATS[junction]['accuracy'],
                'prediction_horizon': f'{hours_ahead} hours',
                'historical_max': int(historical_max),
                'last_updated': last_datetime.strftime('%Y-%m-%d %H:%M')
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Traffic prediction error: {str(e)}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e),
            'status': 'internal_error'
        }), 500


@traffic_bp.route('/history', methods=['POST'])
def get_history():
    """Get historical traffic data for a junction"""
    
    if traffic_data is None:
        return jsonify({
            'error': 'Traffic data not available',
            'status': 'service_unavailable'
        }), 503
    
    try:
        data = request.get_json()
        
        junction = data.get('junction', 1)
        hours = data.get('hours', 48)
        
        # Validate inputs
        if junction not in [1, 2, 3, 4]:
            return jsonify({
                'error': f'Invalid junction: {junction}. Must be 1, 2, 3, or 4'
            }), 400
        
        if hours < 1 or hours > 168:  # Max 1 week
            return jsonify({
                'error': 'hours must be between 1 and 168 (1 week)'
            }), 400
        
        # Get junction data
        junction_data = traffic_data[traffic_data['Junction'] == junction].sort_values('DateTime')
        recent_data = junction_data.tail(hours)
        
        if len(recent_data) == 0:
            return jsonify({
                'error': f'No data available for junction {junction}'
            }), 404
        
        history = [
            {
                'datetime': row['DateTime'].strftime('%Y-%m-%d %H:%M'),
                'vehicles': int(row['Vehicles'])
            }
            for _, row in recent_data.iterrows()
        ]
        
        return jsonify({
            'success': True,
            'junction': junction,
            'history': history,
            'records': len(history),
            'period': f'Last {hours} hours'
        })
        
    except Exception as e:
        print(f"❌ History retrieval error: {str(e)}")
        return jsonify({
            'error': 'An error occurred while retrieving history',
            'details': str(e)
        }), 500


@traffic_bp.route('/info')
def info():
    """Get detailed model information"""
    return jsonify({
        'model': {
            'name': 'Traffic Congestion Predictor',
            'algorithm': 'LSTM (Long Short-Term Memory)',
            'framework': 'TensorFlow/Keras',
            'input_features': 'Vehicle count (univariate time-series)',
            'sequence_length': 24,
            'prediction_steps': '1-12 hours ahead'
        },
        'performance': {
            'junction_1': {'mae': '3.12%', 'accuracy': '96.88%'},
            'junction_2': {'mae': '5.62%', 'accuracy': '94.38%'},
            'junction_3': {'mae': '1.81%', 'accuracy': '98.19%'},
            'junction_4': {'mae': '5.91%', 'accuracy': '94.09%'}
        },
        'preprocessing': {
            'normalization': 'MinMaxScaler (0-1)',
            'window_size': 24,
            'data_frequency': 'Hourly'
        },
        'features': {
            'multi_step_forecasting': True,
            'congestion_level_analysis': True,
            'historical_comparison': True,
            'real_time_prediction': True
        }
    })
