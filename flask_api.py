"""
Flask API Backend for Neural Network Threat Detection System
Serves predictions from trained LSTM-3, GRU-3, and GRU-2 models
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables for models and artifacts
models = {}
tokenizer = None
label_encoder = None
config = None


def load_artifacts():
    """Load all trained models and preprocessing artifacts"""
    global models, tokenizer, label_encoder, config
    
    print("="*80)
    print("LOADING NEURAL NETWORK THREAT DETECTION SYSTEM")
    print("="*80)
    
    try:
        # Load models - using best_ prefix
        print("\n[1] Loading trained models...")
        models['LSTM-3'] = load_model('best_lstm3_model.h5')
        print("  ✓ LSTM-3 loaded (Accuracy: 97.96%)")
        
        models['GRU-3'] = load_model('best_gru3_model.h5')
        print("  ✓ GRU-3 loaded (Accuracy: 97.95%)")
        
        models['GRU-2'] = load_model('best_gru2_model.h5')
        print("  ✓ GRU-2 loaded (Accuracy: 97.60%)")
        
        # Load tokenizer
        print("\n[2] Loading tokenizer...")
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("  ✓ Tokenizer loaded")
        
        # Load label encoder
        print("\n[3] Loading label encoder...")
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"  ✓ Label encoder loaded (Classes: {', '.join(label_encoder.classes_)})")
        
        # Load config
        print("\n[4] Loading configuration...")
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        print(f"  ✓ Configuration loaded")
        
        # Load results
        print("\n[5] Loading training results...")
        with open('training_results.json', 'r') as f:
            results = json.load(f)
        print("  ✓ Training results loaded")
        
        print("\n" + "="*80)
        print("✅ ALL MODELS LOADED SUCCESSFULLY - API READY")
        print("="*80)
        print("\nModel Performance:")
        for model_result in results['models']:
            print(f"  {model_result['model']:10s} - Accuracy: {model_result['accuracy']*100:.2f}%")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR LOADING MODELS: {e}")
        print("\nPlease ensure these files exist:")
        print("  - best_lstm3_model.h5")
        print("  - best_gru3_model.h5")
        print("  - best_gru2_model.h5")
        print("  - tokenizer.pkl")
        print("  - label_encoder.pkl")
        print("  - model_config.json")
        print("  - training_results.json")
        return False


def predict_single_model(url, model_name):
    """Predict using a single model"""
    # Preprocess URL
    sequence = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequence, maxlen=config['max_len'], padding='post')
    
    # Predict
    model = models[model_name]
    prediction_proba = model.predict(padded, verbose=0)[0]
    prediction_class = np.argmax(prediction_proba)
    threat_type = label_encoder.classes_[prediction_class]
    confidence = float(prediction_proba[prediction_class] * 100)
    
    # Get accuracy for this model
    with open('training_results.json', 'r') as f:
        results = json.load(f)
    model_accuracy = next(r['accuracy'] * 100 for r in results['models'] if r['model'] == model_name)
    
    return {
        'model': model_name,
        'prediction': threat_type,
        'confidence': round(confidence, 2),
        'accuracy': round(model_accuracy, 2),
        'all_probabilities': {
            label_encoder.classes_[i]: round(float(prediction_proba[i] * 100), 2)
            for i in range(len(label_encoder.classes_))
        }
    }


def predict_ensemble(url):
    """Predict using ensemble of all three models"""
    predictions = []
    
    # Get predictions from all models
    for model_name in ['LSTM-3', 'GRU-3', 'GRU-2']:
        pred = predict_single_model(url, model_name)
        predictions.append(pred)
    
    # Calculate ensemble prediction (weighted average)
    all_probs = []
    for pred in predictions:
        probs = [pred['all_probabilities'][cls] for cls in label_encoder.classes_]
        all_probs.append(probs)
    
    avg_probs = np.mean(all_probs, axis=0)
    final_class = label_encoder.classes_[np.argmax(avg_probs)]
    final_confidence = float(np.max(avg_probs))
    
    return {
        'url': url,
        'finalPrediction': final_class,
        'avgConfidence': round(final_confidence, 2),
        'models': predictions,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


# API Routes

@app.route('/')
def index():
    """Serve the main HTML page"""
    # Check if index.html is in static folder or root
    if os.path.exists('static/index.html'):
        return send_from_directory('static', 'index.html')
    elif os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    else:
        return """
        <html>
        <body>
            <h1>Neural Threat Detection System</h1>
            <p>API is running! But index.html not found.</p>
            <p>Please create a 'static' folder and put index.html inside it.</p>
            <p>Or place index.html in the root directory.</p>
            <h3>API Endpoints:</h3>
            <ul>
                <li><a href="/api/health">/api/health</a> - Health check</li>
                <li>/api/predict - POST URL for prediction</li>
                <li><a href="/api/models">/api/models</a> - Model information</li>
            </ul>
        </body>
        </html>
        """


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) == 3,
        'available_models': list(models.keys()),
        'classes': label_encoder.classes_.tolist() if label_encoder else []
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Get ensemble prediction
        result = predict_ensemble(url)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/model/<model_name>', methods=['POST'])
def predict_specific_model(model_name):
    """Predict using a specific model"""
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        result = predict_single_model(url, model_name)
        result['url'] = url
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models"""
    try:
        with open('training_results.json', 'r') as f:
            results = json.load(f)
        
        return jsonify({
            'models': results['models'],
            'timestamp': results.get('timestamp', '')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """Get list of available visualization images"""
    viz_dir = 'visualizations'
    if os.path.exists(viz_dir):
        files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        return jsonify({'visualizations': sorted(files)})
    return jsonify({'visualizations': []})


@app.route('/api/visualizations/<filename>', methods=['GET'])
def get_visualization(filename):
    """Serve a specific visualization image"""
    try:
        return send_from_directory('visualizations', filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple URLs"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls or not isinstance(urls, list):
            return jsonify({'error': 'URLs array is required'}), 400
        
        results = []
        for url in urls[:100]:  # Limit to 100 URLs
            if url.strip():
                result = predict_ensemble(url.strip())
                results.append(result)
        
        return jsonify({
            'total': len(results),
            'results': results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        with open('training_results.json', 'r') as f:
            results = json.load(f)
        
        # Calculate average metrics
        models_data = results['models']
        avg_accuracy = np.mean([m['accuracy'] for m in models_data]) * 100
        avg_precision = np.mean([m['precision'] for m in models_data]) * 100
        avg_recall = np.mean([m['recall'] for m in models_data]) * 100
        avg_f1 = np.mean([m['f1_score'] for m in models_data]) * 100
        
        return jsonify({
            'models': len(models),
            'classes': label_encoder.classes_.tolist(),
            'average_metrics': {
                'accuracy': round(avg_accuracy, 2),
                'precision': round(avg_precision, 2),
                'recall': round(avg_recall, 2),
                'f1_score': round(avg_f1, 2)
            },
            'individual_models': models_data,
            'training_date': results.get('timestamp', '')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load models before starting server
    if load_artifacts():
        print("\n🚀 Starting Flask API Server...")
        print("📍 Access the web interface at: http://localhost:5000")
        print("📍 API Health Check: http://localhost:5000/api/health")
        print("\nPress CTRL+C to stop the server\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please check the error messages above.")