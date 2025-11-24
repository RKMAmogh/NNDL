"""
Neural Logic Engine - Flask Web Application
Pure Neural Network for Web Threat Detection
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import sys
import pickle
import traceback

# Add scripts to path
sys.path.append('scripts')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and extractors
model = None
feature_extractor = None
scaler = None

# Load model and components at startup
def load_components():
    """Load neural network model and feature extractor"""
    global model, feature_extractor, scaler
    
    print("Loading components...")
    
    # Load feature extractor
    try:
        from feature_extractor import FeatureExtractor
        feature_extractor = FeatureExtractor()
        print("✅ Feature extractor loaded")
    except Exception as e:
        print(f"❌ Feature extractor error: {e}")
        feature_extractor = None
    
    # Load neural network model
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        model_path = 'nle_model/best_nle_model.keras'
        if os.path.exists(model_path):
            try:
                # Try loading with safe_mode=False and compile=False
                import keras as keras_lib
                model = keras_lib.saving.load_model(model_path, compile=False, safe_mode=False)
            except:
                # Fallback: try standard TensorFlow loading
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                except:
                    # Last resort: rebuild model from config
                    print("⚠️ Standard loading failed, rebuilding model...")
                    from scripts.nle_model import build_nle_model
                    
                    # Load weights separately
                    model = build_nle_model(input_dim=14, num_classes=4)
                    try:
                        model.load_weights(model_path)
                    except:
                        print("❌ Could not load weights")
                        model = None
            
            if model:
                # Recompile the model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print(f"✅ Neural network loaded from {model_path}")
                print(f"✅ Model has {model.count_params():,} parameters")
            else:
                print(f"❌ Failed to load model")
        else:
            print(f"❌ Model not found at {model_path}")
            model = None
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        model = None
    
    # Load scaler
    try:
        scaler_path = 'features/scaler_params.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Feature scaler loaded")
        else:
            print("⚠️ Scaler not found, will use unscaled features")
            scaler = None
    except Exception as e:
        print(f"⚠️ Scaler loading error: {e}")
        scaler = None
    
    return model is not None and feature_extractor is not None

# Load components on startup
components_loaded = load_components()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         model_loaded=model is not None,
                         extractor_loaded=feature_extractor is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on URL (fetches HTML) or provided HTML"""
    try:
        # Check if components are loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Neural network model not loaded. Please train the model first.'
            })
        
        if feature_extractor is None:
            return jsonify({
                'success': False,
                'error': 'Feature extractor not available.'
            })
        
        # Get URL (primary input)
        url = request.form.get('url', '').strip()
        html_content = None
        
        # If URL is provided, fetch HTML from it
        if url and url != 'http://example.com':
            try:
                import requests
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                html_content = response.text
                print(f"✅ Fetched HTML from URL: {url}")
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to fetch URL: {str(e)}'
                })
        
        # If no URL or fetching failed, check for file upload (optional)
        if not html_content and 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                html_content = file.read().decode('utf-8', errors='ignore')
                print(f"✅ HTML loaded from file: {file.filename}")
        
        # If still no content, check for pasted HTML (optional)
        if not html_content:
            html_content = request.form.get('html_content', '')
            if html_content:
                print("✅ HTML loaded from paste")
        
        if not html_content:
            return jsonify({
                'success': False,
                'error': 'No URL provided or unable to fetch content'
            })
        
        # Extract features
        try:
            features = feature_extractor.extract_features(html_content, url)
            features = features.reshape(1, -1)
            
            # Scale if scaler available
            if scaler is not None:
                features = scaler.transform(features)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Feature extraction failed: {str(e)}'
            })
        
        # Make prediction
        try:
            prediction = model.predict(features, verbose=0)
            
            # Get class and confidence
            class_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_idx]) * 100
            
            # Map to label
            label_map = {0: 'Benign', 1: 'Defacement', 2: 'Malware', 3: 'Phishing'}
            threat_class = label_map.get(class_idx, f'Class {class_idx}')
            
            # Get all probabilities
            probabilities = {
                label_map[i]: float(prediction[0][i] * 100)
                for i in range(len(prediction[0]))
            }
            
            # Determine threat level
            if threat_class == 'Benign':
                threat_level = 'safe'
                message = 'No threat detected'
            elif threat_class == 'Phishing':
                threat_level = 'warning'
                message = 'Phishing attempt detected'
            else:
                threat_level = 'danger'
                message = f'{threat_class} detected'
            
            return jsonify({
                'success': True,
                'prediction': {
                    'class': threat_class,
                    'confidence': round(confidence, 2),
                    'threat_level': threat_level,
                    'message': message,
                    'probabilities': probabilities,
                    'features_count': len(features[0])
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/status')
def status():
    """System status endpoint"""
    try:
        # Check data files
        data_status = {
            'training_data': os.path.exists('features/X_train.npy'),
            'test_data': os.path.exists('features/X_test.npy'),
            'labels': os.path.exists('features/Y_labels.npy'),
        }
        
        # Get dataset info if available
        dataset_info = {}
        if data_status['training_data']:
            try:
                X_train = np.load('features/X_train.npy')
                Y_train = np.load('features/Y_train.npy')
                
                dataset_info = {
                    'training_samples': int(X_train.shape[0]),
                    'features': int(X_train.shape[1]),
                    'test_samples': int(np.load('features/X_test.npy').shape[0]) if data_status['test_data'] else 0
                }
                
                # Class distribution
                unique, counts = np.unique(Y_train, return_counts=True)
                label_map = {0: 'Benign', 1: 'Defacement', 2: 'Malware', 3: 'Phishing'}
                dataset_info['class_distribution'] = {
                    label_map.get(int(u), f'Class {u}'): int(c)
                    for u, c in zip(unique, counts)
                }
            except:
                pass
        
        return jsonify({
            'model_loaded': model is not None,
            'extractor_loaded': feature_extractor is not None,
            'scaler_loaded': scaler is not None,
            'data_status': data_status,
            'dataset_info': dataset_info
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred.'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧠 Neural Logic Engine - Flask Application")
    print("="*70)
    print(f"\nModel loaded: {'✅' if model else '❌'}")
    print(f"Feature extractor loaded: {'✅' if feature_extractor else '❌'}")
    print(f"Scaler loaded: {'✅' if scaler else '⚠️ (optional)'}")
    print("\n" + "="*70)
    print("Starting server on http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

