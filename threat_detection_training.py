"""
NEURAL NETWORK THREAT DETECTION SYSTEM
Complete Training & Deployment Pipeline
Top 3 Models: LSTM-3, GRU-3, GRU-2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
import warnings
import pickle
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


class ThreatDetectionSystem:
    """Complete Neural Network-based Threat Detection System"""
    
    def __init__(self, data_file='malicious_phish.csv'):
        self.data_file = data_file
        self.models = {}
        self.histories = {}
        self.results = {}
        self.tokenizer = None
        self.label_encoder = None
        self.max_words = 10000
        self.max_len = 100
        self.num_classes = None
        
        # Model configurations
        self.model_configs = {
            'LSTM-3': {
                'embedding_dim': 128,
                'units': [128, 64],
                'dropout': 0.4,
                'bidirectional': True
            },
            'GRU-3': {
                'embedding_dim': 112,
                'units': [112, 56],
                'dropout': 0.35,
                'bidirectional': True
            },
            'GRU-2': {
                'embedding_dim': 80,
                'units': [80, 40],
                'dropout': 0.3,
                'bidirectional': False
            }
        }
        
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Configure GPU settings"""
        print("="*100)
        print("GPU CONFIGURATION")
        print("="*100)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU DETECTED: {len(gpus)} GPU(s) available")
                set_global_policy('mixed_float16')
                print(f"✅ Mixed Precision (FP16) Enabled")
            except RuntimeError as e:
                print(f"⚠️  GPU Error: {e}")
        else:
            print("⚠️  Using CPU (slower training)")
        print()
        
    def check_trained_models(self):
        """Check if models are already trained"""
        required_files = [
            'final_lstm3_model.h5',
            'final_gru3_model.h5',
            'final_gru2_model.h5',
            'tokenizer.pkl',
            'label_encoder.pkl',
            'model_config.json',
            'training_results.json'
        ]
        
        all_exist = all(os.path.exists(f) for f in required_files)
        
        if all_exist:
            print("="*100)
            print("✅ TRAINED MODELS FOUND - LOADING FROM DISK")
            print("="*100)
            return True
        else:
            print("="*100)
            print("🔄 NO TRAINED MODELS FOUND - STARTING TRAINING PROCESS")
            print("="*100)
            return False
    
    def load_trained_models(self):
        """Load pre-trained models and artifacts"""
        print("\n[Loading Models & Artifacts...]")
        
        try:
            # Load models
            self.models['LSTM-3'] = load_model('final_lstm3_model.h5')
            self.models['GRU-3'] = load_model('final_gru3_model.h5')
            self.models['GRU-2'] = load_model('final_gru2_model.h5')
            print("✓ Models loaded")
            
            # Load tokenizer
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("✓ Tokenizer loaded")
            
            # Load label encoder
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Label encoder loaded")
            
            # Load config
            with open('model_config.json', 'r') as f:
                config = json.load(f)
                self.max_words = config['max_words']
                self.max_len = config['max_len']
                self.num_classes = config['num_classes']
            print("✓ Configuration loaded")
            
            # Load results
            with open('training_results.json', 'r') as f:
                self.results = json.load(f)
            print("✓ Training results loaded")
            
            print("\n✅ ALL ARTIFACTS LOADED SUCCESSFULLY!\n")
            return True
            
        except Exception as e:
            print(f"\n❌ Error loading models: {e}")
            print("Will proceed with training...\n")
            return False
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("\n" + "="*100)
        print("DATA LOADING & PREPROCESSING")
        print("="*100)
        
        # Load data
        print(f"\n[1] Loading dataset: {self.data_file}")
        df = pd.read_csv(self.data_file)
        
        print(f"\n✓ Dataset loaded")
        print(f"  Total samples: {df.shape[0]:,}")
        print(f"  Features: {df.shape[1]}")
        
        print(f"\n📊 Class Distribution:")
        class_dist = df['type'].value_counts()
        for threat_type, count in class_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {threat_type:12s}: {count:6,} ({percentage:5.2f}%)")
        
        # Prepare data
        X = df['url'].values
        y = df['type'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"\n[2] Label Encoding")
        print(f"  Classes: {', '.join(self.label_encoder.classes_)}")
        
        # Tokenization
        print(f"\n[3] Tokenizing URLs (character-level)")
        self.tokenizer = Tokenizer(num_words=self.max_words, char_level=True)
        self.tokenizer.fit_on_texts(X)
        X_sequences = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(X_sequences, maxlen=self.max_len, padding='post')
        
        print(f"  Vocabulary size: {min(len(self.tokenizer.word_index), self.max_words)}")
        print(f"  Sequence length: {self.max_len}")
        
        # Convert to categorical
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n[4] Data Split")
        print(f"  Training: {X_train.shape[0]:,} samples")
        print(f"  Testing:  {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test, df
    
    def build_model(self, model_name):
        """Build a specific model architecture"""
        config = self.model_configs[model_name]
        
        model = Sequential(name=model_name.replace('-', '_'))
        
        # Embedding layer
        model.add(Embedding(
            input_dim=self.max_words,
            output_dim=config['embedding_dim'],
            input_length=self.max_len
        ))
        
        # First recurrent layer
        if model_name.startswith('LSTM'):
            if config['bidirectional']:
                model.add(Bidirectional(LSTM(config['units'][0], return_sequences=True)))
            else:
                model.add(LSTM(config['units'][0], return_sequences=True))
        else:  # GRU
            if config['bidirectional']:
                model.add(Bidirectional(GRU(config['units'][0], return_sequences=True)))
            else:
                model.add(GRU(config['units'][0], return_sequences=True))
        
        model.add(Dropout(config['dropout']))
        
        # Second recurrent layer
        if model_name.startswith('LSTM'):
            if config['bidirectional']:
                model.add(Bidirectional(LSTM(config['units'][1])))
            else:
                model.add(LSTM(config['units'][1]))
        else:  # GRU
            if config['bidirectional']:
                model.add(Bidirectional(GRU(config['units'][1])))
            else:
                model.add(GRU(config['units'][1]))
        
        model.add(Dropout(config['dropout']))
        
        # Dense layers
        model.add(Dense(config['units'][1], activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, X_train, y_train):
        """Train all three models"""
        print("\n" + "="*100)
        print("MODEL TRAINING")
        print("="*100)
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        ]
        
        for idx, model_name in enumerate(['LSTM-3', 'GRU-3', 'GRU-2'], 1):
            print(f"\n{'='*100}")
            print(f"[{idx}/3] TRAINING {model_name}")
            print('='*100)
            
            config = self.model_configs[model_name]
            print(f"\nArchitecture:")
            print(f"  Type: {'Bidirectional' if config['bidirectional'] else 'Standard'}")
            print(f"  Units: {config['units'][0]} → {config['units'][1]}")
            print(f"  Dropout: {config['dropout']}")
            print(f"  Embedding: {config['embedding_dim']}")
            
            # Build model
            model = self.build_model(model_name)
            
            # Add checkpoint
            checkpoint = ModelCheckpoint(
                f"best_{model_name.lower().replace('-', '')}_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
            
            # Train
            print(f"\nTraining started...")
            history = model.fit(
                X_train, y_train,
                epochs=10,  # REDUCED FROM 40 TO 10
                batch_size=256,  # INCREASED for faster training
                validation_split=0.2,
                callbacks=callbacks + [checkpoint],
                verbose=1
            )
            
            self.models[model_name] = model
            self.histories[model_name] = history
            
            print(f"\n✓ {model_name} training complete!")
            print(f"  Best Val Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*100)
        print("MODEL EVALUATION")
        print("="*100)
        
        results = []
        
        for model_name, model in self.models.items():
            print(f"\n{'='*100}")
            print(f"EVALUATING: {model_name}")
            print('='*100)
            
            # Predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            # Metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'model': model_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
            
            print(f"\n📊 Performance Metrics:")
            print(f"  Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall:    {recall*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")
            
            print(f"\n📋 Classification Report:")
            print(classification_report(y_true, y_pred, 
                                       target_names=self.label_encoder.classes_, 
                                       zero_division=0))
        
        self.results = {'models': results, 'timestamp': datetime.now().isoformat()}
        return results
    
    def generate_visualizations(self, X_test, y_test):
        """Generate all visualization plots"""
        print("\n" + "="*100)
        print("GENERATING VISUALIZATIONS")
        print("="*100)
        
        # 1. Training History
        self._plot_training_history()
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(X_test, y_test)
        
        # 3. ROC Curves
        self._plot_roc_curves(X_test, y_test)
        
        # 4. Precision-Recall Curves
        self._plot_precision_recall(X_test, y_test)
        
        # 5. Performance Comparison
        self._plot_performance_comparison()
        
        # 6. Detailed Analysis per Model
        self._plot_detailed_analysis(X_test, y_test)
        
        print("\n✅ All visualizations generated!")
    
    def _plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        for idx, (model_name, history) in enumerate(self.histories.items()):
            # Accuracy
            ax = axes[0, idx]
            ax.plot(history.history['accuracy'], label='Train', linewidth=2.5, marker='o')
            ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2.5, marker='s')
            ax.set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Loss
            ax = axes[1, idx]
            ax.plot(history.history['loss'], label='Train', linewidth=2.5, marker='o')
            ax.plot(history.history['val_loss'], label='Validation', linewidth=2.5, marker='s')
            ax.set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training History - Top 3 Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/1_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 1_training_history.png")
    
    def _plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 2_confusion_matrices.png")
    
    def _plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred_proba = model.predict(X_test, verbose=0)
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                axes[idx].plot(fpr, tpr, lw=2.5, color=colors[i],
                             label=f'{self.label_encoder.classes_[i]} (AUC={roc_auc:.3f})')
            
            axes[idx].plot([0, 1], [0, 1], 'k--', lw=2)
            axes[idx].set_xlim([0.0, 1.0])
            axes[idx].set_ylim([0.0, 1.05])
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].legend(loc="lower right", fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('ROC Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/3_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 3_roc_curves.png")
    
    def _plot_precision_recall(self, X_test, y_test):
        """Plot precision-recall curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ['#9B59B6', '#3498DB', '#E74C3C', '#2ECC71']
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred_proba = model.predict(X_test, verbose=0)
            
            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
                axes[idx].plot(recall, precision, lw=2.5, color=colors[i],
                             label=f'{self.label_encoder.classes_[i]}')
            
            axes[idx].set_xlabel('Recall')
            axes[idx].set_ylabel('Precision')
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].legend(loc="lower left", fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/4_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 4_precision_recall.png")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df_results = pd.DataFrame(self.results['models'])
        model_names = df_results['model'].values
        
        # Metrics comparison
        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.2
        ax.bar(x - 1.5*width, df_results['accuracy']*100, width, label='Accuracy', alpha=0.9)
        ax.bar(x - 0.5*width, df_results['precision']*100, width, label='Precision', alpha=0.9)
        ax.bar(x + 0.5*width, df_results['recall']*100, width, label='Recall', alpha=0.9)
        ax.bar(x + 1.5*width, df_results['f1_score']*100, width, label='F1-Score', alpha=0.9)
        ax.set_xlabel('Models')
        ax.set_ylabel('Score (%)')
        ax.set_title('Performance Metrics', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Accuracy ranking
        ax = axes[0, 1]
        colors = ['#4ECDC4', '#45B7D1', '#95E1D3']
        ax.barh(range(3), df_results['accuracy']*100, color=colors, alpha=0.9, edgecolor='black', linewidth=2)
        ax.set_yticks(range(3))
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Accuracy Ranking', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for i, acc in enumerate(df_results['accuracy']*100):
            ax.text(acc + 0.1, i, f'{acc:.2f}%', va='center', fontweight='bold')
        
        # F1-Score
        ax = axes[1, 0]
        ax.plot(range(3), df_results['f1_score']*100, 'o-', linewidth=3, markersize=12, color='#4ECDC4')
        ax.set_xlabel('Model')
        ax.set_ylabel('F1-Score (%)')
        ax.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels(model_names)
        ax.grid(True, alpha=0.3)
        
        # Heatmap
        ax = axes[1, 1]
        metrics_matrix = df_results[['accuracy', 'precision', 'recall', 'f1_score']].values.T * 100
        sns.heatmap(metrics_matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax,
                   xticklabels=model_names,
                   yticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                   cbar_kws={'label': 'Score (%)'})
        ax.set_title('Performance Heatmap', fontsize=13, fontweight='bold')
        
        plt.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/5_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 5_performance_comparison.png")
    
    def _plot_detailed_analysis(self, X_test, y_test):
        """Plot detailed analysis for each model"""
        for model_name, model in self.models.items():
            fig = plt.figure(figsize=(16, 12))
            
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            # Confusion Matrix
            plt.subplot(2, 3, 1)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix', fontweight='bold')
            plt.ylabel('True')
            plt.xlabel('Predicted')
            
            # ROC Curve
            plt.subplot(2, 3, 2)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2.5, color=colors[i],
                        label=f'{self.label_encoder.classes_[i]} (AUC={roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves', fontweight='bold')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            # Class-wise Performance
            plt.subplot(2, 3, 3)
            class_acc = []
            for i in range(self.num_classes):
                mask = y_true == i
                if np.sum(mask) > 0:
                    class_acc.append(accuracy_score(y_true[mask], y_pred[mask]) * 100)
                else:
                    class_acc.append(0)
            plt.bar(self.label_encoder.classes_, class_acc, color=colors[:self.num_classes], alpha=0.8)
            plt.ylabel('Accuracy (%)')
            plt.title('Class-wise Accuracy', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Prediction Confidence
            plt.subplot(2, 3, 4)
            max_probs = np.max(y_pred_proba, axis=1)
            plt.hist(max_probs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(np.mean(max_probs), color='r', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(max_probs):.3f}')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # Training History
            history = self.histories[model_name]
            plt.subplot(2, 3, 5)
            plt.plot(history.history['accuracy'], label='Train', linewidth=2)
            plt.plot(history.history['val_accuracy'], label='Val', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training History', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Metrics Summary
            plt.subplot(2, 3, 6)
            plt.axis('off')
            result = [r for r in self.results['models'] if r['model'] == model_name][0]
            metrics_text = f"""
            PERFORMANCE SUMMARY
            {'='*35}
            
            Accuracy:  {result['accuracy']*100:.2f}%
            Precision: {result['precision']*100:.2f}%
            Recall:    {result['recall']*100:.2f}%
            F1-Score:  {result['f1_score']*100:.2f}%
            
            {'='*35}
            Total Samples: {len(y_test):,}
            Correct: {np.sum(y_true == y_pred):,}
            Errors: {np.sum(y_true != y_pred):,}
            """
            plt.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'Detailed Analysis - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'visualizations/6_detailed_{model_name.lower().replace("-", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: 6_detailed_{model_name.lower().replace('-', '_')}.png")
    
    def save_artifacts(self):
        """Save all models and artifacts"""
        print("\n" + "="*100)
        print("SAVING MODELS & ARTIFACTS")
        print("="*100)
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"final_{model_name.lower().replace('-', '')}_model.h5"
            model.save(filename)
            print(f"✓ Saved: {filename}")
        
        # Save tokenizer
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print("✓ Saved: tokenizer.pkl")
        
        # Save label encoder
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✓ Saved: label_encoder.pkl")
        
        # Save config
        config = {
            'max_words': self.max_words,
            'max_len': self.max_len,
            'num_classes': self.num_classes,
            'classes': self.label_encoder.classes_.tolist()
        }
        with open('model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✓ Saved: model_config.json")
        
        # Save results
        with open('training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("✓ Saved: training_results.json")
        
        # Save results CSV
        df_results = pd.DataFrame(self.results['models'])
        df_results.to_csv('model_results.csv', index=False)
        print("✓ Saved: model_results.csv")
    
    def predict_url(self, url, model_name='LSTM-3'):
        """Predict threat type for a URL"""
        # Preprocess
        sequence = self.tokenizer.texts_to_sequences([url])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        
        # Predict
        model = self.models[model_name]
        prediction_proba = model.predict(padded, verbose=0)[0]
        prediction_class = np.argmax(prediction_proba)
        threat_type = self.label_encoder.classes_[prediction_class]
        confidence = float(prediction_proba[prediction_class] * 100)
        
        return {
            'url': url,
            'threat_type': threat_type,
            'confidence': confidence,
            'model': model_name,
            'all_probabilities': {
                self.label_encoder.classes_[i]: float(prediction_proba[i] * 100)
                for i in range(self.num_classes)
            }
        }
    
    def predict_url_ensemble(self, url):
        """Predict using all three models (ensemble)"""
        predictions = []
        
        for model_name in ['LSTM-3', 'GRU-3', 'GRU-2']:
            pred = self.predict_url(url, model_name)
            predictions.append(pred)
        
        # Calculate ensemble prediction
        all_probs = np.array([
            [p['all_probabilities'][cls] for cls in self.label_encoder.classes_]
            for p in predictions
        ])
        avg_probs = np.mean(all_probs, axis=0)
        final_class = self.label_encoder.classes_[np.argmax(avg_probs)]
        final_confidence = float(np.max(avg_probs))
        
        return {
            'url': url,
            'final_prediction': final_class,
            'final_confidence': final_confidence,
            'individual_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        print("\n" + "="*100)
        print("NEURAL NETWORK THREAT DETECTION SYSTEM")
        print("Complete Training & Deployment Pipeline")
        print("="*100)
        
        # Check if models already trained
        if self.check_trained_models():
            if self.load_trained_models():
                print("\n✅ Using pre-trained models. Ready for predictions!")
                return True
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, df = self.load_and_preprocess_data()
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Generate visualizations
        self.generate_visualizations(X_test, y_test)
        
        # Save everything
        self.save_artifacts()
        
        # Test predictions
        self._test_predictions(df)
        
        print("\n" + "="*100)
        print("✅ PIPELINE COMPLETE!")
        print("="*100)
        print("\nSummary:")
        for result in self.results['models']:
            print(f"  {result['model']:10s} - Accuracy: {result['accuracy']*100:.2f}%")
        
        return True
    
    def _test_predictions(self, df):
        """Test predictions on sample URLs"""
        print("\n" + "="*100)
        print("TESTING PREDICTIONS")
        print("="*100)
        
        test_urls = [
            'br-icloud.com.br',
            'espn.go.com/nba/player/_/id/3457/brandon-rush',
            'youtube.com',
            'facebook.com'
        ]
        
        print("\nSample URL Predictions:\n")
        for url in test_urls:
            print(f"URL: {url}")
            result = self.predict_url_ensemble(url)
            print(f"  Final Prediction: {result['final_prediction']} ({result['final_confidence']:.2f}%)")
            for pred in result['individual_predictions']:
                print(f"    {pred['model']:10s} → {pred['threat_type']:12s} ({pred['confidence']:.2f}%)")
            print()


# Main execution
if __name__ == "__main__":
    # Initialize system
    system = ThreatDetectionSystem(data_file='malicious_phish.csv')
    
    # Run complete pipeline
    system.run_complete_pipeline()
    
    print("\n" + "="*100)
    print("SYSTEM READY FOR DEPLOYMENT")
    print("="*100)
    print("\nGenerated files:")
    print("  Models: final_lstm3_model.h5, final_gru3_model.h5, final_gru2_model.h5")
    print("  Artifacts: tokenizer.pkl, label_encoder.pkl, model_config.json")
    print("  Results: training_results.json, model_results.csv")
    print("  Visualizations: visualizations/*.png")
    print("\nTo use in web app, run: python flask_api.py")
    print("="*100)
    