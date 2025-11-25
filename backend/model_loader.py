"""
Model Loader for Respiratory Disease Detection
Loads the trained CNN-Transformer model and provides prediction functionality
"""

import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Reshape, LayerNormalization, 
                                     GlobalAveragePooling1D, Dense,
                                     MultiHeadAttention)
from tensorflow.keras.models import Model

# ==================== Configuration ====================
MODEL_PATH = r"D:\Project Phase 1\MobileApp\backend\best_model.keras"
SYMPTOMS_PATH = r"D:\Project Phase 1\MobileApp\backend\symptoms.json"

# Audio preprocessing parameters (MUST match training)
IMG_SIZE = (128, 128)
N_MFCC = 40
MAX_PAD_LEN = 862
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Class labels (MUST match training order)
CLASSES = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 
           'Healthy', 'LRTI', 'Pneumonia', 'URTI']
NUM_CLASSES = len(CLASSES)

# ==================== Model Architecture ====================
def build_model(input_shape=(128, 128, 1), num_classes=8):
    """Defines the CNN-Transformer architecture (must match training)"""
    inp = Input(shape=input_shape)
    
    # CNN Backbone
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Reshape for Transformer
    shape = tf.keras.backend.int_shape(x)
    seq_len = shape[1] * shape[2]
    features = shape[3]
    x = Reshape((seq_len, features))(x)

    # Transformer Block
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = x + attn
    x = LayerNormalization()(x)
    
    x_ff = Dense(256, activation='relu')(x)
    x = x + x_ff
    x = LayerNormalization()(x)

    # Classification Head
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    return Model(inputs=inp, outputs=out)

# ==================== Global Model Loader ====================
_model = None
_symptoms_data = None

def load_model():
    """Loads the model once and caches it"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        print(f"Loading model from {MODEL_PATH}...")
        try:
            _model = tf.keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                custom_objects={'MultiHeadAttention': MultiHeadAttention}
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    return _model

def load_symptoms():
    """Loads symptoms data once and caches it"""
    global _symptoms_data
    if _symptoms_data is None:
        if os.path.exists(SYMPTOMS_PATH):
            with open(SYMPTOMS_PATH, 'r') as f:
                _symptoms_data = json.load(f)
            print("✅ Symptoms data loaded successfully!")
        else:
            print(f"⚠️ Symptoms file not found at {SYMPTOMS_PATH}, using defaults")
            _symptoms_data = {cls: ["No symptom data available"] for cls in CLASSES}
    return _symptoms_data

# ==================== Audio Preprocessing ====================
def preprocess_audio(audio_path):
    """
    Preprocesses audio file to match training format:
    1. Load audio at 22050 Hz
    2. Extract 40 MFCCs
    3. Pad/truncate to 862 time steps
    4. Resize to 128x128
    5. Normalize using Z-score
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=N_MFCC, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
        
        # Pad or truncate to MAX_PAD_LEN
        current_len = mfccs.shape[1]
        if current_len > MAX_PAD_LEN:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        elif current_len < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - current_len
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Convert to float32 and add channel dimension
        mfccs = mfccs.astype(np.float32)
        mfccs = np.expand_dims(mfccs, axis=-1)  # Shape: (40, 862, 1)
        
        # Convert to tensor for TF operations
        mfccs_tensor = tf.convert_to_tensor(mfccs)
        
        # Resize to IMG_SIZE
        x = tf.image.resize(mfccs_tensor, IMG_SIZE)
        x = tf.cast(x, tf.float32)
        
        # Z-score normalization (matches training)
        mean = tf.reduce_mean(x)
        std = tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-8)
        
        # Add batch dimension
        x = tf.expand_dims(x, axis=0)
        
        return x.numpy()
    
    except Exception as e:
        print(f"❌ Error preprocessing audio: {e}")
        raise

# ==================== Prediction Function ====================
def predict(audio_path):
    """
    Main prediction function called by Flask API
    
    Args:
        audio_path (str): Path to the .wav file
    
    Returns:
        dict: {
            'label': str,           # Predicted disease name
            'confidence': float,    # Confidence score (0-1)
            'all_probabilities': dict,  # All class probabilities
            'symptoms': list        # Associated symptoms
        }
    """
    try:
        # Load model and symptoms
        model = load_model()
        symptoms_data = load_symptoms()
        
        # Preprocess audio
        input_data = preprocess_audio(audio_path)
        
        # Run inference
        predictions = model.predict(input_data, verbose=0)
        
        # Get predicted class
        predicted_idx = np.argmax(predictions[0])
        predicted_label = CLASSES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        # Get all probabilities
        all_probs = {
            CLASSES[i]: float(predictions[0][i]) 
            for i in range(NUM_CLASSES)
        }
        
        # Get symptoms for predicted disease
        symptoms = symptoms_data.get(predicted_label, ["No symptoms available"])
        
        return {
            'label': predicted_label,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'symptoms': symptoms
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            'error': str(e)
        }

# ==================== Module Test ====================
if __name__ == "__main__":
    # Test the model loader
    print("Testing model loader...")
    model = load_model()
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Classes: {CLASSES}")
    
    symptoms = load_symptoms()
    print(f"Loaded symptoms for {len(symptoms)} diseases")