from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from model_loader import predict, load_symptoms  # Import symptoms loader

app = Flask(__name__)
CORS(app)

# Load symptoms once at startup
SYMPTOMS_DATA = load_symptoms()

# ✅ Default route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Smart Respire Backend is running!"})

# ✅ Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        file = request.files['file']

        # Validate file type
        if not file.filename.endswith('.wav'):
            return jsonify({"error": "Only .wav files are supported"}), 400

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        print(f"📁 Received file: {file.filename}")
        print(f"💾 Saved to temp: {tmp_path}")

        # Call ML model to get result
        result = predict(tmp_path)

        # Remove temp file
        os.remove(tmp_path)

        if "error" in result:
            print(f"❌ Prediction error: {result['error']}")
            return jsonify({"error": result["error"]}), 500

        # Get the predicted label
        predicted_label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0)
        
        # Get symptoms from loaded JSON
        symptoms = SYMPTOMS_DATA.get(predicted_label, ["No symptoms information available"])

        # Format response for frontend
        response = {
            "disease": predicted_label,
            "confidence": round(confidence * 100, 2),  # Convert to percentage
            "symptoms": symptoms,
            "all_probabilities": result.get("all_probabilities", {})
        }

        print(f"✅ Prediction: {predicted_label} ({response['confidence']}%)")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Backend Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("🚀 SmartRespire Backend Starting...")
    print("="*50)
    print("📦 Loading model and symptoms data...")
    
    # Preload model at startup (optional but recommended)
    try:
        from model_loader import load_model
        load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not preload model: {e}")
    
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)