# SmartRespire



## 🌟 Overview

SmartRespire is an AI-powered web application designed to assist in the early detection and prediction of respiratory diseases based on user symptoms. Leveraging a trained machine learning model, it provides quick, accessible, and reliable health insights.

---

## 🚀 Features
- **AI Model**: Utilizes a Keras-based model for disease prediction
- **User-Friendly Interface**: Simple HTML frontend for symptom input
- **Fast Inference**: Real-time predictions via a lightweight Flask backend
- **Customizable**: Easily extendable for more diseases or symptoms

---

## 🗂️ Project Structure
```
SmartRespire/
│   test.html                # Frontend UI
└───backend/
    │   app.py               # Flask API server
    │   best_model.keras     # Trained ML model
    │   model_loader.py      # Model loading logic
    │   symptoms.json        # Symptom definitions
    │   test.py              # Backend tests
    └───__pycache__/         # Python cache
```

---

## ⚡ Quickstart

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SmartRespire.git
   cd SmartRespire
   ```
2. **Install dependencies**
   ```bash
   pip install flask keras
   ```
3. **Run the backend**
   ```bash
   cd backend
   python app.py
   ```
4. **Open the frontend**
   - Open `test.html` in your browser.

---

## 🧠 Model
- Trained using Keras
- Model file: `backend/best_model.keras`
- Input: Symptoms from user
- Output: Predicted respiratory disease

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements
- Inspired by the need for accessible healthcare tools
- Built with Python, Flask, and Keras

---

<p align="center">
  <em>Empowering early detection, one breath at a time.</em>
</p>
