
from tensorflow.keras.models import load_model

model = load_model("D:/Project Phase 1/MobileApp/backend/best_model.keras", compile=False)
print(model.output_shape)
