from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class PredictRequest(BaseModel):
    x: float


# ----------------------------
# 1. Create API app
# ----------------------------
app = FastAPI()

# ----------------------------
# 2. Load trained model
# ----------------------------
model = joblib.load("model.pkl")

# ----------------------------
# 3. Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(request: PredictRequest):
    x=request.x;
    prediction = model.predict([[x]])
    print(f"Received input: {x}, Prediction: {prediction[0]}")
    return {
        "input": x,
        "prediction": prediction[0]
    }