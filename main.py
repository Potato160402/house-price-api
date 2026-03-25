from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib

# ── Model definition (same as training) ──────────────────
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(36, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

# ── Load model & scalers ──────────────────────────────────
model = NeuralNetwork()
model.load_state_dict(torch.load("best_house_model.pth",
                                  map_location=torch.device("cpu")))
model.eval()

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ── Input schema (36 numeric features) ───────────────────
class HouseFeatures(BaseModel):
    MSSubClass: float = 60
    LotFrontage: float = 65
    LotArea: float = 8450
    OverallQual: float = 7
    OverallCond: float = 5
    YearBuilt: float = 2003
    YearRemodAdd: float = 2003
    MasVnrArea: float = 196
    BsmtFinSF1: float = 706
    BsmtFinSF2: float = 0
    BsmtUnfSF: float = 150
    TotalBsmtSF: float = 856
    FirstFlrSF: float = 856
    SecondFlrSF: float = 854
    LowQualFinSF: float = 0
    GrLivArea: float = 1710
    BsmtFullBath: float = 1
    BsmtHalfBath: float = 0
    FullBath: float = 2
    HalfBath: float = 1
    BedroomAbvGr: float = 3
    KitchenAbvGr: float = 1
    TotRmsAbvGrd: float = 8
    Fireplaces: float = 0
    GarageYrBlt: float = 2003
    GarageCars: float = 2
    GarageArea: float = 548
    WoodDeckSF: float = 0
    OpenPorchSF: float = 61
    EnclosedPorch: float = 0
    ThreeSsnPorch: float = 0
    ScreenPorch: float = 0
    PoolArea: float = 0
    MiscVal: float = 0
    MoSold: float = 2
    YrSold: float = 2008

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="House Price Predictor",
    description="Predicts house sale prices using a PyTorch neural network trained on the Ames Housing dataset.",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "House Price Predictor API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model": "PyTorch Neural Network",
        "dataset": "House Prices - Advanced Regression Techniques (Kaggle)",
        "input_features": 36,
        "output": "Predicted SalePrice in USD",
        "rmse": "$27,657"
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to numpy array (same order as training)
    input_data = np.array([[
        features.MSSubClass, features.LotFrontage, features.LotArea,
        features.OverallQual, features.OverallCond, features.YearBuilt,
        features.YearRemodAdd, features.MasVnrArea, features.BsmtFinSF1,
        features.BsmtFinSF2, features.BsmtUnfSF, features.TotalBsmtSF,
        features.FirstFlrSF, features.SecondFlrSF, features.LowQualFinSF,
        features.GrLivArea, features.BsmtFullBath, features.BsmtHalfBath,
        features.FullBath, features.HalfBath, features.BedroomAbvGr,
        features.KitchenAbvGr, features.TotRmsAbvGrd, features.Fireplaces,
        features.GarageYrBlt, features.GarageCars, features.GarageArea,
        features.WoodDeckSF, features.OpenPorchSF, features.EnclosedPorch,
        features.ThreeSsnPorch, features.ScreenPorch, features.PoolArea,
        features.MiscVal, features.MoSold, features.YrSold
    ]])

    # Normalize → predict → denormalize
    input_scaled  = scaler_X.transform(input_data)
    input_tensor  = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()

    predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]

    return {
        "predicted_price_usd": round(float(predicted_price), 2),
        "message": f"Estimated house price: ${predicted_price:,.0f}"
    }