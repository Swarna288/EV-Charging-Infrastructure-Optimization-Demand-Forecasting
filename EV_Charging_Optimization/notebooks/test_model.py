import joblib

model = joblib.load("artifacts/final_ev_charging_ridge_model.pkl")

print("Model loaded successfully!")
print(model)