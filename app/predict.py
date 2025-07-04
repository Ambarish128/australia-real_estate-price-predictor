"""
predict.py - Core prediction logic for NSW Property Price Predictor app

This module contains functions to:
- Load saved categorical encoders used during model training
- Prepare and encode user input data into the exact feature format expected by the model
- Predict property sale price (in log scale) using the trained model
- Generate SHAP values for explainability of the prediction

Features include handling of missing or unknown categorical values by defaulting to a fallback,
derivation of seasonal dummy variables from sale month, and output of SHAP explanations
to help understand feature contributions to predictions.

This module is designed to be called by UI or API layers to centralize prediction logic.
"""


import pandas as pd
import pickle
import os
import numpy as np
import shap

# Load saved encoders from models/
def load_encoders():
    with open("models/suburb_encoder.pkl", "rb") as f:
        suburb_le = pickle.load(f)
    with open("models/street_encoder.pkl", "rb") as f:
        street_le = pickle.load(f)
    with open("models/address_encoder.pkl", "rb") as f:
        address_le = pickle.load(f)
    with open("models/description_encoder.pkl", "rb") as f:
        desc_le = pickle.load(f)
    return suburb_le, street_le, address_le, desc_le

# Prepare input features for prediction
def prepare_features(user_input, suburb_le, street_le, address_le, desc_le):
    # Fallbacks for missing values
    postcode = int(user_input.get("Postcode", 2000))
    suburb = user_input.get("Suburb", "UNKNOWN")
    street = user_input.get("Street_Name", "UNKNOWN")
    address = user_input.get("Address", "UNKNOWN")
    description = user_input.get("Description", "UNKNOWN")

    # Encode categorical features
    suburb_encoded = suburb_le.transform([suburb])[0] if suburb in suburb_le.classes_ else 0
    street_encoded = street_le.transform([street])[0] if street in street_le.classes_ else 0
    address_encoded = address_le.transform([address])[0] if address in address_le.classes_ else 0
    description_encoded = desc_le.transform([description])[0] if description in desc_le.classes_ else 0

    # Time features
    year = int(user_input.get("Sale_Year", 2024))
    month = int(user_input.get("Sale_Month", 1))
    weekday = int(user_input.get("Sale_Weekday", 0))

    # Derive season
    if month in [12, 1, 2]:
        season = "Summer"
    elif month in [3, 4, 5]:
        season = "Autumn"
    elif month in [6, 7, 8]:
        season = "Winter"
    else:
        season = "Spring"

    # Final feature vector for prediction (matches training schema)
    features = {
        "Postcode": postcode,
        "Sale_Year": year,
        "Sale_Month": month,
        "Sale_Weekday": weekday,
        "Suburb_encoded": suburb_encoded,
        "Street_Name_encoded": street_encoded,
        "Description_encoded": description_encoded,
        "Address_encoded": address_encoded,
        "Property_Use_R": 1,  # default use case (can be updated later)
        "Property_Use_V": 0,
        "Season_Spring": 1 if season == "Spring" else 0,
        "Season_Summer": 1 if season == "Summer" else 0,
        "Season_Winter": 1 if season == "Winter" else 0,
    }

    return pd.DataFrame([features])

# Predict price
def predict_price(user_input, model):
    suburb_le, street_le, address_le, desc_le = load_encoders()
    X = prepare_features(user_input, suburb_le, street_le, address_le, desc_le)
    expected_columns = [
        'Postcode', 'Sale_Year', 'Sale_Month', 'Sale_Weekday',
        'Property_Use_R', 'Property_Use_V', 'Season_Spring', 'Season_Summer', 'Season_Winter',
        'Suburb_encoded', 'Street_Name_encoded', 'Description_encoded', 'Address_encoded'
    ]

    X_test = X[expected_columns]
    log_price = model.predict(X_test)[0]

    #SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return log_price,shap_values,X_test
