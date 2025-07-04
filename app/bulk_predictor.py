"""
bulk_predictor.py - Batch prediction module for NSW Property Price Predictor app

This module handles bulk property price predictions from CSV files uploaded by users.
It provides functions to:
- Validate required input columns for bulk prediction
- Safely encode categorical features using pretrained label encoders
- Prepare feature dataframes matching the model's expected input schema
- Generate predictions with uncertainty intervals (95% confidence) using the trained model
- Provide a Streamlit UI component for CSV file upload, validation, batch prediction, 
  and result display/download

Key points:
- Uses cached loading of label encoders and the trained XGBoost model
- Ensures feature column order and names match model training schema
- Supports robust handling of unknown categorical values by encoding them as zero
- Allows users to download a sample input CSV and predicted results CSV
"""


import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os

def get_season(month):
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"

def safe_transform_bulk(encoder, values):
    known_classes = set(encoder.classes_) if hasattr(encoder, 'classes_') else set()
    return [encoder.transform([val])[0] if val in known_classes else 0 for val in values]

def prepare_features_df(df, suburb_le, street_le, address_le, desc_le):
    # Validate required columns exist
    required_cols = ["Postcode", "Sale_Year", "Sale_Month", "Sale_Weekday", "Property_Use",
                     "Suburb", "Street_Name", "Description", "Address"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # Encode categoricals
    df["Suburb_encoded"] = safe_transform_bulk(suburb_le, df["Suburb"].astype(str))
    df["Street_Name_encoded"] = safe_transform_bulk(street_le, df["Street_Name"].astype(str))
    df["Address_encoded"] = safe_transform_bulk(address_le, df["Address"].astype(str))
    df["Description_encoded"] = safe_transform_bulk(desc_le, df["Description"].astype(str))

    # Property use
    df["Property_Use_R"] = (df["Property_Use"] == "R").astype(int)
    df["Property_Use_V"] = (df["Property_Use"] == "V").astype(int)

    # Seasons
    df["Season"] = df["Sale_Month"].apply(get_season)
    df["Season_Spring"] = (df["Season"] == "Spring").astype(int)
    df["Season_Summer"] = (df["Season"] == "Summer").astype(int)
    df["Season_Winter"] = (df["Season"] == "Winter").astype(int)

    final_cols = [
        "Postcode", "Sale_Year", "Sale_Month", "Sale_Weekday",
        "Property_Use_R", "Property_Use_V",
        "Season_Spring", "Season_Summer", "Season_Winter",
        "Suburb_encoded", "Street_Name_encoded", "Description_encoded", "Address_encoded"
    ]
    return df[final_cols]

# --- Load model only once ---
with open("models/tuned_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

#model encoders
@st.cache_resource
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

# --- Batch Prediction Function ---
def predict_bulk(csv_file):
    try:
        # Read user CSV
        df_input = csv_file

        # Validate required columns
        required_cols = [
            "Postcode", "Sale_Year", "Sale_Month", "Sale_Weekday",
            "Property_Use", "Suburb", "Street_Name", "Description", "Address"
        ]
        missing = [col for col in required_cols if col not in df_input.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Load encoders
        suburb_le, street_le, address_le, desc_le = load_encoders()

        # Prepare features
        X_all = prepare_features_df(df_input, suburb_le, street_le, address_le, desc_le)

        # Check feature alignment with model
        expected_columns = [
            "Postcode", "Sale_Year", "Sale_Month", "Sale_Weekday",
            "Property_Use_R", "Property_Use_V",
            "Season_Spring", "Season_Summer", "Season_Winter",
            "Suburb_encoded", "Street_Name_encoded", "Description_encoded", "Address_encoded"
        ]

        if list(X_all.columns) != expected_columns:
            raise ValueError("Feature mismatch: Model expects specific feature names and order.")

        # Predict with uncertainty range
        booster = model.get_booster()
        total_trees = booster.best_ntree_limit if hasattr(booster, 'best_ntree_limit') else booster.num_boosted_rounds()
        if total_trees is None:
            total_trees = booster.num_boosted_rounds()
        dropped_trees = min(10, total_trees - 1)

        predictions = []
        for _, row in X_all.iterrows():
            preds = []
            for i in range(total_trees - dropped_trees, total_trees + 1):
                pred = model.predict(pd.DataFrame([row]), iteration_range=(0, i))[0]
                preds.append(np.expm1(pred))
            mean_pred = np.mean(preds)
            lower = np.percentile(preds, 5)
            upper = np.percentile(preds, 95)
            predictions.append((int(mean_pred), int(lower), int(upper)))

        # Add to original DataFrame
        df_input["Estimated_Price"] = [f"${mean:,}" for mean, _, _ in predictions]
        df_input["95%_CI"] = [f"${low:,} - ${up:,}" for _, low, up in predictions]

        return df_input

    except Exception as e:
        return str(e)
    
# --- Streamlit CSV Upload Section ---
def bulk_prediction_ui():
    st.header("📂 Bulk Property Price Prediction")
    uploaded_file = st.file_uploader("Upload CSV file with property data", type=["csv"])
    st.markdown("📝 **Note:** Make sure your CSV contains all required columns with exact names: `Postcode`, `Sale_Year`, `Sale_Month`, `Sale_Weekday`, `Property_Use`, `Suburb`, `Street_Name`, `Description`, `Address`.")
    sample_path = os.path.join("data/raw/", "test_bulk_predictions.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button("📄 Download Sample CSV", f, file_name="sample_input.csv", mime="text/csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            result_df = predict_bulk(df)
            if isinstance(result_df, pd.DataFrame):
                st.success("✅ Predictions completed successfully!")
                st.dataframe(result_df)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions", csv, file_name="predicted_prices.csv", mime="text/csv")
            else:
                st.error(f"❌ Error: {result_df}")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
