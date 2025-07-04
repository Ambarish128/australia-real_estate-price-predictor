"""
app.py - Main Streamlit application for the NSW Property Price Predictor

This script implements the user interface for the property price prediction app.
It allows users to:
- Select property details interactively (suburb, street, address, date, property type)
- Predict the property sale price with uncertainty intervals
- Visualize historical price trends for the selected suburb
- View SHAP explainability to understand feature impact on predictions
- Upload bulk CSV files for batch predictions (via integration with bulk_predictor.py)
- Navigate between the main prediction page and a "How to Use" info page

Key functionalities:
- Loads pre-trained XGBoost model and label encoders
- Caches data and model loading for performance
- Handles feature preparation and safe categorical encoding
- Provides an interactive and user-friendly UI with Streamlit components and Altair charts
"""


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from bulk_predictor import bulk_prediction_ui
import datetime
from How_to_use import show_how_to_use_page

def navbar():
    st.markdown(
        """
        <style>
            .navbar {
                display: flex;
                fle
                gap: 0px;
                margin-bottom: 20px;
                flex
            }
            .nav-button {
                background-color: #f0f2f6;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                border-radius: 8px;
                transition: background-color 0.3s;      

            }
            .nav-button:hover {
                background-color: #d9e3f0;
            }
            .nav-button.selected {
                background-color: #0e1117;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    nav_options = ["Home", "How to Use"]
    cols = st.columns(len(nav_options))

    for i, page in enumerate(nav_options):
        if cols[i].button(page):
            st.session_state["page"] = page

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"



# --- Load model and encoders ---
@st.cache_resource
def load_model_and_encoders():
    with open("models/tuned_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/suburb_encoder.pkl", "rb") as f:
        suburb_le = pickle.load(f)
    with open("models/street_encoder.pkl", "rb") as f:
        street_le = pickle.load(f)
    with open("models/address_encoder.pkl", "rb") as f:
        address_le = pickle.load(f)
    with open("models/description_encoder.pkl", "rb") as f:
        desc_le = pickle.load(f)
    return model, suburb_le, street_le, address_le, desc_le

# --- Load and cache data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/feature_engineered_data.csv")

@st.cache_data
def create_mapping_df(df):
    return df[['Suburb', 'Street_Name', 'Address', 'Postcode', 'Description']].drop_duplicates()

# --- Season helper ---
def get_season(month):
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"

# --- Encode categorical safely ---
def safe_transform(encoder, val):
    if val in encoder.classes_:
        return encoder.transform([val])[0]
    else:
        return 0

# --- Prepare features for prediction ---
def prepare_features(inputs, encoders):
    suburb_le, street_le, address_le, desc_le = encoders
    
    suburb_encoded = safe_transform(suburb_le, inputs['Suburb'])
    street_encoded = safe_transform(street_le, inputs['Street_Name'])
    address_encoded = safe_transform(address_le, inputs['Address'])
    description_encoded = safe_transform(desc_le, inputs['Description'])
    
    season = get_season(inputs['Sale_Month'])
    
    features = {
        "Postcode": int(inputs['Postcode']),
        "Sale_Year": int(inputs['Sale_Year']),
        "Sale_Month": int(inputs['Sale_Month']),
        "Sale_Weekday": int(inputs['Sale_Weekday']),
        "Property_Use_R": 1 if inputs['Property_Use'] == 'R' else 0,
        "Property_Use_V": 1 if inputs['Property_Use'] == 'V' else 0,
        "Season_Spring": 1 if season == "Spring" else 0,
        "Season_Summer": 1 if season == "Summer" else 0,
        "Season_Winter": 1 if season == "Winter" else 0,
        "Suburb_encoded": suburb_encoded,
        "Street_Name_encoded": street_encoded,
        "Description_encoded": description_encoded,
        "Address_encoded": address_encoded
    }
    
    return pd.DataFrame([features])

# --- Main app ---
def main():
    st.set_page_config(layout="wide")
    navbar()
    st.title("🏡 NSW Property Price Predictor")

    

    if st.session_state["page"] == "Home":
        run_prediction_page()
    elif st.session_state["page"] == "How to Use":
        show_how_to_use_page() 
    
def run_prediction_page():
        model, suburb_le, street_le, address_le, desc_le = load_model_and_encoders()
        df = load_data()
        mapping_df = create_mapping_df(df)

        # --- Layout: Left for prediction, right for trend/shap ---
        col1, col2 = st.columns([1.3, 1.7])  # Adjust column width ratio as desired

        with col1:
            st.header("📋 Property Input & Prediction")

            suburb_options = sorted(mapping_df['Suburb'].unique())
            selected_suburb = st.selectbox("Select Suburb", suburb_options)

            streets_filtered = mapping_df[mapping_df['Suburb'] == selected_suburb]['Street_Name'].unique()
            selected_street = st.selectbox("Select Street Name", sorted(streets_filtered))

            addresses_filtered = mapping_df[
                (mapping_df['Suburb'] == selected_suburb) &
                (mapping_df['Street_Name'] == selected_street)
            ]['Address'].unique()
            selected_address = st.selectbox("Select Address", sorted(addresses_filtered))

            postcode = int(mapping_df[
                (mapping_df['Suburb'] == selected_suburb) &
                (mapping_df['Street_Name'] == selected_street) &
                (mapping_df['Address'] == selected_address)
            ]['Postcode'].values[0])
            st.markdown(f"**Postcode (auto-filled):** `{postcode}`")

            description_options = sorted(mapping_df['Description'].unique())
            selected_description = st.selectbox("Property Description", description_options)

            current_year = datetime.datetime.now().year
            sale_year = st.selectbox("Sale Year", list(range(2019, 2031)), index=6)
            if sale_year>current_year:
                st.warning(f"⚠️ Note: Prediction is made for a future year {sale_year}.")
            sale_month = st.selectbox("Sale Month", list(range(1, 13)), index=0)
            sale_weekday = st.selectbox("Sale Weekday (0=Monday)", list(range(7)), index=0)

            property_use_options = ['R', 'V']
            selected_property_use = st.selectbox("Property Use", property_use_options)

            if st.button("💰 Predict Sale Price"):
                user_input = {
                    "Postcode": postcode,
                    "Sale_Year": sale_year,
                    "Sale_Month": sale_month,
                    "Sale_Weekday": sale_weekday,
                    "Property_Use": selected_property_use,
                    "Suburb": selected_suburb,
                    "Street_Name": selected_street,
                    "Description": selected_description,
                    "Address": selected_address
                }

                try:
                    input_df = prepare_features(user_input, (suburb_le, street_le, address_le, desc_le))

                    booster = model.get_booster()
                    total_trees = booster.best_ntree_limit if hasattr(booster, 'best_ntree_limit') else booster.num_boosted_rounds()
                    if total_trees is None:
                        total_trees = booster.num_boosted_rounds()
                    dropped_trees = min(10, total_trees - 1)

                    n_predictions = [
                        np.expm1(model.predict(input_df, iteration_range=(0, i))[0])
                        for i in range(total_trees - dropped_trees, total_trees + 1)
                    ]

                    price_mean = int(np.mean(n_predictions))
                    price_lower = int(np.percentile(n_predictions, 5))
                    price_upper = int(np.percentile(n_predictions, 95))

                    st.success(f"🏷️ Estimated Price range (95% CI): AUD: {price_lower:,} to AUD: {price_upper:,}")
                    log_price = model.predict(input_df)[0]
                    price = int(np.expm1(log_price))
                    st.success(f"🏷️ Most probably it will cost AUD: {price:,}")

                    # SHAP Explainability
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)

                    st.session_state["shap_df"] = pd.DataFrame({
                        "Feature": input_df.columns,
                        "SHAP Value": shap_values[0]
                    }).sort_values(by="SHAP Value", key=abs, ascending=False)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with col2:
            st.header(f"📈 Historical Price Trends in {selected_suburb}")

            try:
                trend_df = df[df['Suburb'] == selected_suburb]

                if not trend_df.empty:
                    trend_grouped = trend_df.groupby(['Sale_Year', 'Sale_Month'])['Sale_Price'].mean().reset_index()
                    trend_grouped['Date'] = pd.to_datetime(
                        trend_grouped['Sale_Year'].astype(str) + '-' + trend_grouped['Sale_Month'].astype(str) + '-01')
                    trend_grouped = trend_grouped.sort_values('Date')

                    import altair as alt
                    chart = alt.Chart(trend_grouped).mark_line(point=True).encode(
                        x='Date:T',
                        y=alt.Y('Sale_Price:Q', title='Average Sale Price'),
                        tooltip=['Date:T', 'Sale_Price']
                    ).properties(
                        width=650,
                        height=300
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No data available for the selected suburb.")
            except Exception as e:
                st.error(f"Failed to load price trend chart: {e}")

            # --- SHAP Summary ---
            if "shap_df" in st.session_state:
                st.subheader("🔍 Feature Impact (SHAP Explanation)")
                st.markdown("These values show how each feature contributed to the prediction.")
                st.dataframe(st.session_state["shap_df"], use_container_width=True)
            
        st.title("🏡 Wanna Predict price by Uploading CSV?")  

        
        bulk_prediction_ui()


    


    

if __name__ == "__main__":
    main()
