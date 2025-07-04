"""
How_to_use.py - Informational page for the NSW Property Price Predictor app

This module provides a Streamlit page that guides users on how to use the app effectively.
It includes:
- Overview of the app’s features and capabilities
- Step-by-step instructions for making individual property price predictions
- Instructions for uploading bulk property data via CSV and obtaining batch predictions
- Explanation of SHAP values to help users understand model predictions
- Troubleshooting tips and friendly help information
- Placeholder for GitHub link and developer credit

This page is designed to make the app user-friendly and accessible even for non-technical users.
"""
import streamlit as st

def show_how_to_use_page():
    

    st.title("📘 How to Use the NSW Property Price Predictor")
    st.markdown("Welcome! 🏡 Let's walk you through using this app like a pro – no jargon, just plain fun!")

    st.header("🚀 What can this app do?")
    st.markdown("""
    This app helps you:
    - Predict how much a property might sell for based on its features 🏷️
    - Understand what factors influence the price using SHAP explainability 📊
    - Upload a bunch of properties via CSV and get predictions in bulk 📂
    """)

    st.header("🪄 How to make a prediction")
    st.markdown("""
    1. Head to the **main page**.
    2. Select your **Suburb**, **Street**, and **Address** from the dropdowns.
    3. Choose **Property Type** (R = Residential, V = Vacant).
    4. Pick the **year, month, and weekday** of sale.
    5. Hit the **💰 Predict Sale Price** button.
    6. Voilà! You’ll see:
    - A most likely sale price.
    - A price range (95% confidence).
    - A cool explanation of which features influenced the prediction.
    """)

    st.header("📂 Bulk Predictions via CSV")
    st.markdown("""
    Want to save time? Upload a CSV of multiple properties and let the model do all the work.

    **Steps:**
    - Go to the **Bulk Prediction** section on the homepage.
    - Download the 📄 Sample CSV to see the required format(from the main page).
    - Fill it with your data.
    - Upload it using the file uploader.
    - Download the predictions in seconds!

    **Required columns in your CSV:**
    Postcode, Sale_Year, Sale_Month, Sale_Weekday, Property_Use,
    Suburb, Street_Name, Description, Address
    """)

    st.header("🧠 What’s SHAP? (For the curious)")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) tells you **why** the model made a certain prediction.  
    It’s like seeing which ingredients went into your cake 🍰... and which one made it extra tasty (or expensive 💸).
    """)

    st.header("🙋 Need help?")
    st.markdown("""
    If you run into any issues or think the prediction looks off:
    - Make sure your inputs are correct.
    - Check that you're using values that were seen during training (e.g., existing suburbs/streets).
    - Or just reload the app and try again!

   Here to make data science fun, not frustrating 😄
    """)    

    st.header('Wanna Dive🤿 deep into the code')
    st.markdown("""
    My Github : https://github.com/Ambarish128/australia-real_estate-price-predictor
                
    Feel Free to Explore and suggest any improvements :)
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ by [Ambarish Shashank Gadgil](https://www.linkedin.com/in/ambarish-gadgil-484b9a203/)")
