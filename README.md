NSW Property Price Prediction 🏡
Predicting residential property sale prices across New South Wales (NSW), Australia — leveraging real-world bulk property sales data, advanced feature engineering, and state-of-the-art XGBoost modeling, wrapped in an interactive Streamlit UI with SHAP explainability.

Project Overview
This project builds a robust machine learning pipeline to estimate property sale prices based on historical bulk sales data from NSW (2019–2025). The system empowers users to:

Predict individual property sale prices using features like location, date of sale, property type, and more.

Upload CSV files with multiple properties for batch predictions.

Understand feature impacts on predictions via SHAP (SHapley Additive exPlanations).

Visualize historical price trends by suburb.

Data Collection & Preprocessing
Source: NSW Bulk Property Sales dataset (2019–2025), downloaded in .DAT format from official government portals.

Parsing: Custom Python script (parser.py) recursively extracts transaction records from weekly .DAT files.

Filtering: Focused on the top 20 most popular suburbs in NSW for quality and relevance.

Cleaning: Dropped incomplete records, standardized suburb names, and converted date and price fields.

Output: Clean CSV file with approximately 95,000+ reliable property sales records spanning key features.

Feature Engineering
Temporal features extracted from contract date:

Sale_Year, Sale_Month, Sale_Weekday, and derived Season (Summer, Autumn, Winter, Spring).

Categorical variables encoded via label encoding and one-hot encoding for model compatibility.

Sale prices log-transformed (Log_Sale_Price) to stabilize variance and improve model learning.

Irrelevant or sparse features dropped for simplicity (Zone, Land_Size_sqm, Street_No).

Final processed dataset consists of ~95,000 rows and ~18 carefully engineered features.

Model Training & Evaluation
Algorithm: Extreme Gradient Boosting (XGBoost) regressor.

Hyperparameter tuning: Exhaustive grid search with cross-validation to optimize n_estimators, max_depth, learning_rate, subsample, and colsample_bytree.

Performance on test set:

RMSE: ~[0.45]

MAE: ~[0.27]

R² Score: ~[0.5979]

Feature Importance: Visualized to identify key predictors influencing property prices.

Application UI
Built with Streamlit for an intuitive web interface.

Users can input property details via dropdowns and date selectors for instant predictions.

SHAP explanations help users understand feature contributions to each prediction.

Historical price trends for selected suburbs displayed via interactive Altair charts.

Bulk CSV upload feature for batch property price prediction and easy download of results.

Project Structure
text
Copy code
.
app/
├── app.py                   # Streamlit application entry point
├── bulk_predictor.py        # Batch CSV prediction module for Streamlit
├── How_to_use.py            # User guide page for the app
├── predict.py               # Core prediction logic and SHAP explainability

data/
├── raw/
│   ├── clean_sales_data.csv # Parsed and cleaned sales data CSV
│   └── test_bulk_predictions.csv  # Sample CSV for bulk upload testing
├── processed/
│   └── feature_engineered_data.csv # Feature engineered dataset CSV

models/
├── tuned_xgb_model.pkl      # Trained XGBoost model
├── suburb_encoder.pkl       # Label encoders for categorical features
├── street_encoder.pkl
├── address_encoder.pkl
└── description_encoder.pkl

notebooks/
├── EDA.ipynb                # Exploratory Data Analysis notebook
├── feature_engineering.ipynb# Feature engineering notebook
└── Model_training.ipynb     # Model training and tuning notebook

src/
├── parser.py                # Parsing and cleaning script for raw .DAT files
└── Preprocess.py            # Additional preprocessing utilities

.gitignore
README.md
requirements.txt

documentation
How to Run
Clone the repository.

Place raw NSW bulk sales .DAT files in data/raw/ directory.

Run parser.py to extract and clean data.

Execute the feature engineering notebook or script.

Train the model via train_model.ipynb or load the pre-trained model.

Launch the Streamlit UI:

bash
Copy code
streamlit run app.py
Key Technologies
Python 3.x

Pandas, NumPy for data manipulation

XGBoost for high-performance regression

Scikit-learn for preprocessing and evaluation

Streamlit for interactive web UI

SHAP for model interpretability

Altair for visualization

Here is a quick demo of the Streamlit app showcasing the property price prediction, SHAP explainability, and bulk CSV upload feature:

Demo

![App Demo](https://drive.google.com/file/d/1BLi7mHjKulkqd6MrFz-DbqCZf4-h5RTK/view?usp=sharing)

Contributions

Contributions, issues, and feature requests are very welcome!  
If you'd like to contribute:

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m 'Add some feature'`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request

Please make sure your code follows the existing style and includes appropriate comments.

---

Challenges & Learnings

- Handling and parsing large raw `.DAT` files with inconsistent formatting from the NSW bulk sales dataset was complex but rewarding.  
- Designing robust feature engineering pipelines that balance data quality and predictive power.  
- Implementing model interpretability using SHAP to build trust and transparency for users.  
- Developing an intuitive Streamlit UI that supports both single and bulk predictions while managing edge cases gracefully.  
- Ensuring the model performs well despite the variability in property descriptions and locations.

Future Enhancements
Incorporate external economic and infrastructure data for richer modeling.

Extend prediction support beyond top 20 suburbs.

Deploy as a scalable cloud application with REST API endpoints.

Add more granular property features (e.g., number of bedrooms, bathrooms).

Author & Contact
Built with ❤️ by Ambarish Shashank Gadgil

LinkedIn (https://www.linkedin.com/in/ambarish-gadgil-484b9a203/)

