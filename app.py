import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
from collections import Counter
from pathlib import Path

# Load models
@st.cache_resource
def get_image_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set the base directory as the current working directory of the script
BASE_DIR = Path(__file__).resolve().parent  # The project's root directory
MODELS_DIR = BASE_DIR / 'models'            # The models directory
IMAGE_DIR = BASE_DIR / 'data'            # The models directory

# Load images with relative paths
img = get_image_as_base64(IMAGE_DIR / 'back2.jpg')
img2 = get_image_as_base64(IMAGE_DIR / 'for1.jpg')

def load_models():
    # Use relative paths for models
    model_SMV_File = MODELS_DIR / 'svm_model780k_N.pkl'
    scaler_SMV_File = MODELS_DIR / 'scaler_780k_n.pkl'
    model_RF_File = MODELS_DIR / 'RF_model450k.pkl'
    model_KNN_File = MODELS_DIR / 'knn_450k_Test.pkl'
    scaler_Knn_File = MODELS_DIR / 'scaler_450k_Test.pkl'
    model_XGBoost_File = MODELS_DIR / 'xgboost_450k.pkl'

    model_SMV = joblib.load(model_SMV_File)
    scaler_SVM = joblib.load(scaler_SMV_File)
    model_RF = joblib.load(model_RF_File)
    model_KNN = joblib.load(model_KNN_File)
    scaler_Knn = joblib.load(scaler_Knn_File)
    model_XGBoost = joblib.load(model_XGBoost_File)

    return model_SMV, scaler_SVM, model_RF, model_KNN, scaler_Knn, model_XGBoost

# Load the models
model_SMV, scaler_SVM, model_RF, model_KNN, scaler_Knn, model_XGBoost = load_models()

# Add custom CSS with background image
st.markdown(f"""
    <style>
    .main {{
        background-size: cover;
        background-color: #ba45ca;
        font-family: 'Arial', sans-serif;
    }}
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 100%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
    }}
    [data-testid="stSidebarContent"]{{
        background-image: url("data:image/png;base64,{img2}");
        background-size: 120%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
    }}
    .low-risk {{
        color: green;
        font-weight: bold;
        font-size: 20px;
    }}
    .high-risk {{
        color: orange;
        font-weight: bold;
        font-size: 20px;
    }}
    .very-high-risk {{
        color: red;
        font-weight: bold;
        font-size: 20px;
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 18px;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    .stTextInput {{
        font-size: 18px;
    }}
    .css-18e3th9 {{
        padding-top: 3rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# Title with styling
st.markdown("""
    <h1 style='
        text-align: center; 
        color: #f0c4f0; 
        text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>
        Cancer Risk Prediction
    </h1>
    """, unsafe_allow_html=True)

st.write("### Please input the following information:")

# Define feature names
feature_names = ['age_group_5_years', 'first_degree_hx', 'age_menarche', 'age_first_birth', 'BIRADS_breast_density',
                 'current_hrt', 'menopaus', 'bmi_group', 'biophx', 'breast_cancer_history']

import streamlit as st

# Input fields with descriptive labels and format functions

age_group_5_years = st.selectbox(
    'Age Group (in 5-year increments)',
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    format_func=lambda x: '18-29 years' if x == 1 else
                          '30-34 years' if x == 2 else
                          '35-39 years' if x == 3 else
                          '40-44 years' if x == 4 else
                          '45-49 years' if x == 5 else
                          '50-54 years' if x == 6 else
                          '55-59 years' if x == 7 else
                          '60-64 years' if x == 8 else
                          '65-69 years' if x == 9 else
                          '70-74 years' if x == 10 else
                          '75-79 years' if x == 11 else
                          '80-84 years' if x == 12 else
                          '85+ years'
)

first_degree_hx = st.selectbox(
    'First Degree Family History',
    options=[0, 1, 9],
    format_func=lambda x: 'No' if x == 0 else 'Yes' if x == 1 else 'Unknown'
)

age_menarche = st.selectbox(
    'Age at Menarche',
    options=[0, 1, 2, 9],
    format_func=lambda x: '> 14 years' if x == 0 else
                          '12-13 years' if x == 1 else
                          '< 12 years' if x == 2 else
                          'Unknown'
)

age_first_birth = st.selectbox(
    'Age at First Birth',
    options=[0, 1, 2, 3, 4, 9],
    format_func=lambda x: '< 20 years' if x == 0 else
                          '20-24 years' if x == 1 else
                          '25-29 years' if x == 2 else
                          '> 30 years' if x == 3 else
                          'Nulliparous' if x == 4 else
                          'Unknown'
)

BIRADS_breast_density = st.selectbox(
    'BIRADS Breast Density (1-4)',
    options=[1, 2, 3, 4, 9],
    format_func=lambda x: 'Almost entirely fat' if x == 1 else
                          'Scattered fibroglandular densities' if x == 2 else
                          'Heterogeneously dense' if x == 3 else
                          'Extremely dense' if x == 4 else
                          'Unknown or different measurement system'
)

current_hrt = st.selectbox(
    'Current HRT Use',
    options=[0, 1, 9],
    format_func=lambda x: 'No' if x == 0 else 'Yes' if x == 1 else 'Unknown'
)

menopaus = st.selectbox(
    'Menopause Status',
    options=[1, 2, 3, 9],
    format_func=lambda x: 'Pre- or peri-menopausal' if x == 1 else
                          'Post-menopausal' if x == 2 else
                          'Surgical menopause' if x == 3 else
                          'Unknown'
)

bmi_group = st.selectbox(
    'BMI Group',
    options=[1, 2, 3, 4, 9],
    format_func=lambda x: '10-24.99 (Normal weight)' if x == 1 else
                          '25-29.99 (Overweight)' if x == 2 else
                          '30-34.99 (Obesity I)' if x == 3 else
                          '35 or more (Obesity II)' if x == 4 else
                          'Unknown'
)

biophx = st.selectbox(
    'Biopsy History',
    options=[0, 1, 9],
    format_func=lambda x: 'No' if x == 0 else 'Yes' if x == 1 else 'Unknown'
)

breast_cancer_history = st.selectbox(
    'Breast Cancer History',
    options=[0, 1, 9],
    format_func=lambda x: 'No' if x == 0 else 'Yes' if x == 1 else 'Unknown'
)


# Collect the inputs into a DataFrame
user_input = pd.DataFrame([[age_group_5_years, first_degree_hx, age_menarche, age_first_birth, BIRADS_breast_density,
                            current_hrt, menopaus, bmi_group, biophx, breast_cancer_history]], columns=feature_names)

# Display user input
st.write("### Input Data:")
st.write(user_input)

# Custom Voting Logic with Tie-Breaking
def custom_hard_voting(predictions):
    # Get the most common prediction
    vote_counts = Counter(predictions)
    most_common = vote_counts.most_common()

    # Check for ties
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # We have a tie, we need to resolve it
        risk_levels = [level for level, count in most_common]
        
        # Define the hierarchy of risk levels
        risk_priority = {2: 1, 3: 2, 4: 3}  # 2: Low, 3: Intermediate, 4: High
        
        # Determine the highest priority risk in the tied predictions
        highest_priority_risk = max(risk_levels, key=lambda x: risk_priority[x])
        return highest_priority_risk
    else:
        return most_common[0][0]  # Return the most common prediction

# Sidebar for predictions
if st.button('Predict'):
    with st.sidebar:
        st.write("## Prediction Results")
        
        # Scale and get predictions from each model
        Test_data_scaled_SVM = scaler_SVM.transform(user_input)
        rf_pred = model_RF.predict(user_input)  # Random Forest
        xgb_pred = model_XGBoost.predict(user_input)  # XGBoost
        knn_pred = model_KNN.predict(scaler_Knn.transform(user_input))  # KNN (scaled input)
        svm_pred = model_SMV.predict(Test_data_scaled_SVM)  # SVM (scaled input)

        # Remap XGBoost predictions
        xgb_pred_remapped = np.where(xgb_pred == 0, 2, np.where(xgb_pred == 1, 3, 4))

        # Combine predictions for voting
        predictions = np.array([rf_pred, xgb_pred_remapped, knn_pred, svm_pred])
        
        # Get the final prediction
        final_prediction = custom_hard_voting(predictions.flatten())
        
        if final_prediction == 2:
            st.markdown("""
                <h2 style='
                    text-align: left; 
                    color: #f0c4f0; 
                    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>
                    Final Predicted Risk Category: low risk
                </h2>
                """, unsafe_allow_html=True)
        elif final_prediction == 3:
            st.markdown("""
                <h2 style='
                    text-align: left; 
                    color: #f0c4f0; 
                    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>
                    Final Predicted Risk Category: high risk
                </h2>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <h2 style='
                    text-align: left; 
                    color: #f0c4f0; 
                    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>
                    Final Predicted Risk Category: very high risk
                </h2>
                """, unsafe_allow_html=True)
        
