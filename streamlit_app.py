
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the theme (optional, but can enhance appearance)
st.set_page_config(layout="wide", page_title="HIV/ART Outcome Prediction 2030", page_icon="🌍")

# Add a clear title
st.title("🌍 Predicting HIV/ART Outcomes by 2030")

# Write a brief introduction
st.write("""
Welcome to the HIV/ART Outcome Prediction application.
This tool uses trained models to forecast the potential outcome
regarding ART coverage and the number of people living with HIV in a country by 2030,
based on current data and regional factors.
""")

# Load the trained models
try:
    art_model = joblib.load('art_coverage_model.pkl')
    hiv_model = joblib.load('hiv_cases_model.pkl')
    st.success("Machine learning models loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'art_coverage_model.pkl' and 'hiv_cases_model.pkl' are in the correct directory.")
    st.stop() # Stop the app if models are not loaded

# Define the thresholds for positive outcomes (these were defined in a previous step)
# Replace with the actual values used in the training process if they are not loaded
art_coverage_threshold = 90
hiv_reduction_target = 15250.00 # This was calculated as the median * 0.5 in previous steps

# Add a subheader for user input
st.subheader("Enter Country Data")
st.markdown("Please provide the current data for the country you want to predict for.")

# Input fields for features with informative labels
estimated_not_receiving_art = st.number_input("Estimated number of people not receiving ART", value=0.0, min_value=0.0)
art_coverage_ratio = st.number_input("ART coverage ratio", value=0.0, min_value=0.0, max_value=1.0, format="%.6f")
estimated_hiv_range = st.number_input("Estimated HIV range", value=0.0, min_value=0.0)
estimated_art_coverage_range = st.number_input("Estimated ART coverage range", value=0.0, min_value=0.0)

# Input for WHO Region (using selectbox for categorical feature)
who_regions = [
    'Africa',
    'Americas',
    'Eastern Mediterranean',
    'Europe',
    'South-East Asia',
    'Western Pacific'
]
selected_who_region = st.selectbox("WHO Region", who_regions)

# Create dummy variables for the selected WHO Region
who_region_dummies = {f'WHO_Region_{region}': 0 for region in who_regions}
who_region_dummies[f'WHO_Region_{selected_who_region}'] = 1

# Combine all inputs into a dictionary
input_data = {
    'Estimated number of people not receiving ART': estimated_not_receiving_art,
    'ART coverage ratio': art_coverage_ratio,
    'Estimated HIV range': estimated_hiv_range,
    'Estimated ART coverage range': estimated_art_coverage_range,
    **who_region_dummies  # Add the WHO region dummy variables
}

# Convert input data to a pandas DataFrame (required by the model)
input_df = pd.DataFrame([input_data])

# Ensure the column order of input_df matches the training data.
# We can use the columns from X_train_art as they now correctly exclude 'Outcome'.
# Access feature names from the trained model
if hasattr(art_model, 'feature_names_in_'):
    input_df = input_df.reindex(columns=art_model.feature_names_in_, fill_value=0)
else:
    # Fallback if feature_names_in_ is not available (e.g., older scikit-learn versions)
    # In a real application, you would save the training column order.
    # For this exercise, we'll assume the order from the previous successful run is correct.
    # You might need to manually list the expected columns here if the model doesn't expose them.
    expected_columns = [
       'Estimated number of people not receiving ART', 'ART coverage ratio',
       'Estimated HIV range', 'Estimated ART coverage range',
       'WHO_Region_Africa', 'WHO_Region_Americas',
       'WHO_Region_Eastern Mediterranean', 'WHO_Region_Europe',
       'WHO_Region_South-East Asia', 'WHO_Region_Western Pacific'
    ]
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)


# Add a subheader for prediction results
st.subheader("Prediction Results for 2030")

# Make predictions for ART coverage
predicted_art_coverage = art_model.predict(input_df)

# Make predictions for HIV cases
predicted_hiv_cases = hiv_model.predict(input_df)


st.write(f"Predicted Estimated ART coverage among people living with HIV (%): **{predicted_art_coverage[0]:.2f}%**")
st.write(f"Predicted Estimated number of people living with HIV: **{predicted_hiv_cases[0]:.2f}**")

# Determine the overall outcome based on the defined thresholds
predicted_outcome = "Negative"
if (predicted_art_coverage[0] >= art_coverage_threshold) and (predicted_hiv_cases[0] <= hiv_reduction_target):
    predicted_outcome = "Positive"

# Add a subheader for the overall outcome
st.subheader("Overall Predicted Outcome by 2030")

if predicted_outcome == "Positive":
    st.balloons()
    st.success(f"Based on the input, the predicted outcome by 2030 is: **{predicted_outcome}** 🎉")
else:
    st.error(f"Based on the input, the predicted outcome by 2030 is: **{predicted_outcome}** 😟")

# Include a note explaining the interpretation
st.info(f"""
**Note on Interpretation:**
A 'Positive' outcome is predicted if the estimated ART coverage is projected to be {art_coverage_threshold}% or higher AND the estimated number of people living with HIV is projected to decrease to or below {hiv_reduction_target:.2f} (a 50% reduction from the current median in the dataset).
A 'Negative' outcome is predicted if either of these conditions is not met.
These predictions are based on a linear regression model trained on historical data and should be interpreted as estimates.
""")

# Add a footer
st.markdown("---")
st.markdown("Created for the HIV/ART Outcome Prediction Project.")
