import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Custom CSS to increase input box size, set background color, and widen columns
st.markdown("""
    <style>
        /* Increase the width of input boxes */
        input[type="number"] {
            width: 100% !important;
            padding: 10px !important;
            font-size: 16px !important;
        }
        /* Set the background color of the app */
        .main {
            background-color: black !important;
        }
        /* Adjust width of columns to make inputs look wider */
        .css-1adrfps, .css-1oe5cao {
            flex: 1 !important;
            max-width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained models
knn_model = joblib.load('knn_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')

# Title of the app
st.title("Mobile Behavior Detector")

# Input features for prediction
st.header("Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    feature1 = st.number_input("Feature 1 (Screen On Time - hours)", value=0.0)
    feature4 = st.number_input("Feature 4 (Notification Responses - per day)", value=0.0)
    feature7 = st.number_input("Feature 7 (App Install Count - total)", value=0.0)
    feature10 = st.number_input("Feature 10 (Session Duration - minutes)", value=0.0)
    feature13 = st.number_input("Feature 13 (Average Data Consumption - MB per day)", value=0.0)

with col2:
    feature2 = st.number_input("Feature 2 (App Usage Time - hours per day)", value=0.0)
    feature5 = st.number_input("Feature 5 (Purchase Frequency - per month)", value=0.0)
    feature8 = st.number_input("Feature 8 (Time Spent on Apps - hours per week)", value=0.0)
    feature11 = st.number_input("Feature 11 (Features Used - count)", value=0.0)

with col3:
    feature3 = st.number_input("Feature 3 (Data Usage - MB per day)", value=0.0)
    feature6 = st.number_input("Feature 6 (Social Media Engagement - posts per week)", value=0.0)
    feature9 = st.number_input("Feature 9 (Interaction Rate - interactions per hour)", value=0.0)
    feature12 = st.number_input("Feature 12 (Feedback Score - scale 1-5)", value=0.0)

# When the user clicks the predict button
if st.button("Predict"):
    # Prepare the input data
    features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6,
                        feature7, feature8, feature9, feature10, feature11, feature12, feature13]])

    # Make predictions using each model
    knn_prediction = knn_model.predict(features)[0]
    dt_prediction = dt_model.predict(features)[0]
    nb_prediction = nb_model.predict(features)[0]

    # Display the predictions
    st.subheader("Predictions:")
    st.write(f"KNN Prediction: {'Heavy User' if knn_prediction == 1 else 'Light User'}")
    st.write(f"Decision Tree Prediction: {'Heavy User' if dt_prediction == 1 else 'Light User'}")
    st.write(f"Naive Bayes Prediction: {'Heavy User' if nb_prediction == 1 else 'Light User'}")

    # Provide insights based on predictions
    st.subheader("Insights:")
    if knn_prediction == 1:
        st.write("KNN: This user is classified as a 'Heavy User'. Consider sending special offers to increase engagement.")
    else:
        st.write("KNN: This user is classified as a 'Light User'. A personalized recommendation can enhance their experience.")

    if dt_prediction == 1:
        st.write("Decision Tree: Target this user group with tailored marketing strategies.")
    else:
        st.write("Decision Tree: Consider exploring new features that might attract these users.")

    if nb_prediction == 1:
        st.write("Naive Bayes: This user exhibits heavy usage patterns. Engage with specific offers.")
    else:
        st.write("Naive Bayes: Light usage detected. Consider improving user engagement strategies.")

# Batch prediction functionality
st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Ensure the data has the necessary features
    required_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
                        'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
                        'Feature11', 'Feature12', 'Feature13']

    if all(feature in data.columns for feature in required_features):
        # Prepare the data for predictions
        batch_features = data[required_features].values

        # Make batch predictions
        knn_batch_predictions = knn_model.predict(batch_features)
        dt_batch_predictions = dt_model.predict(batch_features)
        nb_batch_predictions = nb_model.predict(batch_features)

        # Store predictions in DataFrame
        data['KNN Prediction'] = knn_batch_predictions
        data['Decision Tree Prediction'] = dt_batch_predictions
        data['Naive Bayes Prediction'] = nb_batch_predictions

        # Display the predictions
        st.subheader("Batch Predictions:")
        st.write(data)

        # Provide insights based on batch predictions
        for index, row in data.iterrows():
            st.write(f"User {index + 1}:")
            if row['KNN Prediction'] == 1:
                st.write(" - KNN: Heavy User - Consider sending special offers.")
            else:
                st.write(" - KNN: Light User - Personalize their experience.")

            if row['Decision Tree Prediction'] == 1:
                st.write(" - Decision Tree: Target this user group with tailored marketing strategies.")
            else:
                st.write(" - Decision Tree: Consider exploring new features that might attract these users.")

            if row['Naive Bayes Prediction'] == 1:
                st.write(" - Naive Bayes: Heavy User - Engage with specific offers.")
            else:
                st.write(" - Naive Bayes: Light User - Improve engagement strategies.")
    else:
        st.write("Error: The uploaded CSV file must contain the required features.")
