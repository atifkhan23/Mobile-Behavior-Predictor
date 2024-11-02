# Mobile Behavior Detector

## Overview
The **Mobile Behavior Detector** is a web application developed using Streamlit that leverages machine learning models to predict user behavior on mobile devices. This application allows users to input various features related to mobile usage and receive predictions about whether they are classified as "Heavy Users" or "Light Users." Additionally, it supports batch predictions through CSV file uploads.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Single User Predictions**: Input features to receive real-time predictions using KNN, Decision Tree, and Naive Bayes models.
- **Batch Predictions**: Upload a CSV file to make predictions for multiple users simultaneously.
- **Insights**: The application provides insights based on the predictions to enhance user engagement strategies.
- **User-Friendly Interface**: An interactive and responsive web interface built with Streamlit.

## Technologies Used
- Python
- Streamlit
- scikit-learn (for machine learning models)
- NumPy (for numerical operations)
- Pandas (for data manipulation)
- joblib (for loading pre-trained models)
