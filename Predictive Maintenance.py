# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st

# Generate synthetic predictive maintenance dataset
np.random.seed(42)

# Generate random sensor readings for normal operation
normal_data = pd.DataFrame({
    'Sensor1': np.random.normal(loc=20, scale=5, size=1000),
    'Sensor2': np.random.normal(loc=30, scale=8, size=1000),
    'Sensor3': np.random.normal(loc=25, scale=4, size=1000),
    'Failure': 0  # 0 indicates normal operation
})

# Introduce anomalies for failing equipment
anomaly_data = pd.DataFrame({
    'Sensor1': np.random.normal(loc=10, scale=5, size=100),
    'Sensor2': np.random.normal(loc=15, scale=8, size=100),
    'Sensor3': np.random.normal(loc=18, scale=4, size=100),
    'Failure': 1  # 1 indicates equipment failure
})

# Concatenate normal and anomaly data to create the synthetic dataset
synthetic_data = pd.concat([normal_data, anomaly_data], ignore_index=True)

# Split the dataset into features (X) and labels (y)
X = synthetic_data.drop('Failure', axis=1)
y = synthetic_data['Failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a predictive maintenance model using a Random Forest classifier
model = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', RandomForestClassifier(random_state=42))  # Random Forest Classifier
])

# Train the model
model.fit(X_train, y_train)

# Streamlit App
st.title('Predictive Maintenance Model')
st.write('This app predicts equipment failure based on synthetic sensor data.')

# Display sample data
st.subheader('Sample Data:')
st.write(synthetic_data.head())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.subheader('Model Evaluation:')
st.write(f'Accuracy: {accuracy}')
st.write('Classification Report:')
st.code(classification_report(y_test, y_pred))

# Integrate with maintenance workflow (replace this with your actual implementation)
st.subheader('Maintenance Workflow:')
if any(y_pred == 1):
    st.warning('Maintenance alert: Predicted equipment failure. Schedule preventive maintenance.')
else:
    st.success('No maintenance needed.')

# Optionally, you can save the trained model for future use
# import joblib
# joblib.dump(model, 'predictive_maintenance_model.joblib')
