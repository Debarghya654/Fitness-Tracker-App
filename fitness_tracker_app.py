# fitness_tracker_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title and description
st.title("Personal Fitness Tracker")
st.write("""
This app provides personalized fitness insights and recommendations using Machine Learning.
Track your progress and achieve your fitness goals!
""")

# Sidebar for user input
st.sidebar.header("User Input")
age = st.sidebar.slider("Age", 18, 80, 25)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 100, 250, 170)
activity_level = st.sidebar.selectbox("Activity Level", ["Low", "Moderate", "High"])
goal = st.sidebar.selectbox("Fitness Goal", ["Lose Weight", "Maintain Weight", "Gain Muscle"])

# Simulated dataset
def generate_data():
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 80, 500),
        "Weight": np.random.randint(30, 200, 500),
        "Height": np.random.randint(100, 250, 500),
        "Activity_Level": np.random.choice(["Low", "Moderate", "High"], 500),
        "Goal": np.random.choice(["Lose Weight", "Maintain Weight", "Gain Muscle"], 500),
        "Recommendation": np.random.choice(["Increase Protein", "Increase Cardio", "Maintain Current Plan"], 500)
    }
    return pd.DataFrame(data)

# Load and preprocess data
data = generate_data()
activity_mapping = {"Low": 0, "Moderate": 1, "High": 2}
goal_mapping = {"Lose Weight": 0, "Maintain Weight": 1, "Gain Muscle": 2}
recommendation_mapping = {"Increase Protein": 0, "Increase Cardio": 1, "Maintain Current Plan": 2}

data["Activity_Level"] = data["Activity_Level"].map(activity_mapping)
data["Goal"] = data["Goal"].map(goal_mapping)
data["Recommendation"] = data["Recommendation"].map(recommendation_mapping)

X = data[["Age", "Weight", "Height", "Activity_Level", "Goal"]]
y = data["Recommendation"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
svm_model = SVC(probability=True)  # Enable probability predictions
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model.fit(X_train, y_train)

# Model selection
model_choice = st.sidebar.selectbox("Choose a Model", ["SVM", "Random Forest", "Logistic Regression"])
if model_choice == "SVM":
    model = svm_model
elif model_choice == "Random Forest":
    model = rf_model
else:
    model = lr_model

# Make prediction
user_data = pd.DataFrame({
    "Age": [age],
    "Weight": [weight],
    "Height": [height],
    "Activity_Level": [activity_mapping.get(activity_level, 1)],  # Default Moderate if missing
    "Goal": [goal_mapping.get(goal, 1)]  # Default Maintain Weight if missing
})

prediction = model.predict(user_data)
recommendation = {0: "Increase Protein", 1: "Increase Cardio", 2: "Maintain Current Plan"}[prediction[0]]

# Display results
st.subheader("Your Personalized Recommendation")
st.write(f"Based on your input, we recommend: **{recommendation}**")

# Model accuracy
st.subheader("Model Accuracy")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"The accuracy of the selected model is: **{accuracy:.2f}**")