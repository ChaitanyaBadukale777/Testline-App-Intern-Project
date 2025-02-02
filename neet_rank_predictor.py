import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load historical data
@st.cache_data
def load_json_data(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

# Preprocess historical data
def preprocess_data(historical_data):
    if not historical_data:
        return pd.DataFrame(), None

    df = pd.DataFrame(historical_data)

    # Extract rank from 'rank_text'
    if 'rank_text' in df.columns:
        df['neet_rank'] = df['rank_text'].str.extract(r'(\d+)').astype(float)
        df.drop(columns=['rank_text'], inplace=True)

    # Ensure required accuracy columns exist (Silently filling missing values)
    required_cols = ['bio_accuracy', 'chem_accuracy', 'physics_accuracy']
    for col in required_cols:
        df[col] = df.get(col, 50)  # Default to 50% if column is missing

    # Drop NaN values
    df.dropna(inplace=True)
    df['neet_rank'] = df['neet_rank'].astype(int)

    # Standardize data
    scaler = StandardScaler()
    df[required_cols] = scaler.fit_transform(df[required_cols])

    return df, scaler

# Train Model
def train_neet_model(df):
    X = df[['bio_accuracy', 'chem_accuracy', 'physics_accuracy']]
    y = df['neet_rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    return model

# Predict Rank
def predict_rank(model, user_data, scaler, df):
    user_df = pd.DataFrame([user_data])
    user_df[['bio_accuracy', 'chem_accuracy', 'physics_accuracy']] = scaler.transform(user_df)

    predicted_rank = model.predict(user_df)[0]
    
    # Adjust Rank to Ensure Variation
    min_rank, max_rank = df['neet_rank'].min(), df['neet_rank'].max()
    percentile = (100 - np.mean(list(user_data.values()))) / 100
    adjusted_rank = min_rank + (max_rank - min_rank) * percentile

    return max(1, int((predicted_rank + adjusted_rank) / 2))  # Average for better variation

# Streamlit UI
st.title("📊 NEET Rank Predictor")

# Load historical data
historical_data = load_json_data("historical_data.json")
quiz_data = load_json_data("quiz_endpoint.json")

df, scaler = preprocess_data(historical_data)

if not df.empty:
    model = train_neet_model(df)
    st.success("✅ Model trained successfully!")

    # Display Quiz Details
    if quiz_data:
        st.sidebar.header("📚 Quiz Information")
        st.sidebar.write(f"**Title:** {quiz_data.get('title', 'Unknown')}")
        st.sidebar.write(f"**Topic:** {quiz_data.get('topic', 'Unknown')}")
        st.sidebar.write(f"**Difficulty Level:** {quiz_data.get('difficulty_level', 'N/A')}")
        st.sidebar.write(f"**Questions Count:** {quiz_data.get('questions_count', 'N/A')}")

    # User Input Section
    st.sidebar.header("🔍 User Input")
    user_input = {
        'bio_accuracy': st.sidebar.number_input("Biology Accuracy (%)", min_value=0, max_value=100, value=50),
        'chem_accuracy': st.sidebar.number_input("Chemistry Accuracy (%)", min_value=0, max_value=100, value=50),
        'physics_accuracy': st.sidebar.number_input("Physics Accuracy (%)", min_value=0, max_value=100, value=50),
    }

    if st.sidebar.button("🔮 Predict Rank"):
        predicted_rank = predict_rank(model, user_input, scaler, df)
        st.success(f"🎯 Predicted NEET Rank: **{predicted_rank}**")

else:
    st.warning("⚠️ Failed to load data. Check historical_data.json and quiz_endpoint.json files.")
