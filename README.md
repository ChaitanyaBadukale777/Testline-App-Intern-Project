# Student Rank Predictor


Project Overview

The Student NEET Rank Predictor is a Streamlit-based application that analyzes students' quiz performance and predicts their possible NEET rank. The project involves:

Key Features

Upload and analyze quiz performance data from historical_data.json and quiz_endpoint.json.

Generate insights on subject-wise strengths and weaknesses.

Predict the probable NEET rank using a Gradient Boosting Regressor trained on historical performance.

Fetch and display quiz details (title, topic, difficulty, and questions count) from quiz_endpoint.json.

Adjust rank prediction dynamically based on accuracy changes.


How It Works

1. Data Analysis

Parses student quiz submissions and historical performance from historical_data.json.

Identifies patterns based on subject accuracy and difficulty levels.

2. Insights Generation

Extracts title, topic, difficulty level, and question count from quiz_endpoint.json.

Highlights weak subjects and areas needing improvement.

3. Rank Prediction

Trains a Gradient Boosting Regressor model using past NEET score data.

Predicts a student’s NEET rank using accuracy inputs in Biology, Chemistry, and Physics.

Adjusts rank dynamically to ensure variation in predictions.

User Guide

Uploading Quiz Data

Ensure your historical_data.json file contains quiz performance details.

Ensure quiz_endpoint.json has quiz metadata like title, topic, and question count.

Using the Predictor

Enter your Biology, Chemistry, and Physics accuracy percentages.

Click Predict Rank to generate the estimated NEET rank.

The sidebar will display quiz details fetched from quiz_endpoint.json
