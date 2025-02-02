# Student Rank Predictor

<!-- ## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt -->



Student NEET Rank Predictor

Project Overview

The Student NEET Rank Predictor is a FastAPI-based application that analyzes students' quiz performance to generate insights and predict their possible NEET rank. The project involves:

Data Analysis: Identifying patterns in student performance across subjects and difficulty levels.

Insights Generation: Highlighting weak areas, improvement trends, and performance gaps.

Rank Prediction: Using a probabilistic model trained on past NEET scores to estimate a student's rank.

Features

Upload and analyze quiz performance data.

Generate insights on subject-wise strengths and weaknesses.

Predict the probable NEET rank based on historical performance.

Suggest eligible colleges based on predicted rank.

Setup Instructions

Prerequisites

Ensure you have Python installed (>=3.7). Then, install the required dependencies:

pip install fastapi>=0.68.0 uvicorn>=0.15.0 pandas>=1.3.0 numpy>=1.21.0 \
scikit-learn>=0.24.2 matplotlib>=3.4.2 python-multipart>=0.0.5 pydantic>=1.8.2

Running the Application

Start the FastAPI server

uvicorn main:app --reload

Access API documentation

Open http://127.0.0.1:8000/docs to explore API endpoints using Swagger UI.

API Endpoints

POST /analyze: Analyze quiz performance and generate insights.

POST /predict-rank: Predict the NEET rank based on historical data.

GET /predict-college/{rank}: Get eligible colleges based on predicted rank.

Approach

1. Data Analysis

Parse student quiz submissions and historical performance.

Identify patterns based on subject accuracy, difficulty levels, and response trends.

2. Insights Generation

Highlight weak subjects and areas requiring improvement.

Track performance trends over time.

3. Rank Prediction

Train a Linear Regression model using past NEET score data.

Predict a student’s score using quiz performance metrics.

Map the predicted score to a rank based on historical NEET rank trends.

Screenshots

(Add relevant screenshots of data analysis, insights, and predictions here.)

Future Improvements

Improve model accuracy with real NEET exam data.

Add support for adaptive learning recommendations.

Enhance visualization of insights and trends.

Contributors

(Chaitanya Badukale)

License

MIT License