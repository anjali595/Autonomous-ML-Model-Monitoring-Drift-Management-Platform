# Autonomous-ML-Model-Monitoring-Drift-Management-Platform

# Project Overview

Machine learning models do not stay accurate forever. After deployment, the data that the model receives in the real world may slowly change. Because of this, the model’s predictions may become worse over time.

This project focuses on solving that problem.

The Autonomous ML Model Monitoring & Drift Management Platform is a system that monitors a deployed machine learning model, detects when the incoming data starts to change, and automatically triggers retraining to maintain model performance.

The goal of this project is to demonstrate how machine learning models can be maintained in production using MLOps practices.

# Dataset used :- LOAN PREDICTION PROBLEM DATASET (Kaggle)

# Team Roles :

ML Model + Integeration - Anjali kumari 

Frontend - Vaishnavi Pandey

Backend - Abhay 

# Problem Statement

Many machine learning projects focus only on training models. However, once the model is deployed, several problems can occur:

* The input data distribution changes over time

* Model predictions slowly become inaccurate

* Teams do not notice performance degradation quickly

* Retraining decisions are done manually

These issues can cause silent failures in production ML systems.

# Proposed Solution

This project builds a simple platform that:

* Logs model predictions and inputs

* Monitors production data

* Detects data drift

* Triggers retraining when necessary

* Provides a simple monitoring dashboard

This helps ensure that the ML model continues to perform well after deployment.

# Key Features

* Model Prediction API - 
A REST API that serves predictions using a trained machine learning model.

* Prediction Monitoring - 
Logs prediction requests, input data, and timestamps for analysis.

* Drift Detection - 
Compares training data with production data to detect changes in data distribution.

* Automated Retraining -
If significant drift is detected, the model can be retrained automatically.

* Monitoring Dashboard - 
Displays model activity and prediction data.

# System Architecture
User Request
      │
      ▼
Prediction API
      │
      ▼
Prediction Logger
      │
      ▼
Monitoring Data Storage
      │
      ▼
Drift Detection Engine
      │
 ┌────┴─────┐
 │          │
No Drift   Drift Detected
 │          │
 ▼          ▼
Continue  Retrain Model
              │
              ▼
        Deploy New Model
              │
              ▼
          Dashboard
          
# Technologies Used
* Programming

* Python

* Machine Learning(RandomForest)

* Scikit-learn

* Pandas

* NumPy

* Backend(Flask , Django etc)

* Monitoring - 
Evidently AI

Model Tracking

MLflow

# Project Structure
ml-monitoring-platform/

data/
    training dataset

model/
    train_model.py
    saved model file

api/
    predict.py

monitoring/
    logger.py
    prediction_logs.csv

drift_detection/
    detect_drift.py

dashboard/
    dashboard.py

requirements.txt
README.md

# How the System Works

* Train a machine learning model using historical data.

* Deploy the model through a prediction API.

* Every prediction request is logged.

* The system collects production data over time.

* Drift detection compares production data with training data.

* If drift is detected, the model can be retrained.

* Monitoring results are shown in a dashboard.

# Future Improvements - 

Some improvements that can be added in the future:

* Real-time data streaming

* Alert system for drift detection

* Advanced concept drift detection

* Containerization using Docker

* Full cloud deployment


