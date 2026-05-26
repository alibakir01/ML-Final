# Airbnb Booking Prediction — Kaggle K353

Final Project for **COMP 468 — Introduction to Machine Learning** (Abdullah Gül University, Dr. Khaled Hejja).

## Problem
Predict the **total number of booked nights in Q3 2016** for NYC Airbnb properties (regression task).
Evaluation: **MSE** on hidden test set.

Competition: <https://www.kaggle.com/competitions/a-cloned-airbnb-booking-prediction-competition-k-353>

## Repository Structure
```
Airbnb-K353/
├── data/                  # raw CSVs (gitignored)
│   ├── property_info.csv
│   ├── listing_2016Q1.csv
│   ├── listing_2016Q2.csv
│   ├── reserve_2016Q3_train.csv
│   └── PropertyID_test.csv
├── notebooks/
│   └── 01_EDA.ipynb       # Exploratory Data Analysis
├── outputs/               # plots, submissions
└── README.md
```

## Allowed Models (COMP 468 only)
DecisionTree, Bagging, RandomForest, GradientBoosting, XGBoost, MLP (PyTorch + Adam/SGD/RMSprop/AdaGrad).

## Acknowledgment
This work was conducted as part of COMP 468 at **Abdullah Gül University (AGU)**.
