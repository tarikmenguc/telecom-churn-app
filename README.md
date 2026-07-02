# 🔮 Telecom Churn Predictor: AI-Powered Customer Retention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telecom-churn-appp.streamlit.app/)

Welcome to the **Telecom Churn Predictor**! This repository contains a production-ready Machine Learning web application designed to help telecommunication businesses identify customers who are at high risk of leaving (churning). 

**🚀 LIVE DEMO:** [Click here to try the app](https://telecom-churn-appp.streamlit.app/)

By predicting churn *before* it happens, businesses can take proactive measures—such as offering tailored discounts or loyalty programs—to retain valuable customers and protect recurring revenue.

---

## 🎯 Business Value

Acquiring a new customer is often 5-25 times more expensive than retaining an existing one. This tool empowers Customer Success and Retention teams to:
1. **Identify At-Risk Accounts:** Instantly calculate churn probability based on historical profiles.
2. **Quantify Financial Impact:** Automatically estimate the potential annual revenue loss if the customer leaves.
3. **Take Actionable Steps:** The system recommends precise, immediate actions (e.g., immediate outreach vs. standard loyalty track) based on the calculated risk threshold.

## ✨ Key Features

- **Interactive Dashboard**: A sleek, user-friendly interface built with Streamlit, requiring zero technical knowledge from end-users (e.g., sales agents or managers).
- **Real-Time ML Inference**: Calculates churn probabilities on the fly using a robust, pre-trained Scikit-Learn classification model (Random Forest / XGBoost).
- **Dynamic Visualizations**: Features a Plotly-powered "Churn Risk Meter" to provide an immediate, visual understanding of customer health.
- **Comprehensive Profiling**: Accepts inputs across three key pillars: Customer Identity, Services Subscribed, and Financial/Contractual Details.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-learn, Imbalanced-learn (SMOTE)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have Python installed (3.8 or higher recommended). 

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/tarikmenguc/telecom-churn-app.git
   cd TelecomChurnApp
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📁 Project Architecture

- `app.py`: The core application script. Handles the UI layout, state management, and real-time model inference.
- `models/`: Directory housing the serialized, pre-trained assets:
  - `churn_model.pkl`: The core predictive model.
  - `scaler.pkl`: The data scaler used to normalize continuous features during inference.
  - `columns.pkl`: The exact feature layout expected by the model to prevent data misalignment.
- `requirements.txt`: Environment dependencies.

## 🤝 Let's Connect

This project showcases my ability to bridge the gap between **Data Science** and **Business Value** by deploying predictive models into beautifully designed, interactive web applications. 

If you are looking for a developer to build end-to-end data products, automated pipelines, or deploy ML models, feel free to reach out to me on Upwork!
