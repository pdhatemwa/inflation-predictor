# 📈 Inflation Predictor

A data science project focused on predicting inflation rates using historical data from the World Bank. This repository contains code and documentation for the full machine learning pipeline, from data acquisition and preprocessing to model deployment via Streamlit.

---

## 🔍 Project Overview

Inflation is a critical economic indicator that affects purchasing power, policy decisions, and overall financial stability. This project aims to build a predictive model that estimates future inflation figures based on macroeconomic data sourced from the [World Bank Open Data](https://data.worldbank.org/).

Using robust data science techniques—including **feature engineering**, **exploratory data analysis**, **model selection**, and **evaluation**—this project seeks to deliver an interpretable and accurate inflation forecasting tool.

---

## 🧠 Model Choice

The primary model used in this project is **XGBoost Regressor**. XGBoost is well-suited for this problem due to:

* Its ability to handle non-linear relationships
* Robustness to overfitting
* High accuracy and interpretability through feature importance
* Speed and scalability for large datasets

Other models (e.g., Random Forests, Linear Regression, LSTM) may be tested for comparison.

---

## 🔧 Features & Workflow

### ✅ Data Source:

* World Bank macroeconomic indicators (GDP, interest rates, exchange rates, CPI, etc.)

### 🔬 Key Processes:

1. **Data Collection** – Fetching datasets from the World Bank API or CSV files.
2. **Data Cleaning** – Handling missing values, formatting inconsistencies, and outliers.
3. **Exploratory Data Analysis** – Understanding distributions, trends, and correlations.
4. **Feature Engineering** – Generating relevant features such as lag variables, rolling averages, and economic ratios.
5. **Model Training** – Training and tuning the XGBoost model using scikit-learn pipelines.
6. **Evaluation** – Measuring performance using RMSE, MAE, and R².
7. **Visualization** – Presenting insights and predictions in a user-friendly format.
8. **Deployment** – Deploying the predictive model using Streamlit.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/inflation-predictor.git
cd inflation-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```
inflation-predictor/
│
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── models/                # Saved models and evaluation metrics
├── app.py                 # Streamlit app
├── utils/                 # Helper functions for preprocessing and visualization
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📈 Sample Output

The web app allows users to:

* View key economic indicators
* Predict upcoming inflation figures
* Explore feature importance and model insights

---

## 🛠 Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
* **Streamlit** for deployment
* **World Bank API** for data
* **Matplotlib & Seaborn** for visualization

---

## 📌 Future Work

* Incorporate more regional data
* Test deep learning models like LSTM for time-series forecasting
* Enable user-uploaded data and real-time updates

---

## 📬 Contact

**Patrick Dhatemwa**
*Data Science & Statistics Student | Strathmore University*
📧 [patrick.dhatemwa@example.com](mailto:patrickdhatemwa7@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/patrick-dhatemwa-64737b223/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3B9dW79KHGSh2In5qfdalMPA%3D%3D)
🔗 [GitHub](https://github.com/pdhatemwa)

