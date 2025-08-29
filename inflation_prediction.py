import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd 
import numpy as np 
import streamlit as st 
import xgboost as xgb

# Step 1: Load the data
my_data = r"/Users/d/Desktop/world_inflation_data/Sheet 1-API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_122376.csv"
df = pd.read_csv(my_data, skiprows=4)

# Step 2: Reshape (melt) the wide format into long format
df_long = df.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    var_name="Year",
    value_name="Inflation"
)

# Step 3: Clean the Year column
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")

# Step 4: Handle missing values → fill with median
df_long["Inflation"] = df_long["Inflation"].fillna(df_long["Inflation"].median())

# Step 5: Create lagged features
df_long = df_long.sort_values(["Country Name", "Year"])
df_long["inflation_lag1"] = df_long.groupby("Country Name")["Inflation"].shift(1)
df_long["inflation_lag4"] = df_long.groupby("Country Name")["Inflation"].shift(4)

# Fill lagged missing with median too
df_long["inflation_lag1"] = df_long["inflation_lag1"].fillna(df_long["inflation_lag1"].median())
df_long["inflation_lag4"] = df_long["inflation_lag4"].fillna(df_long["inflation_lag4"].median())

# Step 6: Define features and target
features = ["inflation_lag1", "inflation_lag4"]
X = df_long[features]
y = df_long["Inflation"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size = 0.25)

# Step 7: Train model
model = xgb.XGBRegressor(objective = "reg:squarederror")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Step 8: Visualisation
sns.lineplot(x="Year", y="Inflation", data=df_long, marker="o")
plt.title("Inflation Rate Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))  # show every 5 years
plt.xticks(rotation=45)
plt.show()

st.title("Inflation Forecasting Tool")

# Step 9 : Streamlit app.
# Country selection
countries = df_long["Country Name"].unique()
selected_country = st.sidebar.selectbox("Select Country", sorted(countries))

# Filter dataset
country_data = df_long[df_long["Country Name"] == selected_country].copy()

# User inputs (not yet integrated in model)
st.sidebar.header("Policy Inputs (for simulation only)")
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
policy_rate = st.sidebar.number_input("Monetary Policy Rate (%)", 0.00, 25.00, 10.00, step=0.01, format="%.2f")
exchange_rate = st.sidebar.number_input("Exchange Rate (Local/USD)", 3000, 6000, 3700)
oil_price = st.sidebar.number_input("Global Oil Price (USD/barrel)", 20, 200, 80)
food_index = st.sidebar.number_input("Food Price Index", 50, 200, 120)
m2_growth = st.sidebar.number_input("Money Supply Growth (%)", -10, 30, 10)

# Train model with caching to avoid retraining on every interaction.
@st.cache_resource
def train_model(X, y):
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    return model

# Run forecast
if st.button("Run Forecast"):
    if country_data.shape[0] < 5:
        st.warning("Not enough data to generate forecast for this country.")
    else:
        # Define features & target
        features = ["inflation_lag1", "inflation_lag4"]
        X = country_data[features]
        y = country_data["Inflation"]

        # Train only once (cached)
        model = train_model(X, y)

        # Predict using last available data
        latest_data = country_data.tail(1)
        X_future = latest_data[["inflation_lag1", "inflation_lag4"]]
        prediction = model.predict(X_future)[0]

        # Display one clean result
        st.metric("Predicted Inflation (Quarterly %)", f"{prediction:.2f}")

        # Show historical chart
        st.line_chart(country_data.set_index("Year")["Inflation"])




