import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd 
import numpy as np 
import streamlit as st 

# Loading the data
my_data = r"/Users/d/Desktop/Uganda_inflation/Sheet 1-Table 1.csv"
my_df = pd.read_csv(my_data, index_col = 0).T
my_df.columns = ["Consumer inflation rate"]
print(my_df.head(10))

# Visualising the data.
plt.figure(figsize=(10,6))
plt.plot(my_df.index, my_df["Consumer inflation rate"], marker="o", linestyle="-")
plt.title("Uganda Consumer Inflation Rate Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.grid(True)
plt.show()

# Visuales with seaborn
sns.lineplot(x=my_df.index, y=my_df["Consumer inflation rate"], marker="o")
plt.title("Uganda Consumer Inflation Rate Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Inflation Rate (%)")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))  # show every 5 years
plt.xticks(rotation=45)
plt.show()