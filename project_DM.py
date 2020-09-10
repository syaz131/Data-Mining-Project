import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import descartes
import geopandas as gpd

from matplotlib.collections import PatchCollection

import yfinance as yf
import streamlit as st

print('3')

st.write(
    # Simple header
    'Shown are stock'
)
df = pd.read_csv('Bank_CS.csv')

# Change data

# state = df['State']
df = pd.read_csv('Bank_CS.csv')
df['State'] = df['State'].replace("Johor B", "Johor")
df['State'] = df['State'].replace("SWK", "Sarawak")
df['State'] = df['State'].replace("N.Sembilan", "Negeri Sembilan")
df['State'] = df['State'].replace("N.S", "Negeri Sembilan")
df['State'] = df['State'].replace("Trengganu", "Terengganu")
df['State'] = df['State'].replace("K.L", "Kuala Lumpur")
df['State'] = df['State'].replace("P.Pinang", "Penang")
df['State'] = df['State'].replace("Pulau Penang", "Penang")

df['State'] = df['State'].replace("\s", "", regex=True)  # remove white spaces
df['State'] = df['State'].replace("[^a-zA-Z]", "", regex=True)  # remove symbol

# print(df['State'].head())

# use uppercase  is better
# df.shape(0)



for i in range(0, df.shape[0]):
    df['State'][i] = df['State'][i].upper()
    df['Decision'][i] = df['Decision'][i].upper()
    df['More_Than_One_Products'][i] = df['More_Than_One_Products'][i].upper()
    df['Employment_Type'][i] = df['Employment_Type'][i].upper()
    if (type(df['Property_Type'][i]) != float):
        df['Property_Type'][i] = df['Property_Type'][i].upper()

df1 = df.copy()
# fill property type
df1['Property_Type'] = df1['Property_Type'].ffill(axis = 0)

# year
df1.Loan_Tenure_Year = df1.Loan_Tenure_Year.fillna(df1.Loan_Tenure_Year.median())
df1.Years_to_Financial_Freedom = df1.Years_to_Financial_Freedom.fillna(df1.Loan_Tenure_Year.median())
df1.Years_for_Property_to_Completion = df1.Years_for_Property_to_Completion.fillna(df1.Loan_Tenure_Year.median())

df1.Loan_Tenure_Year = df1.Loan_Tenure_Year.astype(int)
df1.Years_to_Financial_Freedom = df1.Years_to_Financial_Freedom.astype(int)
df1.Years_for_Property_to_Completion = df1.Years_for_Property_to_Completion.astype(int)

# number
df1.Number_of_Credit_Card_Facility = df1.Number_of_Credit_Card_Facility.fillna(
    df1.Number_of_Credit_Card_Facility.median())
df1.Number_of_Properties = df1.Number_of_Properties.fillna(df1.Number_of_Properties.median())
df1.Number_of_Bank_Products = df1.Number_of_Bank_Products.fillna(df1.Number_of_Bank_Products.median())
df1.Number_of_Side_Income = df1.Number_of_Side_Income.fillna(df1.Number_of_Side_Income.median())

# salary & total
df1.Loan_Amount = df1.Loan_Amount.fillna(df1.Loan_Amount.median())
df1.Monthly_Salary = df1.Monthly_Salary.fillna(df1.Monthly_Salary.median())
df1.Total_Income_for_Join_Application = df1.Total_Income_for_Join_Application.fillna(
    df1.Total_Income_for_Join_Application.median())
df1.Total_Sum_of_Loan = df1.Total_Sum_of_Loan.fillna(df1.Total_Sum_of_Loan.median())

corr_matrix = df.corr().abs()

st.write('Correlation Heatmap')
# your codes here...
plt.figure(figsize=(40, 40))
corr_plot = sns.heatmap(corr_matrix, vmax=0.8, square=True, fmt='.3f', annot=True,
                        annot_kws={'size': 18}, cmap=sns.color_palette('Blues'))
st.pyplot()

st.write('Frequency of each Loan Decision  and  Percentage of each Loan Decision')
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='Decision',data=df1,ax=axs[0])
axs[0].set_title("Frequency of each Loan Decision")
df.Decision.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan Decision")
st.pyplot()

st.write('No Name')
df['Employment_Type'].value_counts().plot(kind='bar')
st.pyplot()

df['Property_Type'].value_counts().plot(kind='bar')
st.pyplot()

sns.catplot(x="Credit_Card_Exceed_Months", hue="Decision", kind="count", data=df)
st.pyplot()

sns.catplot("Decision", col="Employment_Type", col_wrap=5,
            data=df, kind="count", height=10, aspect=.3)
st.pyplot()

st.write('Box Plot')
sns.boxplot(x='Employment_Type', y='Loan_Amount', data=df)
st.pyplot()

#HISTOGRAM OF LOAN AMOUNT
st.write('HISTOGRAM OF LOAN AMOUNT ')
data = df1["Loan_Amount"]
plt.hist(data, bins=[100000,200000,300000,400000,500000,600000,700000,800000])
st.pyplot()


sns.catplot(x="Number_of_Properties", hue="Decision", kind="count", data=df1)
st.pyplot()


# Map Malaysia
df_gbp = df1[["State", "Total_Sum_of_Loan"]]
gbp = df_gbp.groupby(["State"],as_index=False).median()

fp = "./map Malaysia/Malaysia_Polygon.shp"
map_df = gpd.read_file(fp)
map_df['name'] = map_df['name'].str.upper()
map_df["name"]= map_df["name"].replace("KUALA LUMPUR", "KUALALUMPUR")
map_df["name"]= map_df["name"].replace("NEGERI SEMBILAN", "NEGERISEMBILAN")
# check data type so we can see that this is not a normal dataframe, but a GEOdataframe
#map_df.plot()
merged = map_df.set_index('name').join(gbp.set_index('State'))
# set a variable that will call whatever column we want to visualise on the map
variable = "Total_Sum_of_Loan"
# set the range for the choropleth
vmin, vmax = 0, 33000
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axis('off')

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

merged.plot(column=variable, cmap='Greens', linewidth=0.5, ax=ax, edgecolor='0')
st.pyplot()