import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
    df['Property_Type'][i] = df['Property_Type'][i].upper()

df1 = df.copy()
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
# st.line_chart(corr_plot)
st.write('not bad')