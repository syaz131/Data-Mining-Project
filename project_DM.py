import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import geopandas as gpd
import descartes
import streamlit as st

from imblearn.over_sampling import SMOTE
from boruta import BorutaPy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from yellowbrick.cluster import silhouette_visualizer

from apyori import apriori

# ============================ import libr
df = pd.read_csv('Bank_CS.csv')
df_eda = df.copy()

# ================= Data Cleaning ======================
df_eda['State'] = df_eda['State'].replace("Johor B", "Johor")
df_eda['State'] = df_eda['State'].replace("SWK", "Sarawak")
df_eda['State'] = df_eda['State'].replace("N.Sembilan", "Negeri Sembilan")
df_eda['State'] = df_eda['State'].replace("N.S", "Negeri Sembilan")
df_eda['State'] = df_eda['State'].replace("Trengganu", "Terengganu")
df_eda['State'] = df_eda['State'].replace("K.L", "Kuala Lumpur")
df_eda['State'] = df_eda['State'].replace("P.Pinang", "Penang")
df_eda['State'] = df_eda['State'].replace("Pulau Penang", "Penang")

df_eda['State'] = df_eda['State'].replace("\s", "", regex=True)  # remove white spaces
df_eda['State'] = df_eda['State'].replace("[^a-zA-Z]", "", regex=True)  # remove symbol

for i in range(0, df_eda.shape[0]):
    df_eda['State'][i] = df_eda['State'][i].upper()
    df_eda['Decision'][i] = df_eda['Decision'][i].upper()
    df_eda['More_Than_One_Products'][i] = df_eda['More_Than_One_Products'][i].upper()
    df_eda['Employment_Type'][i] = df_eda['Employment_Type'][i].upper()
    if (type(df_eda['Property_Type'][i]) != float):
        df_eda['Property_Type'][i] = df_eda['Property_Type'][i].upper()
    if (type(df_eda['Credit_Card_types'][i]) != float):
        df_eda['Credit_Card_types'][i] = df_eda['Credit_Card_types'][i].upper()

# ================= Fill Nan Data ======================
df_eda2 = df_eda.copy()

# fill property type
df_eda['Property_Type'] = df_eda['Property_Type'].ffill(axis=0)
df_eda = df_eda.dropna(subset=['Credit_Card_types'])

# year
df_eda.Loan_Tenure_Year = df_eda.Loan_Tenure_Year.fillna(df_eda.Loan_Tenure_Year.median())
df_eda.Years_to_Financial_Freedom = df_eda.Years_to_Financial_Freedom.fillna(df_eda.Loan_Tenure_Year.median())
df_eda.Years_for_Property_to_Completion = df_eda.Years_for_Property_to_Completion.fillna(
    df_eda.Loan_Tenure_Year.median())

df_eda.Loan_Tenure_Year = df_eda.Loan_Tenure_Year.astype(int)
df_eda.Years_to_Financial_Freedom = df_eda.Years_to_Financial_Freedom.astype(int)
df_eda.Years_for_Property_to_Completion = df_eda.Years_for_Property_to_Completion.astype(int)

# number
df_eda.Number_of_Credit_Card_Facility = df_eda.Number_of_Credit_Card_Facility.fillna(
    df_eda.Number_of_Credit_Card_Facility.median())
df_eda.Number_of_Properties = df_eda.Number_of_Properties.fillna(df_eda.Number_of_Properties.median())
df_eda.Number_of_Bank_Products = df_eda.Number_of_Bank_Products.fillna(df_eda.Number_of_Bank_Products.median())
df_eda.Number_of_Side_Income = df_eda.Number_of_Side_Income.fillna(df_eda.Number_of_Side_Income.median())

# salary & total
df_eda.Loan_Amount = df_eda.Loan_Amount.fillna(df_eda.Loan_Amount.mean())
df_eda.Monthly_Salary = df_eda.Monthly_Salary.fillna(df_eda.Monthly_Salary.mean())
df_eda.Total_Income_for_Join_Application = df_eda.Total_Income_for_Join_Application.fillna(
    df_eda.Total_Income_for_Join_Application.mean())
df_eda.Total_Sum_of_Loan = df_eda.Total_Sum_of_Loan.fillna(df_eda.Total_Sum_of_Loan.mean())

# reset index
df_eda = df_eda.reset_index(drop=True)
df_eda = df_eda.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

# ================================= New Input
# new_Total_Income_for_Join_Application = 0
# new_Monthly_Salary = 0
# new_Loan_Amount = 0
# new_Credit_Card_Exceed_Months = 0
#
# new_Employment_Type = ''
# new_Property_Type = ''
# new_More_Than_One_Products = ''

select_yes_no = [np.nan, 'YES', 'NO']
select_Employment_Type = [np.nan, 'EMPLOYER', 'SELF_EMPLOYED', 'GOVERNMENT', 'EMPLOYEE', 'FRESH_GRADUATE']
select_Property_Type = [np.nan, 'CONDOMINIUM', 'BUNGALOW', 'TERRACE', 'FLAT']
select_State = [np.nan, 'JOHOR', 'SELANGOR', 'KUALALUMPUR', 'PENANG', 'NEGERISEMBILAN', 'SARAWAK', 'SABAH',
                'TERENGGANU', 'KEDAH']
select_CC_type = [np.nan, 'PLATINUM', 'NORMAL', 'GOLD']

# ============================ Streamlit UI

menu = ['Clustering', 'Classification', 'EDA', 'ARM']
st.sidebar.subheader('Menu')
choice = st.sidebar.selectbox("", menu)

if choice == 'Classification' or choice == 'Clustering' or choice == 'ARM':
    st.sidebar.write('****************')
    st.sidebar.subheader('Input Form')

    new_Credit_Card_Exceed_Months = st.sidebar.number_input('Credit Card Exceed Months')
    new_Employment_Type = st.sidebar.selectbox('Employment Type', select_Employment_Type)
    new_Loan_Amount = st.sidebar.number_input('Loan Amount')
    new_Loan_Tenure_Year = st.sidebar.number_input('Loan Tenure Year')

    new_More_Than_One_Products = st.sidebar.selectbox('More Than One Products', select_yes_no)
    new_Credit_Card_types = st.sidebar.selectbox('Credit Card types', select_CC_type)
    new_Number_of_Dependents = st.sidebar.number_input('Number of Dependents')
    new_Years_to_Financial_Freedom = st.sidebar.number_input('Years to Financial Freedom')

    new_Number_of_Credit_Card_Facility = st.sidebar.number_input('Number of Credit Card Facility')
    new_Number_of_Properties = st.sidebar.number_input('Number of Properties')
    new_Number_of_Bank_Products = st.sidebar.number_input('Number of Bank Products')
    new_Number_of_Loan_to_Approve = st.sidebar.number_input('Number of Loan to Approve')

    new_Property_Type = st.sidebar.selectbox('Property Type', select_Property_Type)
    new_Years_for_Property_to_Completion = st.sidebar.number_input('Years for Property to Completion')
    new_State = st.sidebar.selectbox('State', select_State)

    new_Number_of_Side_Income = st.sidebar.number_input('Number of Side Income')
    new_Monthly_Salary = st.sidebar.number_input('Monthly Salary')
    new_Total_Sum_of_Loan = st.sidebar.number_input('Total Sum of Loan')
    new_Total_Income_for_Join_Application = st.sidebar.number_input('Total Income for Join Application')
    new_Score = st.sidebar.number_input('Score')

    new_data_object = [new_Employment_Type, new_More_Than_One_Products, new_Credit_Card_types,
                       new_Property_Type, new_State]

    new_data_int = [new_Credit_Card_Exceed_Months, new_Loan_Tenure_Year, new_Number_of_Dependents,
                    new_Years_to_Financial_Freedom, new_Number_of_Loan_to_Approve,
                    new_Years_for_Property_to_Completion, new_Score]

    new_data_float = [new_Loan_Amount, new_Loan_Tenure_Year, new_Years_to_Financial_Freedom,
                      new_Number_of_Credit_Card_Facility, new_Number_of_Properties,
                      new_Years_for_Property_to_Completion, new_Number_of_Side_Income, new_Monthly_Salary,
                      new_Total_Sum_of_Loan, new_Total_Income_for_Join_Application]

    isna_data_list_object = False
    isna_data_list_int = False
    isna_data_list_float = False

    if np.nan in new_data_object:
        isna_data_list_object = True

    for x in new_data_int:
        if x < 0:
            isna_data_list_int = True
            break

    # if not isna_data_list_int:
    # [int(i) for i in new_data_int]

    for x in new_data_float:
        if x < 0:
            isna_data_list_float = True
            break

    isna_data_list = isna_data_list_float == False and isna_data_list_int == False and isna_data_list_object == False

    new_series = df_eda.copy()
    new_series = new_series.drop(new_series.index[2:])

    if st.sidebar.button('Update Input Data') and isna_data_list == True:
        new_series.iloc[0, 0] = int(new_Credit_Card_Exceed_Months)
        new_series.iloc[0, 1] = new_Employment_Type
        new_series.iloc[0, 2] = new_Loan_Amount
        new_series.iloc[0, 3] = int(new_Loan_Tenure_Year)
        new_series.iloc[0, 4] = new_More_Than_One_Products
        new_series.iloc[0, 5] = new_Credit_Card_types
        new_series.iloc[0, 6] = int(new_Number_of_Dependents)
        new_series.iloc[0, 7] = int(new_Years_to_Financial_Freedom)
        new_series.iloc[0, 8] = new_Number_of_Credit_Card_Facility
        new_series.iloc[0, 9] = new_Number_of_Properties
        new_series.iloc[0, 10] = new_Number_of_Bank_Products
        new_series.iloc[0, 11] = int(new_Number_of_Loan_to_Approve)
        new_series.iloc[0, 12] = new_Property_Type
        new_series.iloc[0, 13] = int(new_Years_for_Property_to_Completion)
        new_series.iloc[0, 14] = new_State
        new_series.iloc[0, 15] = new_Number_of_Side_Income
        new_series.iloc[0, 16] = new_Monthly_Salary
        new_series.iloc[0, 17] = new_Total_Sum_of_Loan
        new_series.iloc[0, 18] = new_Total_Income_for_Join_Application
        new_series.iloc[0, 20] = int(new_Score)

        new_series = new_series.drop(["Decision"], axis=1)
        st.sidebar.success('Input Data Updated')
    else:
        st.sidebar.warning('Input Data Not Updated')
        st.sidebar.write('Insert non NaN value and Positive value to Update Input Data')

if choice == 'Classification':

    st.title('Classification')

    # ================= Normalization ======================
    # df_normz = df_eda.copy()
    #
    # features_to_scale = ["Loan_Amount", "Monthly_Salary", "Total_Income_for_Join_Application", "Total_Sum_of_Loan"]
    # to_scale = df_normz[features_to_scale]
    #
    # min_max_scaler = StandardScaler()
    # x_scaled = min_max_scaler.fit_transform(to_scale)
    # df_x_scaled = pd.DataFrame(x_scaled, columns=features_to_scale)
    # # df_x_scaled
    #
    # df_normz = df_normz.drop(features_to_scale, axis=1)
    # df_normz = pd.concat([df_normz, df_x_scaled], sort=True, axis=1)

    # ================= Label Encoding ======================
    df_le = df_eda.copy()

    df_le['More_Than_One_Products'] = LabelEncoder().fit_transform(df_le.More_Than_One_Products)
    df_le['Employment_Type'] = LabelEncoder().fit_transform(df_le.Employment_Type)
    df_le['Property_Type'] = LabelEncoder().fit_transform(df_le.Property_Type)
    df_le['State'] = LabelEncoder().fit_transform(df_le.State)
    df_le['Decision'] = LabelEncoder().fit_transform(df_le.Decision)
    df_le['Credit_Card_types'] = LabelEncoder().fit_transform(df_le.Credit_Card_types)

    # ================= SMOTENC ======================
    y = df_le.Decision
    X = df_le.drop(["Decision"], axis=1)

    os = SMOTE(random_state=0)
    columns = X.columns
    os_data_X, os_data_y = os.fit_sample(X, y)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['Decision'])

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.countplot(x='Decision', data=os_data_y, ax=axs[0])
    axs[0].set_title("Frequency of each Loan Decision")
    os_data_y.Decision.value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
    axs[1].set_title("Percentage of each Loan Decision")

    # =========================== Classification ===============================
    st.header('Classification')

    X_train_os, X_test_os, y_train_os, y_test_os = train_test_split(os_data_X, os_data_y.values.ravel(), test_size=0.30,
                                                                    random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.30, random_state=10)

    # ====== Naive Bayes ================
    st.header('Naive Bayes - Imbalanced Dataset')
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    # Model Accuracy

    st.write("Accuracy on test set: {:.3f}".format(nb.score(X_test, y_test)))
    prob_nb = nb.predict_proba(X_test)
    prob_nb = prob_nb[:, 1]

    auc_nb = roc_auc_score(y_test, prob_nb)
    st.write('AUC: %.2f' % auc_nb)
    confusion_majority = confusion_matrix(y_test, y_pred)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'.format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test, y_pred)))
    st.write('**********************')

    st.header('Naive Bayes - Oversampled Dataset')
    nb_os = GaussianNB()
    nb_os.fit(X_train_os, y_train_os)

    y_pred_os = nb_os.predict(X_test_os)

    # Model Accuracy

    st.write("Accuracy on test set: {:.3f}".format(nb_os.score(X_test_os, y_test_os)))
    prob_nb_os = nb_os.predict_proba(X_test_os)
    prob_nb_os = prob_nb_os[:, 1]

    auc_nb_os = roc_auc_score(y_test_os, prob_nb_os)
    st.write('AUC: %.2f' % auc_nb_os)
    confusion_majority = confusion_matrix(y_test_os, y_pred_os)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test_os, y_pred_os)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test_os, y_pred_os)))
    st.write('F1= {:.2f}'.format(f1_score(y_test_os, y_pred_os)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test_os, y_pred_os)))
    st.write('**********************')

    # ====== Random Forest ================
    st.header('Random Forest - Imbalance Dataset')

    clf = RandomForestClassifier(random_state=10)
    # clf.fit(os_data_X, os_data_y.values.ravel())
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Model Accuracy

    st.write("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
    prob_clf = clf.predict_proba(X_test)
    prob_clf = prob_clf[:, 1]

    auc_clf = roc_auc_score(y_test, prob_clf)
    st.write('AUC: %.2f' % auc_clf)
    confusion_majority = confusion_matrix(y_test, y_pred)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'.format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test, y_pred)))
    st.write('**********************')

    st.header('Random Forest - Oversampled Dataset')
    clf_os = RandomForestClassifier(random_state=10)
    # clf.fit(os_data_X, os_data_y.values.ravel())
    clf_os.fit(X_train_os, y_train_os)

    y_pred_os = clf_os.predict(X_test_os)

    # Model Accuracy

    st.write("Accuracy on test set: {:.3f}".format(clf_os.score(X_test_os, y_test_os)))
    prob_clf_os = clf_os.predict_proba(X_test_os)
    prob_clf_os = prob_clf_os[:, 1]

    auc_clf_os = roc_auc_score(y_test_os, prob_clf_os)
    st.write('AUC: %.2f' % auc_clf_os)
    confusion_majority = confusion_matrix(y_test_os, y_pred_os)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test_os, y_pred_os)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test_os, y_pred_os)))
    st.write('F1= {:.2f}'.format(f1_score(y_test_os, y_pred_os)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test_os, y_pred_os)))
    st.write('**********************')

    # ====== KNN ================
    st.header('K Nearest Neighbour - Imbalance Dataset')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    st.write("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))
    prob_knn = knn.predict_proba(X_test)
    prob_knn = prob_knn[:, 1]

    auc_model = roc_auc_score(y_test, prob_knn)
    st.write('AUC: %.2f' % auc_model)

    confusion_majority = confusion_matrix(y_test, y_pred)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'.format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test, y_pred)))
    st.write('**********************')

    st.header('K Nearest Neighbour - Oversampled Dataset')
    knn_os = KNeighborsClassifier(n_neighbors=5)
    knn_os.fit(X_train_os, y_train_os)

    y_pred_os = knn_os.predict(X_test_os)
    st.write("Accuracy on test set: {:.3f}".format(knn_os.score(X_test_os, y_test_os)))
    prob_knn_os = knn_os.predict_proba(X_test_os)
    prob_knn_os = prob_knn_os[:, 1]

    auc_model_os = roc_auc_score(y_test_os, prob_knn_os)
    st.write('AUC: %.2f' % auc_model_os)

    confusion_majority = confusion_matrix(y_test_os, y_pred_os)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test_os, y_pred_os)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test_os, y_pred_os)))
    st.write('F1= {:.2f}'.format(f1_score(y_test_os, y_pred_os)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test_os, y_pred_os)))
    st.write('**********************')

    # ====== SVM ================
    st.header('Support Vector Machine - Imbalance Dataset')
    st.write('Here is SVM')
    # ====== SVM ================

    # ====== XGB TREE ================
    st.header('XGBoost TREE - Imbalance Dataset')
    xg_tree = xgb.XGBClassifier(objective='reg:logistic', colsample_bytree=0.35, learning_rate=0.25, max_depth=12,
                                alpha=15,
                                n_estimators=15, booster='gbtree')
    xg_tree.fit(X_train, y_train)
    y_pred = xg_tree.predict(X_test)

    # Model Accuracy

    st.write("Accuracy on test set: {:.3f}".format(xg_tree.score(X_test, y_test)))
    prob_Xgtree = xg_tree.predict_proba(X_test)
    prob_Xgtree = prob_Xgtree[:, 1]

    auc_model = roc_auc_score(y_test, prob_Xgtree)
    st.write('AUC: %.2f' % auc_model)

    confusion_majority = confusion_matrix(y_test, y_pred)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test, y_pred)))
    st.write('F1= {:.2f}'.format(f1_score(y_test, y_pred)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test, y_pred)))
    st.write('**********************')

    st.header('XGBoost TREE - Oversampled Dataset')
    xg_tree_os = xgb.XGBClassifier(objective='reg:logistic', colsample_bytree=0.35, learning_rate=0.25,
                                   max_depth=12, alpha=15, n_estimators=15, booster='gbtree')
    xg_tree_os.fit(X_train_os, y_train_os)

    y_pred_os = xg_tree_os.predict(X_test_os)

    # Model Accuracy

    st.write("Accuracy on test set: {:.3f}".format(xg_tree_os.score(X_test_os, y_test_os)))
    prob_Xgtree_os = xg_tree_os.predict_proba(X_test_os)
    prob_Xgtree_os = prob_Xgtree_os[:, 1]

    auc_model_os = roc_auc_score(y_test_os, prob_Xgtree_os)
    st.write('AUC: %.2f' % auc_model_os)

    confusion_majority = confusion_matrix(y_test_os, y_pred_os)

    st.write('**********************')
    st.write('Mjority TN= ', confusion_majority[0][0])
    st.write('Mjority FP=', confusion_majority[0][1])
    st.write('Mjority FN= ', confusion_majority[1][0])
    st.write('Mjority TP= ', confusion_majority[1][1])
    st.write('**********************')

    st.write('Precision= {:.2f}'.format(precision_score(y_test_os, y_pred_os)))
    st.write('Recall= {:.2f}'.format(recall_score(y_test_os, y_pred_os)))
    st.write('F1= {:.2f}'.format(f1_score(y_test_os, y_pred_os)))
    st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test_os, y_pred_os)))
    st.write('**********************')

    # ====== Graph ROC ================
    st.header('Graph ROC - Imbalance Dataset')

    fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_nb)
    fpr_clf, tpr_clf, thresholds_clf = roc_curve(y_test, prob_clf)
    # fpr_model, tpr_model, thresholds_model = roc_curve(y_test, prob_model)
    fpr_XGtree, tpr_XGtree, thresholds_XGtree = roc_curve(y_test, prob_Xgtree)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, prob_knn)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(fpr_NB, tpr_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(fpr_clf, tpr_clf, color='blue', label='RF', linewidth=linewidth)
    # plt.plot(fpr_model, tpr_model, color='red', label='SVM', linewidth=linewidth)
    plt.plot(fpr_XGtree, tpr_XGtree, color='purple', label='XG Tree', linewidth=linewidth)
    plt.plot(fpr_knn, tpr_knn, color='green', label='KNN', linewidth=linewidth)

    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()

    st.header('Graph ROC - Oversampled Dataset')

    fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test_os, prob_nb_os)
    fpr_clf, tpr_clf, thresholds_clf = roc_curve(y_test_os, prob_clf_os)
    # fpr_model, tpr_model, thresholds_model = roc_curve(y_test_os, prob_model_os)
    fpr_XGtree, tpr_XGtree, thresholds_XGtree = roc_curve(y_test_os, prob_Xgtree_os)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test_os, prob_knn_os)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(fpr_NB, tpr_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(fpr_clf, tpr_clf, color='blue', label='RF', linewidth=linewidth)
    # plt.plot(fpr_model, tpr_model, color='red', label='SVM', linewidth=linewidth)
    plt.plot(fpr_XGtree, tpr_XGtree, color='purple', label='XG Tree', linewidth=linewidth)
    plt.plot(fpr_knn, tpr_knn, color='green', label='KNN', linewidth=linewidth)

    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()

    # ====== Precision - Recall ================
    st.header('Graph Precision-Recall - Imbalance Dataset')
    prec_NB, rec_NB, thresholds_NB = precision_recall_curve(y_test, prob_nb)
    prec_clf, rec_clf, thresholds_clf = precision_recall_curve(y_test, prob_clf)
    # prec_model, rec_model, thresholds_model = precision_recall_curve(y_test, prob_model)
    prec_knn, rec_knn, thresholds_knn = precision_recall_curve(y_test, prob_knn)
    prec_XGtree, rec_XGtree, thresholds_XGtree = precision_recall_curve(y_test, prob_Xgtree)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(prec_NB, rec_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(prec_clf, rec_clf, color='blue', label='RF', linewidth=linewidth)
    plt.plot(prec_knn, rec_knn, color='green', label='KNN', linewidth=linewidth)
    # plt.plot(prec_model, rec_model, color='red', label='SVM', linewidth=linewidth)
    plt.plot(prec_XGtree, rec_XGtree, color='purple', label='XG Tree', linewidth=linewidth)

    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()

    st.header('Graph Precision-Recall - Oversampled Dataset')
    prec_NB, rec_NB, thresholds_NB = precision_recall_curve(y_test_os, prob_nb_os)
    prec_clf, rec_clf, thresholds_clf = precision_recall_curve(y_test_os, prob_clf_os)
    # prec_model, rec_model, thresholds_model = precision_recall_curve(y_test_os, prob_model_os)
    prec_knn, rec_knn, thresholds_knn = precision_recall_curve(y_test_os, prob_knn_os)
    prec_XGtree, rec_XGtree, thresholds_XGtree = precision_recall_curve(y_test_os, prob_Xgtree_os)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(prec_NB, rec_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(prec_clf, rec_clf, color='blue', label='RF', linewidth=linewidth)
    plt.plot(prec_knn, rec_knn, color='green', label='KNN', linewidth=linewidth)
    # plt.plot(prec_model, rec_model, color='red', label='SVM', linewidth=linewidth)
    plt.plot(prec_XGtree, rec_XGtree, color='purple', label='XG Tree', linewidth=linewidth)

    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()

    # =================== Predict Input ============
    st.header('Predict Input - Random Forest Classification')

    st.write('Prediction of your input:')
    st.write(new_series.iloc[0])

    # place to translate
    Temp_dict = {'EMPLOYEE': 0, 'EMPLOYER': 1, 'FRESH_GRADUATE': 2, 'GOVERNMENT': 3, 'SELF_EMPLOYED': 4}
    new_series['Employment_Type'] = new_series.Employment_Type.map(Temp_dict)
    Temp_dict = {'NO': 0, 'YES': 1}
    new_series['More_Than_One_Products'] = new_series.More_Than_One_Products.map(Temp_dict)
    Temp_dict = {'BUNGALOW': 0, 'CONDOMINIUM': 1, 'FLAT': 2, 'TERRACE': 3}
    new_series['Property_Type'] = new_series.Property_Type.map(Temp_dict)
    Temp_dict = {'JOHOR': 0, 'KEDAH': 1, 'KUALALUMPUR': 2, 'NEGERISEMBILAN': 3, 'PENANG': 4, 'SABAH': 5, 'SARAWAK': 6,
                 'SELANGOR': 7, 'TERENGGANU': 8}
    new_series['State'] = new_series.State.map(Temp_dict)
    Temp_dict = {'GOLD': 0, 'NORMAL': 1, 'PLATINUM': 2}
    new_series['Credit_Card_types'] = new_series.Credit_Card_types.map(Temp_dict)

    try:
        new_series = new_series.drop(["Decision"], axis=1)
    except:
        print('already drop')

    y_pred_os_input = clf_os.predict(new_series)

    st.write('Prediction result:')
    if y_pred_os_input[0] == 1:
        st.error('REJECT')
    else:
        st.success('ACCEPT')

    # =========================== Stop ===============================
    st.header('End Classification')

if choice == 'Clustering':

    # =========================== Clustering ===============================
    st.header('Clustering - K Mean Clustering')
    df_cluster = df_eda.copy()
    df_ori = df_cluster.copy()

    # st.write(df_ori)

    # # Transform the decision column in df_cluster into 1 and 0s using Label Encoding
    df_cluster['Decision'] = LabelEncoder().fit_transform(df_cluster.Decision)
    # # Split the dataset
    X = df_cluster.drop('Decision', axis=1)
    y = df_cluster['Decision']
    # Perform dummification on X only
    try:
        new_series = new_series.drop(["Decision"], axis=1)
    except:
        print('already drop')

    X_new = X.append(new_series.iloc[0])
    # reset index
    X_new = X_new.reset_index(drop=True)

    X = pd.get_dummies(X, drop_first=True)

    st.write('Plot the graph using df_ori  to see how is the original data scattered around.')
    sns.relplot(x="Loan_Amount", y="Total_Sum_of_Loan", hue="Decision", data=df_ori)
    st.pyplot()

    km = KMeans(n_clusters=3, random_state=1)
    km.fit(X)
    # km.labels_

    # Create a new dataframe and replace the decision column with km.labels_
    df_clust_label = df_ori.copy()
    df_clust_label = df_clust_label.drop("Decision", axis=1)
    df_clust_label['Decision'] = km.labels_

    st.write('Total_Income_for_Join_Application VS Total_Sum_of_Loan - 3 Cluster')
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    sns.scatterplot(x="Total_Income_for_Join_Application", y="Total_Sum_of_Loan", hue="Decision", data=df_ori)
    plt.subplot(212)
    sns.scatterplot(x="Total_Income_for_Join_Application", y="Total_Sum_of_Loan", hue="Decision", data=df_clust_label)
    st.pyplot()

    st.write('Monthly_Salary VS Loan_Amount - 3 Cluster')
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    sns.scatterplot(x="Monthly_Salary", y="Loan_Amount", hue="Decision", data=df_ori)
    plt.subplot(212)
    sns.scatterplot(x="Monthly_Salary", y="Loan_Amount", hue="Decision", data=df_clust_label)
    st.pyplot()

    y_pred = km.predict(X)
    st.write("Silhouette Score (n=3) = ", silhouette_score(X, y_pred))

    st.subheader('Graph Silhouette Plot')
    plt.figure(figsize=(10, 4))
    silhouette_visualizer(KMeans(3, random_state=12), X, colors='yellowbrick')
    st.pyplot()

    # =================== Predict Input ============
    st.header('Clustering Input - K Mean Clustering')
    st.write('Clustering of your input:')
    st.write(X_new.iloc[-1])

    X_new = pd.get_dummies(X_new, drop_first=True)
    X_new = X_new.drop(X_new.index[:-1])

    new_y_pred = km.predict(X_new)
    st.write('Clustering Label of your input :')
    st.success(new_y_pred)

    # =========================== Stop ===============================
    st.header('End Clustering')

if choice == 'EDA':
    st.title('Exploratory Data Analysis')

    # ================= Data EDA ======================

    corr_matrix = df_eda.corr().abs()
    plt.figure(figsize=(30, 30))
    sns.heatmap(corr_matrix, vmax=0.8, square=True, fmt='.3f', annot=True, annot_kws={'size': 18},
                cmap=sns.color_palette('Blues'))
    st.write('Correlation Matrix')
    st.pyplot()

    df_eda['Employment_Type'].value_counts().plot(kind='bar')
    st.write('Employment_Type')
    st.pyplot()

    df['Property_Type'].value_counts().plot(kind='bar')
    st.write('Property_Type')
    st.pyplot()

    sns.catplot(x="Credit_Card_Exceed_Months", hue="Decision", kind="count", data=df_eda)
    st.write('Credit_Card_Exceed_Months')
    st.pyplot()

    sns.catplot("Decision", col="Employment_Type", col_wrap=5, data=df_eda, kind="count", height=10, aspect=.3)
    st.write('Employment_Type')
    st.pyplot()

    sns.boxplot(x='Employment_Type', y='Loan_Amount', data=df_eda)
    st.write('Loan_Amount')
    st.pyplot()

    data = df_eda2["Loan_Amount"]
    plt.hist(data, bins=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000])
    st.write('Loan_Amount')
    st.pyplot()

    sns.catplot(x="Number_of_Properties", hue="Decision", kind="count", data=df_eda)
    st.write('Number_of_Properties')
    st.pyplot()

    # binning Monthly Salary
    df_eda2["Monthly_SalaryG"] = pd.cut(df_eda2["Monthly_Salary"],
                                        bins=[3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000])
    df_eda2["Monthly_SalaryG"].value_counts().plot(kind="bar")
    st.write('Monthly_SalaryG')
    st.pyplot()

    table = pd.crosstab(df_eda2.Monthly_SalaryG, df_eda2.Decision)
    table.plot.barh(stacked=True)
    plt.ylabel('MONTHLY_SALARY_GROUP')
    st.write('MONTHLY_SALARY_GROUP')
    st.pyplot()

    sns.catplot(x='Total_Sum_of_Loan', y='State', kind='bar', ci=None, data=df_eda, height=5, aspect=3, orient='h')
    st.write('State')
    st.pyplot()

    table = pd.crosstab(df_eda.State, df_eda.Decision)
    table.plot.barh(stacked=True, figsize=(12, 6))
    plt.xlabel('Decision')
    plt.ylabel('State')
    st.write('State')
    st.pyplot()

    # Map
    df_gbp = df_eda[["State", "Total_Sum_of_Loan"]]
    gbp = df_gbp.groupby(["State"], as_index=False).median()

    fp = "./map Malaysia/Malaysia_Polygon.shp"
    map_df = gpd.read_file(fp)
    map_df['name'] = map_df['name'].str.upper()
    map_df["name"] = map_df["name"].replace("KUALA LUMPUR", "KUALALUMPUR")
    map_df["name"] = map_df["name"].replace("NEGERI SEMBILAN", "NEGERISEMBILAN")

    merged = map_df.set_index('name').join(gbp.set_index('State'))
    variable = "Total_Sum_of_Loan"
    vmin, vmax = 0, 33000
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    merged.plot(column=variable, cmap='Greens', linewidth=0.5, ax=ax, edgecolor='0')
    st.write('Map of Malaysia')
    st.pyplot()

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.countplot(x='Decision', data=df_eda, ax=axs[0])
    axs[0].set_title("Frequency of each Loan Decision")
    df.Decision.value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
    axs[1].set_title("Percentage of each Loan Decision")
    st.write('Frequency of each Loan Decision and Percentage of each Loan Decision')
    st.pyplot()

    # ================= Normalization ======================
    # df_normz = df_eda.copy()
    #
    # features_to_scale = ["Loan_Amount", "Monthly_Salary", "Total_Income_for_Join_Application", "Total_Sum_of_Loan"]
    # to_scale = df_normz[features_to_scale]
    #
    # min_max_scaler = StandardScaler()
    # x_scaled = min_max_scaler.fit_transform(to_scale)
    # df_x_scaled = pd.DataFrame(x_scaled, columns=features_to_scale)
    # # df_x_scaled
    #
    # df_normz = df_normz.drop(features_to_scale, axis=1)
    # df_normz = pd.concat([df_normz, df_x_scaled], sort=True, axis=1)

    # ================= Label Encoding ======================
    df_le = df_eda.copy()

    df_le['More_Than_One_Products'] = LabelEncoder().fit_transform(df_le.More_Than_One_Products)
    df_le['Employment_Type'] = LabelEncoder().fit_transform(df_le.Employment_Type)
    df_le['Property_Type'] = LabelEncoder().fit_transform(df_le.Property_Type)
    df_le['State'] = LabelEncoder().fit_transform(df_le.State)
    df_le['Decision'] = LabelEncoder().fit_transform(df_le.Decision)
    df_le['Credit_Card_types'] = LabelEncoder().fit_transform(df_le.Credit_Card_types)

    # ================= SMOTENC ======================
    st.header('After SMOTENC')
    y = df_le.Decision
    X = df_le.drop(["Decision"], axis=1)

    os = SMOTE(random_state=0)
    columns = X.columns
    os_data_X, os_data_y = os.fit_sample(X, y)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['Decision'])

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.countplot(x='Decision', data=os_data_y, ax=axs[0])
    axs[0].set_title("Frequency of each Loan Decision")
    os_data_y.Decision.value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
    axs[1].set_title("Percentage of each Loan Decision")
    st.write('Frequency of each Loan Decision and Percentage of each Loan Decision')
    st.pyplot()


    # ================= Boruta ======================
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))


    colnames = X.columns
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators="auto", random_state=1)
    feat_selector_os = BorutaPy(rf, n_estimators="auto", random_state=1)

    feat_selector.fit(X.values, y.values.ravel())
    feat_selector_os.fit(os_data_X.values, os_data_y.values.ravel())

    boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values('Score', ascending=False)

    st.subheader('Boruta Score before SMOTENC ranking')
    st.write('-----------Top 10------------')
    st.table(boruta_score.head(10))

    st.write('-----------Bottom 10------------')
    st.table(boruta_score.tail(10))

    boruta_score_os = ranking(list(map(float, feat_selector_os.ranking_)), colnames, order=-1)
    boruta_score_os = pd.DataFrame(list(boruta_score_os.items()), columns=['Features', 'Score'])
    boruta_score_os = boruta_score_os.sort_values('Score', ascending=False)

    st.subheader('Boruta Score after SMOTENC ranking')
    st.write('---------Top 10----------')
    st.table(boruta_score_os.head(10))

    st.write('---------Bottom 10----------')
    st.table(boruta_score_os.tail(10))

    sns_boruta_plot = sns.catplot(x="Score", y="Features", data=boruta_score[:], kind="bar", height=14, aspect=1.9,
                                  palette='coolwarm')
    plt.title("Boruta (Imbalance Dataset)")
    st.write('Boruta Score (Imbalance Dataset)')
    st.pyplot()

    sns_boruta_plot = sns.catplot(x="Score", y="Features", data=boruta_score_os[:], kind="bar", height=14, aspect=1.9,
                                  palette='coolwarm')
    plt.title("Boruta (SMOTE Dataset)")
    st.write('Boruta Score (SMOTENC Dataset)')
    st.pyplot()

    # =========================== Stop ===============================
    st.header('End EDA')

if choice == 'ARM':
    # =============== Input Data ==========
    st.subheader('Input Data')
    st.write(new_series)

    # ================= ARM ======================
    st.title('Association Rule Mining')
    st.write('Here is ARM')
    # ================= ARM ======================

    # =========================== Stop ===============================
    st.header('End ARM')
