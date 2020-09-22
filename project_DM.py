import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import geopandas as gpd
import descartes
import streamlit as st
import cv2

from imblearn.over_sampling import SMOTE
from boruta import BorutaPy

from sklearn.model_selection import train_test_split, cross_val_score

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

select_yes_no = [np.nan, 'YES', 'NO']
select_Employment_Type = [np.nan, 'EMPLOYER', 'SELF_EMPLOYED', 'GOVERNMENT', 'EMPLOYEE', 'FRESH_GRADUATE']
select_Property_Type = [np.nan, 'CONDOMINIUM', 'BUNGALOW', 'TERRACE', 'FLAT']
select_State = [np.nan, 'JOHOR', 'SELANGOR', 'KUALALUMPUR', 'PENANG', 'NEGERISEMBILAN', 'SARAWAK', 'SABAH',
                'TERENGGANU', 'KEDAH']
select_CC_type = [np.nan, 'PLATINUM', 'NORMAL', 'GOLD']

# ============================ streamlit UI

menu = ['EDA', 'Feature Selection', 'Classification', 'Clustering', 'ARM']
st.sidebar.subheader('Main Menu')
choice = st.sidebar.selectbox("Select Menu Page", menu)

if choice == 'Classification' or choice == 'Clustering':
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

    kcross = "./k_cross/"
    img_knn = 'knn.png'
    img_rf = 'rf.png'
    img_xgb = 'xgb.png'
    img_nb = 'nb.png'

    st.subheader('Line Chart')
    st.write('k-Cross Validation with k = 5, 10, 15, 20')
    # img1 = cv2.imread(kcross+img_nb)
    # st.image(img1)

    # ====== Random Forest ================
    st.header('Random Forest - Imbalance Dataset')

    clf = RandomForestClassifier(random_state=10)
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

    st.subheader('Line Chart')
    st.write('k-Cross Validation with k = 5, 10, 15, 20')
    # img2 = cv2.imread(kcross + img_rf)
    # st.image(img2)

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

    st.subheader('Line Chart')
    st.write('k-Cross Validation with k = 5, 10, 15, 20')
    # img3 = cv2.imread(kcross + img_knn)
    # st.image(img3)

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

    st.subheader('Line Chart')
    st.write('k-Cross Validation with k = 5, 10, 15, 20')
    # img4 = cv2.imread(kcross + img_xgb)
    # st.image(img4)

    # ====== Graph ROC ================
    st.header('Graph ROC - Imbalance Dataset')

    fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_nb)
    fpr_clf, tpr_clf, thresholds_clf = roc_curve(y_test, prob_clf)
    fpr_XGtree, tpr_XGtree, thresholds_XGtree = roc_curve(y_test, prob_Xgtree)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, prob_knn)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(fpr_NB, tpr_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(fpr_clf, tpr_clf, color='blue', label='RF', linewidth=linewidth)
    plt.plot(fpr_XGtree, tpr_XGtree, color='purple', label='XG Tree', linewidth=linewidth)
    plt.plot(fpr_knn, tpr_knn, color='green', label='KNN', linewidth=linewidth)

    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()
    st.write('Graph above shows the ROC curves that fitting all the classifiers with imbalanced'
             ' dataset. All the classifier curve are '
             'closer to the 45-degree diagonal of the ROC space. Hence, the accuracy of the test for all classifiers '
             'are low.')

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
    plt.plot(fpr_XGtree, tpr_XGtree, color='purple', label='XG Tree', linewidth=linewidth)
    plt.plot(fpr_knn, tpr_knn, color='green', label='KNN', linewidth=linewidth)

    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()

    st.write('Graph above shows the ROC curves that fitting all the classifiers with balanced dataset '
             '(after SMOTE). The curve for '
             'Random Forest and XGBoost Tree classifiers are closer to the top left while Naive Bayes classifier is '
             'closer to the  45-degree  diagonal  line  and  K-Nearest  Neighbors  is  in  between  them. Hence, '
             'Random Forest and XGBoost Tree classifiers have the best accuracy and performance followed by Naive '
             'Bayes and K-Nearest Neighbours.')

    st.write('**********************')

    # ====== Precision - Recall ================
    st.header('Graph Precision-Recall - Imbalance Dataset')
    prec_NB, rec_NB, thresholds_NB = precision_recall_curve(y_test, prob_nb)
    prec_clf, rec_clf, thresholds_clf = precision_recall_curve(y_test, prob_clf)
    prec_knn, rec_knn, thresholds_knn = precision_recall_curve(y_test, prob_knn)
    prec_XGtree, rec_XGtree, thresholds_XGtree = precision_recall_curve(y_test, prob_Xgtree)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(prec_NB, rec_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(prec_clf, rec_clf, color='blue', label='RF', linewidth=linewidth)
    plt.plot(prec_knn, rec_knn, color='green', label='KNN', linewidth=linewidth)
    plt.plot(prec_XGtree, rec_XGtree, color='purple', label='XG Tree', linewidth=linewidth)

    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()
    st.write(
        'The graph shows the Precision-Recall curve that of all the classifiers fitting the imbalanced dataset. A '
        'model '
        'with high precision and sensitivity is depicted as a point at (1,1). A skilful model is represented by '
        'a curve that bows towards (1,1). All the classifier curves are far from the point (1,1). Hence, '
        'the precision-recall for all classifiers are low.')

    st.header('Graph Precision-Recall - Oversampled Dataset')
    prec_NB, rec_NB, thresholds_NB = precision_recall_curve(y_test_os, prob_nb_os)
    prec_clf, rec_clf, thresholds_clf = precision_recall_curve(y_test_os, prob_clf_os)
    prec_knn, rec_knn, thresholds_knn = precision_recall_curve(y_test_os, prob_knn_os)
    prec_XGtree, rec_XGtree, thresholds_XGtree = precision_recall_curve(y_test_os, prob_Xgtree_os)

    plt.figure(figsize=(10, 8))
    linewidth = 1
    plt.plot(prec_NB, rec_NB, color='orange', label='NB', linewidth=linewidth)
    plt.plot(prec_clf, rec_clf, color='blue', label='RF', linewidth=linewidth)
    plt.plot(prec_knn, rec_knn, color='green', label='KNN', linewidth=linewidth)
    plt.plot(prec_XGtree, rec_XGtree, color='purple', label='XG Tree', linewidth=linewidth)

    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()
    st.write('Above graph shows the Precision-Recall curve that of all the '
             'classifiers fitting the balanced dataset. Random Forest '
             'and XGBoost Tree classifier curves are closer from the point (1,1). These two classifiers have a high '
             'precision and sensitivy rate. K-Nearest Neighbors has the lowest precision and sensitivity rate.')

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
    st.header('End of Classification')

if choice == 'Clustering':

    # =========================== Clustering ===============================
    st.title('Clustering - K Mean Clustering')
    df_cluster = df_eda.copy()
    df_ori = df_cluster.copy()


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

    distortions = []

    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)
    # plot
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    st.subheader('Line Chart')
    st.pyplot()
    st.write('From the line chart, it is possible to visually determine the best value for k with respect to the '
             'banking dataset.The line chart looks like an arm,  thus the “elbow” (the point of inflection on the '
             'curve) is the best value of k.  In this case, the best value of k is 3.')

    # st.subheader('Scatter Plot')
    # sns.relplot(x="Loan_Amount", y="Total_Sum_of_Loan", hue="Decision", data=df_ori)
    # st.pyplot()
    # st.write('')

    km = KMeans(n_clusters=3, random_state=1)
    km.fit(X)
    # km.labels_

    # Create a new dataframe and replace the decision column with km.labels_
    df_clust_label = df_ori.copy()
    df_clust_label = df_clust_label.drop("Decision", axis=1)
    df_clust_label['Decision'] = km.labels_

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    sns.scatterplot(x="Total_Income_for_Join_Application", y="Total_Sum_of_Loan", hue="Decision", data=df_ori)
    plt.subplot(212)
    sns.scatterplot(x="Total_Income_for_Join_Application", y="Total_Sum_of_Loan", hue="Decision", data=df_clust_label)
    st.subheader('Scatter Plot')
    st.pyplot()
    st.write('Above figures show the before clustering and after clustering scatter plot ofMonthly Salary vs Total '
             'Sum of Loan with '
             'k = 3.  There are three subgroups in the data such that data points in the same cluster are very similar '
             'while data points in different clusters are very different')

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    sns.scatterplot(x="Monthly_Salary", y="Loan_Amount", hue="Decision", data=df_ori)
    plt.subplot(212)
    sns.scatterplot(x="Monthly_Salary", y="Loan_Amount", hue="Decision", data=df_clust_label)
    st.subheader('Scatter Plot')
    st.pyplot()
    st.write('Above figures show the before clustering and after clustering '
             'scatter plot ofMonthly Salary vs Loan Amount. '
             ' There no subgroups or clusters found in data.  This is because the features may be weakly correlated, '
             'hence, there are no groups of data such that the data points are very similar')

    plt.figure(figsize=(10, 8))

    plt.subplot(211)
    sns.scatterplot(x="Monthly_Salary", y="Total_Sum_of_Loan", hue="Decision", data=df_ori)
    plt.subplot(212)
    sns.scatterplot(x="Monthly_Salary", y="Total_Sum_of_Loan", hue="Decision", data=df_clust_label)
    st.subheader('Scatter Plot')
    st.pyplot()
    st.write('Above figures show the scatter plot of Total Income for Join '
             'Application and Total Sum of Loan with k = 3 '
             'before and after clustering .  There are three subgroups  in  the  data  such  that  data  points  in  '
             'the  same  cluster  are  very similar while data points in different clusters are very different')




    st.subheader('Graph Silhouette Plot')
    plt.figure(figsize=(10, 4))
    silhouette_visualizer(KMeans(3, random_state=12), X, colors='yellowbrick')
    st.pyplot()
    y_pred = km.predict(X)
    st.write("Silhouette Score (n=3) = ", silhouette_score(X, y_pred))
    st.write('The value of the silhouette coefficient is between [-1, 1].  A score of 1 denotes the best. '
             'It means that '
             'the data point is extremely compact within the cluster to which it belongs and far away from the other '
             'clusters.  The worst value is -1 and values near 0 represents overlapping clusters The Silhouette '
             'coefficient for k=3 is 0.36593521457784234. '
             'This shows that there are overlapping clusters')




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
    st.header('End of Clustering')

if choice == 'EDA':
    st.title('Exploratory Data Analysis')

    # ================= Data EDA ======================

    # st.write('Managing Missing Data')
    # sns.boxplot(x='Loan_Amount', y='Employment_Type', data=df)
    # st.pyplot()
    # sns.boxplot(x='Monthly_Salary', y='Employment_Type', data=df)
    # st.pyplot()
    # sns.boxplot(x='Total_Sum_of_Loan', y='Employment_Type', data=df)
    # st.pyplot()
    # sns.boxplot(x='Total_Income_for_Join_Application', y='Employment_Type', data=df)
    # st.pyplot()

    corr_matrix = df_eda.corr().abs()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, vmax=0.8, square=True, fmt='.3f', annot=True, annot_kws={'size': 18},
                cmap=sns.color_palette('Blues'))
    st.subheader('Correlation Matrix')
    st.pyplot()
    st.write('Most of the features have weak correlation in which the correlation score is0.01 and lower.There are a '
             'few features that have moderately strong correlation such as YearstoFinancialFreedom and '
             'NumberofDependents with a correlation score of 0.612')

    sns.catplot(x="Property_Type", hue="Decision", kind="count", data=df_eda)
    st.subheader('Count Plot')
    st.pyplot()
    st.write('Based on the count plot, people want-ing  to  purchase  condominiums  has  the  highest  loan  request  '
             'accepted  and rejected,  followed  closely  those  who  owns  terrace.This  chart  tells  us  '
             'that majority of the people requesting for a loan to purchase condominiums.')

    sns.catplot(x="Credit_Card_Exceed_Months", hue="Decision", kind="count", data=df_eda)
    st.subheader('Count Plot')
    st.pyplot()
    st.write('Based on the countplot,people who have not paid the credit card payment for 7 months has thehighest '
             'loan request accepted.  This chart could possibly be evidence that the  feature  '
             'creditcardexceedingmonths  is  not  a  strong  feature  in  loan-approval decision making.')

    sns.catplot("Decision", col="Employment_Type", col_wrap=5,
                data=df_eda, kind="count", height=10, aspect=.3)
    st.subheader('Count Plot')
    st.pyplot()
    st.write('Based on the count plot, employee has the highest loan request accepted among the other employment '
             'types. People who are self employed have the highest loan request rejected')

    sns.catplot(x="Number_of_Properties", hue="Decision", kind="count", data=df_eda)
    st.subheader('Count Plot')
    st.pyplot()
    st.write('Based  on  the  count  plot,people  with  two  properties  has  the  highest  loan  request  accepted  '
             'and  rejected. This chart tells us that majority of the people requesting for a loan has 2 properties')

    df_eda2["Monthly_SalaryG"] = pd.cut(df_eda2["Monthly_Salary"],
                                        bins=[3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000])
    df_eda2["Monthly_SalaryG"].value_counts().plot(kind="bar")
    st.subheader('Barplot')
    st.pyplot()
    st.write('Based on the barplot, majority of the people applying for a loan has a monthly salary ranging from 5000 '
             'to 6000.')

    # CROSSTAB OF MONTHLY SALARY AND DECISION
    table = pd.crosstab(df_eda2.Monthly_SalaryG, df_eda2.Decision)
    table.plot.barh(stacked=True)
    plt.ylabel('MONTHLY_SALARY_GROUP')
    st.subheader('Stack Plot')
    st.pyplot()
    st.write('Based on the plot,people with monthly salary ranging from 5000 to 6000 and 11000 to 12000 has the '
             'highest loan request accepted.  People with monthly salary ranging from 5000 to 6000 also has the '
             'highest loan request rejected.  This further proves and concludes that majority of the people applying '
             'for a loan has a monthly salary ranging from 5000 to 6000.')

    # Decision by State
    table = pd.crosstab(df_eda.State, df_eda.Decision)
    table.plot.barh(stacked=True, figsize=(12, 6))
    plt.xlabel('Decision')
    plt.ylabel('State')
    st.subheader('Stack Plot')
    st.pyplot()
    st.write('Based on the plot, Kuala Lumpur has the highest loan request  accepted  and  rejected.   This  implies  '
             'that  majority  of  the  people applying  for  a  loan  is  from  the  Kuala  Lumpur  area.   Sabah,  '
             'Kedah  and Terengganu have the lowest amount of loan applicants.')

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
    st.subheader('Geographical Plot')
    st.pyplot()
    st.write('Based on the plot, Sarawak has the highest total sum of loan Terengganu has the lowest total sum of '
             ' loan. This could possibly imply that property value in Sarawak is higher compared to all other states in '
             'Malaysia and thus, the high total sum of loan.')

    # ================= Label Encoding ======================
    df_le = df_eda.copy()

    df_le['More_Than_One_Products'] = LabelEncoder().fit_transform(df_le.More_Than_One_Products)
    df_le['Employment_Type'] = LabelEncoder().fit_transform(df_le.Employment_Type)
    df_le['Property_Type'] = LabelEncoder().fit_transform(df_le.Property_Type)
    df_le['State'] = LabelEncoder().fit_transform(df_le.State)
    df_le['Decision'] = LabelEncoder().fit_transform(df_le.Decision)
    df_le['Credit_Card_types'] = LabelEncoder().fit_transform(df_le.Credit_Card_types)

    # ================= SMOTENC ======================
    st.header('Data Imbalance Treatment (SMOTE)')

    st.subheader('Before SMOTE')
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.countplot(x='Decision', data=df_eda, ax=axs[0])
    axs[0].set_title("Frequency of each Loan Decision")
    df.Decision.value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
    axs[1].set_title("Percentage of each Loan Decision")
    st.write('Frequency of each Loan Decision and Percentage of each Loan Decision')
    st.pyplot()
    st.write('There are two instances in the dependent variable,Decision which is Accept and Reject.  Based on the '
             'figure, there is an unequal distribution of classes in dataset.  From 2208 objects in the dataset, '
             '1661 objects have Decision = Accept and 547 objects have Decision = Reject')

    y = df_le.Decision
    X = df_le.drop(["Decision"], axis=1)

    os = SMOTE(random_state=0)
    columns = X.columns
    os_data_X, os_data_y = os.fit_sample(X, y)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['Decision'])

    st.subheader('After SMOTE')
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.countplot(x='Decision', data=os_data_y, ax=axs[0])
    axs[0].set_title("Frequency of each Loan Decision")
    os_data_y.Decision.value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
    axs[1].set_title("Percentage of each Loan Decision")
    st.write('Frequency of each Loan Decision and Percentage of each Loan Decision')
    st.pyplot()
    st.write('Oversampling is done on the examples in the minority class (Decision = Reject).  Based on the figure,'
             'there is an unequal distribution of classes in dataset. After applying SMOTE, the number of objects in '
             'the dataset has increased from 2208 to3322 objects.  1661 objects have Decision = Accept and 1661 '
             'objects have Decision = Reject.The class distribution is now balanced. ')

    # =========================== Stop ===============================
    st.header('End of EDA')

if choice == 'Feature Selection':

    st.title('Feature Selection')
    st.header('Feature Selection using Boruta')

    # Label Encoding
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

    st.subheader('Boruta Score before SMOTE ranking')
    st.write('-----------Top 10------------')
    st.table(boruta_score.head(10))
    st.write('Top 10 Features Selected in the Imbalanced Dataset. Only one feature has the highest score of 1.0 which '
             'is TotalIncomeforJoinApplication. Other features in the top 10 features have a score ranging from 0.53 '
             'to 0.95')

    st.write('-----------Bottom 10------------')
    st.table(boruta_score.tail(10))
    st.write('Bottom  10  Features  Selected  in  the  ImbalancedDataset.There  are  a  few  features  with  score  '
             'between  0.05  to  0.47.   There is one features with score 0.0.  Features with low score are not '
             'relevant to the target variable.')

    sns_boruta_plot = sns.catplot(x="Score", y="Features", data=boruta_score[:], kind="bar", height=14, aspect=1.9,
                                  palette='coolwarm')
    plt.title("Boruta (Imbalance Dataset)")
    st.write('Boruta Score (Imbalance Dataset)')
    st.pyplot()
    st.write('The importance of the features in the imbalanced dataset are ranked and shown in the above figure')

    boruta_score_os = ranking(list(map(float, feat_selector_os.ranking_)),
                              colnames, order=-1)
    boruta_score_os = pd.DataFrame(list(boruta_score_os.items()), columns=['Features', 'Score'])
    boruta_score_os = boruta_score_os.sort_values('Score', ascending=False)

    st.subheader('Boruta Score after SMOTE ranking')
    st.write('---------Top 10----------')
    st.table(boruta_score_os.head(10))
    st.write('All of  the  features  have  a  score  of  1.0.  Features  such  as  MonthlySalary,LoanAmount and '
             'TotalIncomeforJoinApplication are believed to be useful to a model in predicting the target variable.')

    st.write('---------Bottom 10----------')
    st.table(boruta_score_os.tail(10))
    st.write('There are one features with score 0.0.  The rest are 1.0.  Features with low score are not relevant to '
             'the target variable')

    sns_boruta_plot = sns.catplot(x="Score", y="Features", data=boruta_score_os[:], kind="bar", height=14, aspect=1.9,
                                  palette='coolwarm')
    plt.title("Boruta (SMOTE Dataset)")
    st.write('Boruta Score (SMOTE Dataset)')
    st.pyplot()
    st.write('The importance of the features in the balanced dataset are ranked ands hown in the above figure')

    # =========================== Stop ===============================
    st.header('End of Feature Selection')

if choice == 'ARM':
    st.title("Association Rule Mining")
    st.header('Rules Found')

    df_arm = df_eda.copy()
    df_arm = df_arm[["Employment_Type", "Credit_Card_types", "Property_Type"]]
    records = []
    for i in range(0, df_arm.shape[0]):
        records.append([str(df_arm.values[i, j]) for j in range(0, df_arm.shape[1])])

    # ================= ARM ======================
    ar = apriori(records, min_support=0.0045, min_confidence=0.1, min_lift=1.5, min_length=2)
    ar_result = list(ar)
    association_results = ar_result.copy()

    cnt = 0

    for item in association_results:
        cnt += 1
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        st.write("(Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])

        # second index of the inner list
        st.write("Support: " + str(round(item[1], 3)))

        # third index of the list located at 0th
        # of the third index of the inner list

        st.write("Confidence: " + str(round(item[2][0][2], 4)))
        st.write("Lift: " + str(round(item[2][0][3], 4)))
        st.write("***************************************")

    st.write('Based on the following generated rules, let’s analyse the rules between government  employee type '
             'and credit '
             'card types.  The  lift  of  government  employee  to  platinum  credit  card  '
             'is  higher compared to other credit card types.  This indicates that they have a strongassociation. '
             'The confidence between government employee towards platinum credit card is also higher.  '
             'This shows that in all transaction that is made by the government employees, the platinum credit card '
             'appears more often compared to other credit card types.  The support of government employeeto  the  '
             'normal  credit  card  is  higher  compared  to  other  credit  card  types. '
             'Based on these analysis, the bank can recommend the most suitable credit card to potential '
             'customers based on their employment type.')

    st.write('Let\'s also analyse the rules between terrace property type and credit '
             'card types. The lift of terrace property to gold credit card is higher '
             'compared to other credit card types. This indicates that they  have  a  strong  association.   The  '
             'confidence  between  terrace  property towards normal credit card is higher compared to others.  '
             'This shows that in all transaction that is made by someone that wants to purchase a terrace property, '
             ' the  normal  credit  card  appears  more  often  compared  to  other credit card types.  The '
             'support of terrace property to the normal credit card is higher compared to other credit card types. '
             ' The fraction of transactions that contain terrace property and Normal credit card is higher.  Based '
             'on these  analysis,  the  bank  can  recommend  the  most  suitable  credit  card  to potential '
             'customers based on the property they want to purchase.')

    # =========================== Stop ===============================
    st.header('End of ARM')
