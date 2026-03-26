
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")
st.title("🚀 LuxeLoop Executive Analytics Dashboard")

df = pd.read_csv("sample_training_data.csv")

section = st.sidebar.radio("Navigation",[
"Project Overview",
"Descriptive Analytics",
"Classification Results",
"Regression Results",
"Clustering Results",
"Association Rules",
"Business Recommendations"
])

# -------- PROJECT OVERVIEW --------
if section=="Project Overview":
    st.header("Executive Summary")

    col1,col2,col3 = st.columns(3)
    col1.metric("Total Users", len(df))
    col2.metric("Avg Spend", int(df.max_spend.mean()))
    col3.metric("Conversion Rate (%)", round(df.luxe_loop_interest.mean()*100,2))

    st.subheader("Key Insight")
    st.success("Higher income and trust significantly drive spending and platform adoption.")

# -------- DESCRIPTIVE --------
if section=="Descriptive Analytics":
    st.header("Customer Behavior Analysis")

    st.plotly_chart(px.histogram(df,x="max_spend",title="Spend Distribution"))
    st.plotly_chart(px.box(df,x="income_group",y="max_spend",title="Income vs Spend"))
    st.plotly_chart(px.scatter(df,x="trust_score",y="max_spend",title="Trust vs Spend"))

# encoding
df_enc = df.copy()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

# -------- CLASSIFICATION --------
if section=="Classification Results":
    X = df_enc.drop("luxe_loop_interest",axis=1)
    y = df_enc["luxe_loop_interest"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)

    pred = clf.predict(X_test)

    st.header("Model Performance")
    col1,col2 = st.columns(2)
    col1.metric("Accuracy", round(accuracy_score(y_test,pred),2))
    col1.metric("Precision", round(precision_score(y_test,pred),2))
    col2.metric("Recall", round(recall_score(y_test,pred),2))
    col2.metric("F1 Score", round(f1_score(y_test,pred),2))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test,pred))

# -------- REGRESSION --------
if section=="Regression Results":
    Xr = df_enc.drop("max_spend",axis=1)
    yr = df_enc["max_spend"]

    Xr_train,Xr_test,yr_train,yr_test = train_test_split(Xr,yr,test_size=0.2,random_state=42)

    reg = RandomForestRegressor()
    reg.fit(Xr_train,yr_train)

    preds = reg.predict(Xr_test)

    st.metric("MAE", int(mean_absolute_error(yr_test,preds)))
    st.metric("R2 Score", round(r2_score(yr_test,preds),2))

    plot_df = pd.DataFrame({"Actual": yr_test, "Predicted": preds})
    st.plotly_chart(px.scatter(plot_df,x="Actual",y="Predicted"))

# -------- CLUSTERING --------
if section=="Clustering Results":
    scaler = StandardScaler()
    clusters = KMeans(n_clusters=4,random_state=42).fit_predict(scaler.fit_transform(df_enc))

    df["cluster"] = clusters

    st.plotly_chart(px.scatter(df,x="max_spend",y="trust_score",color="cluster"))

    st.info("Segments identified based on spending and trust behavior.")

# -------- ASSOCIATION --------
if section=="Association Rules":
    dummies = pd.get_dummies(df[["income_group","city_tier","purchase_frequency"]])
    freq = apriori(dummies,min_support=0.05,use_colnames=True)
    rules = association_rules(freq,metric="confidence",min_threshold=0.3)

    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

# -------- RECOMMENDATIONS --------
if section=="Business Recommendations":
    st.header("Strategic Insights")

    st.success("Focus on high-income, high-trust users for premium offerings.")
    st.info("Use discounts for price-sensitive segments.")
    st.warning("Improve trust mechanisms for new users.")
