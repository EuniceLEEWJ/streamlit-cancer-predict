import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

st.set_page_config(page_title="Feature Importance", page_icon="📊")

st.title("Feature Importance - Logistic Regression Coefficients")
st.write(
    "The chart below shows the absolute value of logistic regression coefficients, "
    "indicating the relative importance of each feature in predicting breast cancer."
)

# Load model and data
model_path = "model/model.pkl"
scaler_path = "model/scaler.pkl"
data_path = "data/data.csv"

if not (os.path.exists(model_path) and os.path.exists(data_path)):
    st.error("Model or data file not found. Please train the model first.")
else:
    model = pickle.load(open(model_path, "rb"))
    data = pd.read_csv(data_path)
    data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    feature_names = data.drop(columns=["diagnosis"]).columns
    coefficients = model.coef_[0]  # shape (1, n_features)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute Importance": abs(coefficients)
    }).sort_values(by="Absolute Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Absolute Importance",
        y="Feature",
        orientation="h",
        title="Logistic Regression Feature Importance (by Coefficient)",
        labels={"Absolute Importance": "Importance (|Coefficient|)"}
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Coefficients Table"):
        st.dataframe(importance_df, use_container_width=True)
