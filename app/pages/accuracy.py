import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Utility: Load and clean dataset
@st.cache_data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data

st.set_page_config(page_title="Model Accuracy", layout="wide")
st.title("Model Accuracy and Performance")

# Load model, scaler, and data
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
data = get_clean_data()
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Scale and predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Accuracy
acc = accuracy_score(y, y_pred)
st.metric("\nModel Accuracy", f"{acc:.2%}")

# Confusion Matrix
st.subheader("\nConfusion Matrix")
cm = confusion_matrix(y, y_pred)
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(ax=ax_cm, cmap='Blues')
st.pyplot(fig_cm)

# ROC Curve
st.subheader("\nROC Curve")
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)
