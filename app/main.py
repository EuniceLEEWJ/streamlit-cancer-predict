import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from io import BytesIO

# Load and clean dataset
def get_clean_data():
    # Ensure the file is properly located in the deployed environment
    data_path = os.path.join("data", "data.csv") if os.path.exists("data") else "data.csv"
    data = pd.read_csv(data_path)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Unified input: slider + number input
def sync_input(label, min_val, max_val, default_val, key):
    synced_key = f"{key}_synced"
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    def slider_changed():
        st.session_state[synced_key] = st.session_state[slider_key]
        st.session_state[input_key] = st.session_state[slider_key]

    def input_changed():
        st.session_state[synced_key] = st.session_state[input_key]
        st.session_state[slider_key] = st.session_state[input_key]

    if synced_key not in st.session_state:
        st.session_state[synced_key] = default_val
        st.session_state[slider_key] = default_val
        st.session_state[input_key] = default_val

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.slider(label, min_val, max_val, key=slider_key, on_change=slider_changed)
    with col2:
        st.number_input(" ", min_val, max_val, key=input_key, on_change=input_changed)

    return st.session_state[synced_key]

# Sidebar input collection
def add_sidebar(data):
    st.sidebar.header("Cell Nuclei Measurements")
    slider_labels = [(col.replace('_', ' ').title(), col) for col in data.columns if col != 'diagnosis']
    input_dict = {}
    for label, key in slider_labels:
        min_val = float(data[key].min())
        max_val = float(data[key].max())
        default_val = float(data[key].mean())
        input_dict[key] = sync_input(label, min_val, max_val, default_val, key)
    return input_dict

# Scale values between 0 and 1 for radar chart
def get_scaled_values(input_dict, reference_data):
    X = reference_data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        min_val = X[key].min()
        max_val = X[key].max()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict

# Generate radar chart
def get_radar_chart(input_data, reference_data):
    scaled = get_scaled_values(input_data, reference_data)
    categories = [key.replace('_mean', '').replace('_se', '').replace('_worst', '').title() for key in list(scaled)[:10]]

    fig = go.Figure()
    for label, keys in [
        ("Mean", [k for k in scaled if '_mean' in k]),
        ("SE", [k for k in scaled if '_se' in k]),
        ("Worst", [k for k in scaled if '_worst' in k])
    ]:
        fig.add_trace(go.Scatterpolar(
            r=[scaled[k] for k in keys],
            theta=categories,
            fill='toself',
            name=label
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    return fig

# Predict single input
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = pd.DataFrame([input_data])  # Use input_data (singular) here
    input_scaled = scaler.transform(input_array)

    preds = model.predict(input_scaled)
    probs = model.predict_proba(input_scaled)

    output_df = input_array.copy()
    output_df['Prediction'] = ['Benign' if p == 0 else 'Malignant' for p in preds]
    output_df['Benign_Prob'] = probs[:, 0]
    output_df['Malignant_Prob'] = probs[:, 1]

    st.subheader("Prediction Results")
    with st.expander("Show Prediction Table", expanded=True):
        st.dataframe(output_df.round(3), use_container_width=True)

    if len([input_data]) == 1:
        if preds[0] == 0:
            st.success("Benign")
        else:
            st.error("Malignant")
        st.write(f"Probability of being benign: {probs[0][0]:.3f}")
        st.write(f"Probability of being malignant: {probs[0][1]:.3f}")

# Predict multiple inputs and display
def predict_all_saved(saved_inputs):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    df_inputs = pd.DataFrame(saved_inputs)
    scaled_inputs = scaler.transform(df_inputs)
    preds = model.predict(scaled_inputs)
    probs = model.predict_proba(scaled_inputs)
    df_inputs['Prediction'] = ['Benign' if p == 0 else 'Malignant' for p in preds]
    df_inputs['Benign_Prob'] = probs[:, 0]
    df_inputs['Malignant_Prob'] = probs[:, 1]
    return df_inputs

# Convert dataframe to CSV download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Main app
def main():
    st.set_page_config("Breast Cancer Predictor", "🧬", layout="wide")
    if 'saved_inputs' not in st.session_state:
        st.session_state.saved_inputs = []

    # Ensure styles are applied correctly
    try:
        with open("assets/style.css") as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom styles could not be applied as the style.css file was not found.")

    reference_data = get_clean_data()
    uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])

    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
            required_cols = reference_data.drop("diagnosis", axis=1).columns
            if not set(required_cols).issubset(set(user_df.columns)):
                st.sidebar.error("Uploaded file missing required columns.")
                return
            input_data = user_df.iloc[0].to_dict()
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            return
    else:
        input_data = add_sidebar(reference_data)

    st.title("Breast Cancer Predictor")
    st.subheader("Input Parameters")
    st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

    if st.button("💾"):
        st.session_state.saved_inputs.append(input_data.copy())
        st.success("Successfully Saved.")

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data, reference_data)
        st.plotly_chart(radar_chart, use_contain_width=True)
    with col2:
        add_predictions(input_data)

    if st.session_state.saved_inputs:
        st.subheader("Saved Inputs")
        df_saved = pd.DataFrame(st.session_state.saved_inputs)
        st.dataframe(df_saved, use_container_width=True)

        selected_idx = st.selectbox("Select a saved input to use", options=range(len(df_saved)), format_func=lambda i: f"Saved Input {i + 1}")

        if st.button("Predict Selected Input"):
            input_data = st.session_state.saved_inputs[selected_idx]
            st.success(f"Prediction for Saved Input {selected_idx + 1}:")
            add_predictions(input_data)

        if st.button("Predict All"):
            st.subheader("Batch Predictions")
            result_df = predict_all_saved(st.session_state.saved_inputs)
            st.dataframe(result_df.round(3), use_container_width=True)

        # Keep only the "Clear All" button to remove all saved inputs
        if st.button("🗑️ Clear All"):
            st.session_state.saved_inputs = []
            st.warning("Saved inputs cleared.")

if __name__ == '__main__':
    main()
