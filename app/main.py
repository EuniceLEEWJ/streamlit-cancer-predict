import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load dataset and clean
@st.cache_data
def get_clean_data():
    df = pd.read_csv("data/data.csv")
    df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# Sidebar slider + number input

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
        st.slider(label, min_val, max_val,
                  key=slider_key, on_change=slider_changed)
    with col2:
        st.number_input(" ", min_val, max_val,
                        key=input_key, on_change=input_changed)

    return st.session_state[synced_key]

# Sidebar input collection

def add_sidebar(data):
    st.sidebar.header("Cell Nuclei Measurements")
    slider_labels = [(label.replace('_', ' ').title(), label) for label in data.columns if label != 'diagnosis']

    input_dict = {}
    for label, key in slider_labels:
        min_val = float(data[key].min())
        max_val = float(data[key].max())
        default_val = float(data[key].mean())
        input_dict[key] = sync_input(label, min_val, max_val, default_val, key)
    return [input_dict]  # Return as list for consistency

# Radar chart

def get_scaled_values(row_dict, reference_data):
    X = reference_data.drop(['diagnosis'], axis=1, errors='ignore')
    scaled = {}
    for key, value in row_dict.items():
        min_val = X[key].min()
        max_val = X[key].max()
        scaled[key] = (value - min_val) / (max_val - min_val)
    return scaled

def get_radar_chart(input_data, reference_data):
    scaled = get_scaled_values(input_data, reference_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']
    mean_keys = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

    se_keys = ['radius_se', 'texture_se', 'perimeter_se', 'area_se',
               'smoothness_se', 'compactness_se', 'concavity_se',
               'concave points_se', 'symmetry_se', 'fractal_dimension_se']

    worst_keys = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                  'smoothness_worst', 'compactness_worst', 'concavity_worst',
                  'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[scaled[k] for k in mean_keys], theta=categories, fill='toself', name='Mean'))
    fig.add_trace(go.Scatterpolar(r=[scaled[k] for k in se_keys], theta=categories, fill='toself', name='SE'))
    fig.add_trace(go.Scatterpolar(r=[scaled[k] for k in worst_keys], theta=categories, fill='toself', name='Worst'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig

# Prediction logic

def add_predictions(input_dicts):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = pd.DataFrame(input_dicts)
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

    if len(input_dicts) == 1:
        if preds[0] == 0:
            st.success("Benign")
        else:
            st.error("Malignant")
        st.write(f"Probability of being benign: {probs[0][0]:.3f}")
        st.write(f"Probability of being malignant: {probs[0][1]:.3f}")

# Main app

def main():
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🧬", layout="wide")

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("Breast Cancer Predictor")
    st.write("Adjust inputs in the sidebar or upload an Excel file. Predictions and charts will be shown below.")

    uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])
    reference_data = get_clean_data()

    input_data_list = []

    if uploaded_file is not None:
        try:
            user_df = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded.")

            required_cols = reference_data.drop("diagnosis", axis=1).columns
            if not set(required_cols).issubset(set(user_df.columns)):
                st.sidebar.error("Uploaded file is missing required columns.")
                return

            input_data_list = user_df[required_cols].to_dict(orient='records')

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return

    else:
        input_data_list = add_sidebar(reference_data)

    if len(input_data_list) == 1:
        st.subheader("Input Parameters")
        st.dataframe(pd.DataFrame(input_data_list), use_container_width=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            radar_chart = get_radar_chart(input_data_list[0], reference_data)
            st.plotly_chart(radar_chart, use_container_width=True)

        with col2:
            add_predictions(input_data_list)

    else:
        add_predictions(input_data_list)

if __name__ == '__main__':
    main()
