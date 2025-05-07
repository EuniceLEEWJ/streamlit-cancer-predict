import streamlit as st 
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load and clean dataset
def get_clean_data(df=None):
    if df is None:
        df = pd.read_csv("data/data.csv")
    df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

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

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        min_val = float(data[key].min())
        max_val = float(data[key].max())
        default_val = float(data[key].mean())
        input_dict[key] = sync_input(label, min_val, max_val, default_val, key)

    return input_dict

# Scale values between 0 and 1 for radar chart
def get_scaled_values(input_dict, reference_data):
    X = reference_data.drop(['diagnosis'], axis=1, errors='ignore')
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)

    return scaled_dict

# Generate radar chart
def get_radar_chart(input_data, reference_data):
    input_data = get_scaled_values(input_data, reference_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
           input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
           input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
           input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
           input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    
    return fig

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    return model, scaler

# Make prediction and display result
def add_predictions(input_data, model, scaler):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)
    probs = model.predict_proba(input_array_scaled)[0]

    st.subheader("Cell Cluster Prediction")
    if prediction[0] == 0:
        st.success("Benign")
    else:
        st.error("Malignant")

    st.write(f"Probability of being benign: {probs[0]:.3f}")
    st.write(f"Probability of being malignant: {probs[1]:.3f}")
    st.info("This app assists in medical diagnosis but is not a replacement for professional medical advice.")

# Main app
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    st.title("Breast Cancer Predictor")
    st.write("Adjust inputs in the sidebar or upload a file. Below is a preview of your inputs and predictions.\n\n\n")

    uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])
    reference_data = get_clean_data()
    model, scaler = load_model_and_scaler()

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom file uploaded.")

            required_cols = reference_data.drop("diagnosis", axis=1).columns
            if not set(required_cols).issubset(set(user_df.columns)):
                st.sidebar.error("Uploaded file is missing required columns.")
                return

            if user_df.shape[0] == 1:
                input_data = user_df.iloc[0].to_dict()
                st.subheader("Input Parameters")
                st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.plotly_chart(get_radar_chart(input_data, reference_data), use_container_width=True)
                with col2:
                    add_predictions(input_data, model, scaler)

            else:
                predictions = []
                for index, row in user_df.iterrows():
                    input_data = row.to_dict()
                    input_array = np.array(list(input_data.values())).reshape(1, -1)
                    input_array_scaled = scaler.transform(input_array)

                    pred = model.predict(input_array_scaled)[0]
                    probs = model.predict_proba(input_array_scaled)[0]

                    predictions.append({
                        "Patient #": index + 1,
                        "Prediction": "Malignant" if pred else "Benign",
                        "Probability Benign": round(probs[0], 3),
                        "Probability Malignant": round(probs[1], 3)
                    })

                with st.expander("🔍 View Prediction Results Table", expanded=True):
                    st.dataframe(pd.DataFrame(predictions), use_container_width=True)

        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            return

    else:
        input_data = add_sidebar(reference_data)
        st.subheader("Input Parameters")
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(get_radar_chart(input_data, reference_data), use_container_width=True)
        with col2:
            add_predictions(input_data, model, scaler)

if __name__ == '__main__':
    main()
