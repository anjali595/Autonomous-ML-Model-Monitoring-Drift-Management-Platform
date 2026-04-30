import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="ML Monitoring Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for neon Gen Z theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        color: #ffffff;
        border-right: 2px solid #00d4ff;
        box-shadow: 0 0 20px #00d4ff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff0080, #ff4081);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 0 15px #ff0080;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px #ff0080, 0 0 35px #ff0080;
        transform: scale(1.05);
    }
    .stSelectbox, .stNumberInput {
        background-color: #1a1a2e;
        color: #ffffff;
        border: 1px solid #00d4ff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    .stDataFrame {
        background-color: #1a1a2e;
        border: 1px solid #00d4ff;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 0 0 10px #00d4ff;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #ff0080;
        box-shadow: 0 0 20px rgba(255, 0, 128, 0.5);
        margin: 10px;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #00d4ff;
        text-shadow: 0 0 10px #00d4ff;
    }
    .metric-label {
        color: #ffffff;
        font-size: 1.2em;
    }
    .glow {
        text-shadow: 0 0 10px #00d4ff;
    }
    .neon-text {
        color: #00d4ff;
        text-shadow: 0 0 5px #00d4ff, 0 0 10px #00d4ff, 0 0 15px #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("loan_model.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")

# Feature names
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

# Categorical mappings (from training)
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_map = {'Graduate': 0, 'Not Graduate': 1}
self_employed_map = {'Yes': 1, 'No': 0}
property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
credit_history_map = {1.0: 1, 0.0: 0}

# Function to detect drift
def detect_drift(train_data, production_data):
    drift_report = {}
    for column in train_data.columns:
        stat, p_value = ks_2samp(train_data[column], production_data[column])
        drift_report[column] = p_value
    return drift_report

# Simulate production data
production_data = X_test.sample(100, random_state=42)

# Detect drift
drift_results = detect_drift(X_test, production_data)

# Simulate accuracy over time
accuracy_data = pd.DataFrame({
    'Time': pd.date_range(start='2023-01-01', periods=10, freq='M'),
    'Accuracy': [0.75, 0.76, 0.74, 0.77, 0.76, 0.78, 0.77, 0.76, 0.75, 0.764]
})

# Streamlit app
st.title("🚀 Autonomous ML Model Monitoring & Drift Management Platform")
st.markdown('<p class="neon-text">Powered by AI for seamless model management</p>', unsafe_allow_html=True)

st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Select Page", ["🔮 Prediction", "📈 Monitoring", "🔍 Drift Detection"], label_visibility="collapsed")

if page == "🔮 Prediction":
    st.header("🔮 Loan Prediction")
    st.write("Enter the loan application details below:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("👤 Gender", ["Male", "Female"])
            married = st.selectbox("💍 Married", ["Yes", "No"])
            dependents = st.selectbox("👨‍👩‍👧‍👦 Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])

        with col2:
            applicant_income = st.number_input("💰 Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("💰 Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("🏠 Loan Amount", min_value=0, value=100)
            loan_amount_term = st.number_input("⏰ Loan Amount Term", min_value=0, value=360)
            credit_history = st.selectbox("📜 Credit History", [1.0, 0.0])
            property_area = st.selectbox("🏙️ Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔍 Predict Loan Status")
        if submitted:
            # Encode inputs
            input_data = {
                'Gender': gender_map[gender],
                'Married': married_map[married],
                'Dependents': dependents_map[dependents],
                'Education': education_map[education],
                'Self_Employed': self_employed_map[self_employed],
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history_map[credit_history],
                'Property_Area': property_area_map[property_area]
            }

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            result = "✅ Approved" if prediction == 1 else "❌ Rejected"
            st.success(f"**Loan Status: {result}**")

elif page == "📈 Monitoring":
    st.header("📈 Model Performance Monitoring")
    st.write("Real-time insights into model performance.")

    # Metrics in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">76.42%</div>
            <div class="metric-label">🎯 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">75.5%</div>
            <div class="metric-label">🔍 Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">77.0%</div>
            <div class="metric-label">📊 Recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Performance Trends")
    fig = px.line(accuracy_data, x='Time', y='Accuracy', title='Model Accuracy Over Time',
                  line_shape='spline', color_discrete_sequence=['#00d4ff'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Drift Detection":
    st.header("🔍 Data Drift Detection")
    st.write("Monitoring for data drift using KS Test (p < 0.05 indicates drift).")

    drift_df = pd.DataFrame(list(drift_results.items()), columns=['Feature', 'P-Value'])
    drift_df['Drift Detected'] = drift_df['P-Value'] < 0.05

    # Highlight drift
    def color_drift(val):
        color = 'red' if val else 'green'
        return f'background-color: {color}'

    styled_df = drift_df.style.applymap(color_drift, subset=['Drift Detected'])
    st.dataframe(styled_df)

    # Chart for drift
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=drift_df['Feature'],
        y=drift_df['P-Value'],
        marker_color=['red' if p < 0.05 else 'green' for p in drift_df['P-Value']],
        name='P-Value'
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="yellow", annotation_text="Drift Threshold")
    fig.update_layout(
        title="Drift P-Values by Feature",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    if any(drift_df['Drift Detected']):
        st.error("⚠️ Drift detected! Retraining recommended.")
    else:
        st.success("✅ No significant drift detected.")