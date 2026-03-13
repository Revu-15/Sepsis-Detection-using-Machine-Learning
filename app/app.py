import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path
app_dir = Path(__file__).parent
project_root = app_dir.parent
sys.path.insert(0, str(project_root))

# Import DataPreprocessor
import importlib.util
spec = importlib.util.spec_from_file_location(
    "data_preprocessing", 
    project_root / "src" / "data_preprocessing.py"
)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)
DataPreprocessor = data_preprocessing.DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Blood Infection Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid red;
    }
    .low-risk {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid green;
    }
</style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_artifacts():
    """Load pre-trained model and scaler"""
    try:
        model = joblib.load(str(project_root / 'models' / 'infection_model.pkl'))
        scaler = joblib.load(str(project_root / 'models' / 'scaler.pkl'))
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ Model or scaler not found: {e}")
        return None, None

def main():
    # Header
    st.markdown("# 🏥 Blood Infection (Sepsis) Prediction System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio("Select a page:", 
                           ["🔍 Prediction", "📊 About", "ℹ️ Instructions"])
    
    if page == "🔍 Prediction":
        prediction_page()
    elif page == "📊 About":
        about_page()
    else:
        instructions_page()

def prediction_page():
    """Main prediction interface"""
    st.header("🔍 Patient Risk Assessment")
    
    # Load model and scaler
    model, scaler = load_artifacts()
    if model is None or scaler is None:
        st.stop()
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📤 Upload CSV", "📋 Sample Data"])
    
    with tab1:
        st.subheader("Enter Patient Medical Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Vital Signs & Labs")
            wbc_count = st.number_input(
                "WBC Count (10³/μL)",
                min_value=0.0, max_value=100.0, value=7.0,
                help="White Blood Cell count"
            )
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=35.0, max_value=42.0, value=37.0,
                help="Body temperature"
            )
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=40, max_value=200, value=80,
                help="Beats per minute"
            )
            respiratory_rate = st.number_input(
                "Respiratory Rate (/min)",
                min_value=10, max_value=50, value=16,
                help="Breaths per minute"
            )
        
        with col2:
            st.markdown("### Blood Chemistry")
            lactate = st.number_input(
                "Lactate (mmol/L)",
                min_value=0.0, max_value=20.0, value=1.5,
                help="Blood lactate level"
            )
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=40, max_value=400, value=100,
                help="Blood glucose level"
            )
            platelet_count = st.number_input(
                "Platelet Count (10³/μL)",
                min_value=0, max_value=500, value=250,
                help="Platelet count"
            )
            bilirubin = st.number_input(
                "Bilirubin (mg/dL)",
                min_value=0.0, max_value=20.0, value=0.8,
                help="Total bilirubin"
            )
        
        if st.button("🔬 Predict Risk", key="predict_manual", use_container_width=True):
            make_prediction(model, scaler, {
                'wbc_count': wbc_count,
                'temperature': temperature,
                'heart_rate': heart_rate,
                'respiratory_rate': respiratory_rate,
                'lactate': lactate,
                'glucose': glucose,
                'platelet_count': platelet_count,
                'bilirubin': bilirubin
            })
    
    with tab2:
        st.subheader("Upload Patient Data (CSV)")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("🔬 Predict for All Patients", use_container_width=True):
                    # Prepare data
                    X = scaler.transform(df)
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Display results
                    results_df = df.copy()
                    results_df['Risk_Level'] = predictions
                    if probabilities is not None:
                        results_df['Risk_Score'] = probabilities
                    
                    st.success("✅ Predictions completed!")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab3:
        st.subheader("Test with Sample Data")
        
        # Create sample patient data
        sample_data = {
            'wbc_count': [15.2, 4.5, 8.3],
            'temperature': [39.5, 36.8, 38.2],
            'heart_rate': [120, 72, 95],
            'respiratory_rate': [28, 16, 20],
            'lactate': [4.5, 1.2, 2.8],
            'glucose': [250, 100, 180],
            'platelet_count': [80, 250, 200],
            'bilirubin': [3.2, 0.7, 1.5]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        st.write("Sample patients:")
        st.dataframe(sample_df, use_container_width=True)
        
        if st.button("🔬 Predict for Samples", use_container_width=True):
            X = scaler.transform(sample_df)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results_df = sample_df.copy()
            results_df['Prediction'] = ['🔴 High Risk' if p == 1 else '🟢 Low Risk' for p in predictions]
            if probabilities is not None:
                results_df['Risk_Score'] = [f"{prob:.2%}" for prob in probabilities]
            
            st.success("✅ Predictions completed!")
            st.dataframe(results_df, use_container_width=True)

def make_prediction(model, scaler, patient_data):
    """Make prediction for single patient"""
    try:
        # Prepare data
        input_array = np.array([[
            patient_data['wbc_count'],
            patient_data['temperature'],
            patient_data['heart_rate'],
            patient_data['respiratory_rate'],
            patient_data['lactate'],
            patient_data['glucose'],
            patient_data['platelet_count'],
            patient_data['bilirubin']
        ]])
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_scaled)[0]
            risk_score = probability[1]
        else:
            risk_score = None
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="high-risk">
                    <h2 style="color: red; margin: 0;">⚠️ HIGH RISK</h2>
                    <p style="margin: 10px 0; font-size: 1.2rem;">Sepsis Likely</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="low-risk">
                    <h2 style="color: green; margin: 0;">✅ LOW RISK</h2>
                    <p style="margin: 10px 0; font-size: 1.2rem;">Sepsis Unlikely</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if risk_score is not None:
                st.metric("Risk Score", f"{risk_score:.2%}")
            
            # Clinical recommendations
            st.markdown("### 💊 Recommendations")
            if prediction == 1:
                st.warning("""
                - **Immediate medical attention required**
                - Consider sepsis protocol initiation
                - Blood cultures and lactate testing
                - Broad-spectrum antibiotics consideration
                - ICU monitoring recommended
                """)
            else:
                st.info("""
                - Continue routine monitoring
                - Regular vital signs checks
                - Follow-up as clinically indicated
                """)
        
        # Patient data summary
        st.markdown("### 📋 Input Data Summary")
        summary_df = pd.DataFrame({
            'Parameter': list(patient_data.keys()),
            'Value': list(patient_data.values())
        })
        st.dataframe(summary_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

def about_page():
    """About page"""
    st.header("📊 About This System")
    
    st.markdown("""
    ## Overview
    This Blood Infection (Sepsis) Prediction System uses machine learning to assess 
    the risk of sepsis based on patient vital signs and laboratory values.
    
    ## Model Information
    - **Model Type**: Random Forest Classifier / Logistic Regression
    - **Training Data**: Medical sepsis dataset with clinical indicators
    - **Purpose**: Early detection and risk stratification
    
    ## Features Used
    1. **WBC Count**: White blood cell count indicator
    2. **Temperature**: Body temperature
    3. **Heart Rate**: Cardiovascular response
    4. **Respiratory Rate**: Pulmonary function
    5. **Lactate**: Tissue perfusion indicator
    6. **Glucose**: Metabolic status
    7. **Platelet Count**: Coagulation status
    8. **Bilirubin**: Hepatic function
    
    ## Clinical Significance
    Sepsis is a life-threatening condition requiring immediate medical intervention.
    Early detection and appropriate treatment significantly improve outcomes.
    
    ## Disclaimer
    ⚠️ **This tool is for educational and research purposes only.**
    - Not for clinical decision-making without professional review
    - Should complement, not replace, clinical judgment
    - Always consult qualified healthcare professionals
    """)

def instructions_page():
    """Instructions page"""
    st.header("ℹ️ How to Use")
    
    st.markdown("""
    ## Quick Start Guide
    
    ### Step 1: Navigate to Prediction Page
    Click on "🔍 Prediction" in the sidebar
    
    ### Step 2: Choose Input Method
    
    **Option A: Manual Input**
    - Enter patient vital signs and lab values
    - Click "🔬 Predict Risk"
    - View risk assessment and recommendations
    
    **Option B: Upload CSV**
    - Prepare CSV with columns: wbc_count, temperature, heart_rate, etc.
    - Upload the file
    - Predictions for all patients will be generated
    - Download results as CSV
    
    **Option C: Use Sample Data**
    - Test the system with pre-loaded sample patients
    - Click "🔬 Predict for Samples"
    
    ### Step 3: Interpret Results
    
    **High Risk (🔴)**
    - Sepsis indicators present
    - Immediate medical attention recommended
    - Follow protocol initiation steps
    
    **Low Risk (🟢)**
    - Sepsis markers within normal range
    - Continue routine monitoring
    """)

if __name__ == "__main__":
    main()