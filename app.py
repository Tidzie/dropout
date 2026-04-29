import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Dropout Predictor - GZU",
    page_icon="🎓",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('student_dropout_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# Sidebar / Header
st.title("🎓 Great Zimbabwe University")
st.subheader("Student Dropout Prediction Dashboard")
st.markdown("---")

if model is None:
    st.error("Model files not found. Please run the Jupyter Notebook first to generate 'student_dropout_model.joblib' and 'scaler.joblib'.")
else:
    # Form for student details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Demographic Info")
        age = st.number_input("Age", min_value=16, max_value=60, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["Urban", "Rural"])
        part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])

    with col2:
        st.write("### Socio-Economic Factors")
        family_income = st.number_input("Family Monthly Income ($)", min_value=0.0, value=500.0)
        fees_paid = st.selectbox("Fees Payment Status", ["Full", "Partial", "None"])
        internet_access = st.selectbox("Internet Access", ["Good", "Moderate", "Poor"])
        electricity = st.selectbox("Electricity Reliability", ["High", "Medium", "Low"])
        transport_time = st.number_input("Transport Time (mins)", min_value=0.0, value=30.0)

    with col3:
        st.write("### Academic & Behavioral")
        previous_grade = st.number_input("Previous Grade (GPA)", min_value=0.0, max_value=4.0, value=2.5)
        attendance_rate = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0)
        study_hours = st.slider("Study Hours (weekly)", 0.0, 40.0, 15.0)
        lms_logins = st.number_input("LMS Logins (weekly)", min_value=0, value=5)
        stress_level = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0)

    # Preprocessing Input
    if st.button("Predict Student Status", use_container_width=True):
        # Encoding (Mapping to match LabelEncoder during training)
        # Note: In a production environment, you should save/load the LabelEncoders
        # Here we manually map based on standard label encoding (alphabetical)
        gender_map = {"Female": 0, "Male": 1}
        location_map = {"Rural": 0, "Urban": 1}
        internet_map = {"Good": 0, "Moderate": 1, "Poor": 2}
        electricity_map = {"High": 0, "Low": 1, "Medium": 2}
        fees_map = {"Full": 0, "None": 1, "Partial": 2}
        job_map = {"No": 0, "Yes": 1}

        # Create input dict
        input_data = {
            'age': age,
            'gender': gender_map[gender],
            'location': location_map[location],
            'family_income': family_income,
            'internet_access': internet_map[internet_access],
            'electricity_reliability': electricity_map[electricity],
            'transport_time': transport_time,
            'study_hours': study_hours,
            'attendance_rate': attendance_rate,
            'lms_logins': lms_logins,
            'previous_grade': previous_grade,
            'fees_paid': fees_map[fees_paid],
            'part_time_job': job_map[part_time_job],
            'stress_level': stress_level
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Feature Engineering (MUST MATCH TRAINING)
        input_df['academic_score'] = (input_df['study_hours'] * 0.3) + (input_df['attendance_rate'] * 0.4) + (input_df['previous_grade'] * 0.3)
        input_df['engagement_score'] = input_df['lms_logins'] * input_df['study_hours']

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        # Results Display
        st.markdown("---")
        if prediction == 1:
            st.error(f"### Prediction: HIGH RISK OF DROPOUT")
            st.warning(f"Probability of dropout: {probability:.2%}")
        else:
            st.success(f"### Prediction: STUDENT LIKELY TO CONTINUE")
            st.info(f"Probability of dropout: {probability:.2%}")

        # Metrics Visualization
        st.write("### Risk Indicators")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Engagement Score", f"{input_df['engagement_score'][0]:.1f}")
        col_m2.metric("Academic Score", f"{input_df['academic_score'][0]:.1f}")
        col_m3.metric("Financial Risk", "High" if fees_paid == "None" else "Low")

st.markdown("---")
st.caption("Developed for HCS221 Assignment - Great Zimbabwe University")
