import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. PAGE SETTINGS AND MODERN CSS ---
st.set_page_config(
    page_title="ChurnAI - Customer Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautify the UI with custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Metric boxes as modern cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eef2f5;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    
    /* Premium Button style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        color: white;
        border: none;
        height: 3.5em;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
        color: #f8f9fa;
    }
    
    /* Typography improvements */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Subtly style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eef2f5;
    }
    
    /* Footer style */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---
@st.cache_resource
def load_helpers():
    try:
        model = joblib.load('models/churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        columns = joblib.load('models/columns.pkl')
        return model, scaler, columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Check the 'models' folder.")
        return None, None, None

model, scaler, model_columns = load_helpers()

if model is None:
    st.stop()

# --- 3. MAIN HEADER ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/8654/8654298.png", width=80)
with col_title:
    st.title("Telecom Churn Predictor")
    st.markdown("**AI-Powered Customer Churn Analysis Panel**")

st.markdown("---")

# --- 4. LEFT MENU (INPUT AREA) ---
with st.sidebar:
    st.header("👤 Customer Profile")
    
    # Tabbed Input Structure (More Organized)
    tab1, tab2, tab3 = st.tabs(["Identity", "Service", "Finance"])
    
    with tab1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = st.toggle("Senior Citizen?", help="Is the customer 65 years or older?")
        partner = st.toggle("Married/Has Partner")
        dependents = st.toggle("Has Dependents", help="Does the customer have dependents (children, elders etc.)?")
        
    with tab2:
        tenure = st.slider("Tenure (Months)", 0, 72, 24, help="Number of months the customer has stayed with the company.")
        phone_service = st.checkbox("Phone Service", value=True)
        multiple_lines = st.selectbox("Line Type", ["Single Line", "Multiple Lines", "No Service"])
        internet_service = st.selectbox("Internet", ["Fiber Optic", "DSL", "None"])
        
        st.caption("Extra Services")
        extras = st.multiselect(
            "Select:",
            ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies'],
            default=['Online Security']
        )
        
    with tab3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.checkbox("Paperless Billing", value=True)
        payment_method = st.selectbox("Payment Method", 
                                      ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Start Analysis ⚡")


# --- 5. PREDICTION AND VISUALIZATION ---
if analyze_btn:
    
    # -- Data Preparation --
    input_data = {}
    input_data['gender'] = 1 if gender == "Male" else 0
    input_data['SeniorCitizen'] = 1 if senior else 0
    input_data['Partner'] = 1 if partner else 0
    input_data['Dependents'] = 1 if dependents else 0
    input_data['PhoneService'] = 1 if phone_service else 0
    input_data['PaperlessBilling'] = 1 if paperless else 0
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    input_data['Has_Family'] = 1 if (partner or dependents) else 0
    input_data['Service_Count'] = len(extras)
    
    df_input = pd.DataFrame(columns=model_columns)
    df_input.loc[0] = 0
    for col in input_data:
        if col in df_input.columns: df_input.loc[0, col] = input_data[col]
            
    # Categorical Processing
    if multiple_lines == "Multiple Lines": df_input.loc[0, 'MultipleLines_Yes'] = 1
    elif multiple_lines == "No Service": df_input.loc[0, 'MultipleLines_No phone service'] = 1
    
    if internet_service == "Fiber Optic": df_input.loc[0, 'InternetService_Fiber optic'] = 1
    elif internet_service == "None": df_input.loc[0, 'InternetService_No'] = 1
    
    mapping = {'Online Security':'OnlineSecurity_Yes', 'Online Backup':'OnlineBackup_Yes',
               'Device Protection':'DeviceProtection_Yes', 'Tech Support':'TechSupport_Yes',
               'Streaming TV':'StreamingTV_Yes', 'Streaming Movies':'StreamingMovies_Yes'}
    for item in extras:
        if item in mapping: df_input.loc[0, mapping[item]] = 1
            
    if contract == "One year": df_input.loc[0, 'Contract_One year'] = 1
    elif contract == "Two year": df_input.loc[0, 'Contract_Two year'] = 1
    
    if payment_method == "Electronic Check": df_input.loc[0, 'PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed Check": df_input.loc[0, 'PaymentMethod_Mailed check'] = 1
    elif payment_method == "Credit Card": df_input.loc[0, 'PaymentMethod_Credit card (automatic)'] = 1
    
    if tenure <= 12: df_input.loc[0, 'Tenure_Group_Yeni_Musteri'] = 1
    elif tenure <= 48: df_input.loc[0, 'Tenure_Group_Sadik_Musteri'] = 1

    # -- Prediction --
    try:
        input_scaled = scaler.transform(df_input)
        probability = model.predict_proba(input_scaled)[0][1]
        
        # --- DASHBOARD VIEW ---
        
        # Divide into Columns: Left side Summary, Right Side Chart
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.subheader("📋 Result Card")
            if probability > 0.5:
                st.error("HIGH RISK CUSTOMER")
                st.metric("Churn Probability", f"{probability*100:.1f}%", delta="-Risk", delta_color="inverse")
                st.markdown("**Recommendation:** Immediate discount offer or customer representative call needed.")
            else:
                st.success("LOYAL CUSTOMER")
                st.metric("Churn Probability", f"{probability*100:.1f}%", delta="+Safe")
                st.markdown("**Recommendation:** Can be included in the loyalty program.")
                
            st.info(f"**Estimated Loss:** ${monthly_charges * 12:.2f} / Year")

        with col_res2:
            # GAUGE CHART
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Meter"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#FF4B4B" if probability > 0.5 else "#00CC96"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 204, 150, 0.3)'},
                        {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(255, 75, 75, 0.3)'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100}}))
            
            st.plotly_chart(fig, use_container_width=True)

        # Raw Data Details (Expandable)
        with st.expander("🔍 Inspect Raw Data Used by the Model"):
            st.dataframe(df_input.style.highlight_max(axis=0))

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    # Welcome message when the page first opens
    st.info("👈 Please enter customer information from the left menu and click 'Start Analysis' to perform an analysis.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <hr style="margin-bottom: 10px; opacity: 0.5;">
    Developed for demonstrating Machine Learning & Business Intelligence integration.<br>
    <em>Ready for production and scale.</em>
</div>
""", unsafe_allow_html=True)