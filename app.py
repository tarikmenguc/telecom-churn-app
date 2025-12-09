import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. SAYFA AYARLARI VE MODERN CSS ---
st.set_page_config(
    page_title="ChurnAI - Müşteri Analizi",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile arayüzü güzelleştirme
st.markdown("""
<style>
    /* Ana arka planı hafif gri yap */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Metrik kutularını kart gibi göster */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Buton stili */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        height: 3.5em;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Başlık stili */
    h1 {
        color: #182848;
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. YARDIMCI FONKSİYONLAR ---
@st.cache_resource
def load_helpers():
    try:
        model = joblib.load('models/churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        columns = joblib.load('models/columns.pkl')
        return model, scaler, columns
    except FileNotFoundError:
        st.error("⚠️ Model dosyaları bulunamadı! 'models' klasörünü kontrol edin.")
        return None, None, None

model, scaler, model_columns = load_helpers()

if model is None:
    st.stop()

# --- 3. ANA BAŞLIK ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/8654/8654298.png", width=80)
with col_title:
    st.title("Telekom Churn Tahminleyicisi")
    st.markdown("**Yapay Zeka Destekli Müşteri Kayıp Analiz Paneli**")

st.markdown("---")

# --- 4. SOL MENÜ (INPUT ALANI) ---
with st.sidebar:
    st.header("👤 Müşteri Profili")
    
    # Sekmeli Giriş Yapısı (Daha Derli Toplu)
    tab1, tab2, tab3 = st.tabs(["Kimlik", "Hizmet", "Finans"])
    
    with tab1:
        gender = st.radio("Cinsiyet", ["Erkek", "Kadın"], horizontal=True)
        senior = st.toggle("65 Yaş Üstü mü?")
        partner = st.toggle("Evli/Partneri Var")
        dependents = st.toggle("Bakmakla Yükümlü Olduğu Kişi Var")
        
    with tab2:
        tenure = st.slider("Abonelik (Ay)", 0, 72, 24)
        phone_service = st.checkbox("Telefon Hizmeti", value=True)
        multiple_lines = st.selectbox("Hat Tipi", ["Tek Hat", "Çoklu Hat", "Hizmet Yok"])
        internet_service = st.selectbox("İnternet", ["Fiber Optik", "DSL", "Yok"])
        
        st.caption("Ekstra Servisler")
        extras = st.multiselect(
            "Seçiniz:",
            ['Online Güvenlik', 'Yedekleme', 'Cihaz Koruma', 'Teknik Destek', 'TV', 'Film'],
            default=['Online Güvenlik']
        )
        
    with tab3:
        contract = st.selectbox("Sözleşme", ["Aylık", "1 Yıllık", "2 Yıllık"])
        paperless = st.checkbox("Kağıtsız Fatura", value=True)
        payment_method = st.selectbox("Ödeme", 
                                      ["Elektronik Çek", "Posta Çeki", "Banka Transferi", "Kredi Kartı"])
        monthly_charges = st.number_input("Aylık Fatura ($)", 18.0, 150.0, 70.0)
        total_charges = st.number_input("Toplam Ödeme ($)", 0.0, 10000.0, 1500.0)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analizi Başlat ⚡")


# --- 5. TAHMİN VE GÖRSELLEŞTİRME ---
if analyze_btn:
    
    # -- Veri Hazırlığı (Aynı Mantık) --
    input_data = {}
    input_data['gender'] = 1 if gender == "Erkek" else 0
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
            
    # Kategorik İşlemler
    if multiple_lines == "Çoklu Hat": df_input.loc[0, 'MultipleLines_Yes'] = 1
    elif multiple_lines == "Hizmet Yok": df_input.loc[0, 'MultipleLines_No phone service'] = 1
    
    if internet_service == "Fiber Optik": df_input.loc[0, 'InternetService_Fiber optic'] = 1
    elif internet_service == "Yok": df_input.loc[0, 'InternetService_No'] = 1
    
    mapping = {'Online Güvenlik':'OnlineSecurity_Yes', 'Yedekleme':'OnlineBackup_Yes',
               'Cihaz Koruma':'DeviceProtection_Yes', 'Teknik Destek':'TechSupport_Yes',
               'TV':'StreamingTV_Yes', 'Film':'StreamingMovies_Yes'}
    for item in extras:
        if item in mapping: df_input.loc[0, mapping[item]] = 1
            
    if contract == "1 Yıllık": df_input.loc[0, 'Contract_One year'] = 1
    elif contract == "2 Yıllık": df_input.loc[0, 'Contract_Two year'] = 1
    
    if payment_method == "Elektronik Çek": df_input.loc[0, 'PaymentMethod_Electronic check'] = 1
    elif payment_method == "Posta Çeki": df_input.loc[0, 'PaymentMethod_Mailed check'] = 1
    elif payment_method == "Kredi Kartı": df_input.loc[0, 'PaymentMethod_Credit card (automatic)'] = 1
    
    if tenure <= 12: df_input.loc[0, 'Tenure_Group_Yeni_Musteri'] = 1
    elif tenure <= 48: df_input.loc[0, 'Tenure_Group_Sadik_Musteri'] = 1

    # -- Tahmin --
    try:
        input_scaled = scaler.transform(df_input)
        probability = model.predict_proba(input_scaled)[0][1]
        
        # --- DASHBOARD GÖRÜNÜMÜ ---
        
        # Kolonlara Böl: Sol taraf Özet, Sağ Taraf Grafik
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.subheader("📋 Sonuç Kartı")
            if probability > 0.5:
                st.error("RİSKLİ MÜŞTERİ")
                st.metric("Terk Etme İhtimali", f"%{probability*100:.1f}", delta="-Riskli", delta_color="inverse")
                st.markdown("**Öneri:** Acil olarak indirim teklif edilmeli veya müşteri temsilcisi aramalı.")
            else:
                st.success("SADIK MÜŞTERİ")
                st.metric("Terk Etme İhtimali", f"%{probability*100:.1f}", delta="+Güvenli")
                st.markdown("**Öneri:** Sadakat programına dahil edilebilir.")
                
            st.info(f"**Tahmini Kayıp:** ${monthly_charges * 12:.2f} / Yıl")

        with col_res2:
            # GAUGE CHART (İbreli Gösterge)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Metre"},
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

        # Veri Detayı (Açılır Kapanır)
        with st.expander("🔍 Modelin Kullandığı Ham Veriyi İncele"):
            st.dataframe(df_input.style.highlight_max(axis=0))

    except Exception as e:
        st.error(f"Hata oluştu: {e}")

else:
    # Sayfa ilk açıldığında boş kalmasın diye karşılama mesajı
    st.info("👈 Analiz yapmak için sol menüden müşteri bilgilerini girip 'Analizi Başlat' butonuna basın.")