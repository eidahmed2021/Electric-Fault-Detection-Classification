import joblib
import streamlit as st
import pandas as pd
st.set_page_config(page_title="Electric Fault Detection ", page_icon="⚡")

st.title('Electric Fault Detection & Classification System')

# Input fields
st.subheader('Input Measurements')
Ia = st.number_input('Phase A Current (Ia)', format="%.6f", step=0.000001)
Ib = st.number_input('Phase B Current (Ib)', format="%.6f", step=0.000001)
Ic = st.number_input('Phase C Current (Ic)', format="%.6f", step=0.000001)
Va = st.number_input('Phase A Voltage (Va)', format="%.6f", step=0.000001)
Vb = st.number_input('Phase B Voltage (Vb)', format="%.6f", step=0.000001)
Vc = st.number_input('Phase C Voltage (Vc)', format="%.6f", step=0.000001)

# Centered button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button('Analyze System')

if predict_btn:
    model = joblib.load('xgboost_model_binaryclass.pkl')
    model2 = joblib.load('xgboost_model_multiclass.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    # Multiply voltages by 11000 if required by your model
    df = pd.DataFrame({
        'Ia': [Ia],
        'Ib': [Ib],
        'Ic': [Ic],
        'Va': [Va * 11000],
        'Vb': [Vb * 11000],
        'Vc': [Vc * 11000]
    })
    # Normalize input for both models
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    pred = model.predict(df_scaled)

    # --- Sidebar organization starts here ---
    st.sidebar.title("📝 Analysis Result")
    if pred[0] == 0:
        st.sidebar.success('No Fault Detected')
    else:
        st.sidebar.error('Fault Detected')
        st.sidebar.markdown("---")
        pred2 = model2.predict(df_scaled)
        fault_classes = {
            0: ("Single Line-to-Ground Fault", "LG FAULT"),
            1: ("Line-to-Line Fault", "LL Fault"),
            2: ("Line-to-Line-to-Ground Fault", "LLG Fault"),
            3: ("Line-to-Line-to-Line Fault", "LLL Fault"),
            4: ("No Fault", "No Fault"),
        }
        if pred2[0] in fault_classes:
            fault_name, fault_desc = fault_classes[pred2[0]]
            st.sidebar.subheader(f"Fault Type: {fault_name}")
            st.sidebar.info(f"Description: {fault_desc}")
    # --- Sidebar organization ends here ---

st.markdown("---")
st.caption("Electric Fault Detection System | Powered by XGBoost")