import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Electric Fault Detection", 
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0066cc;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>⚡ Electric Fault Detection & Classification System</h1>", unsafe_allow_html=True)

# Information about the system
with st.expander("ℹ️ About This System"):
    st.markdown("""
    This system uses machine learning to detect and classify electrical faults in three-phase power systems.
    
    **Types of faults detected:**
    - Single Line-to-Ground Fault (LG)
    - Line-to-Line Fault (LL)
    - Line-to-Line-to-Ground Fault (LLG)
    - Line-to-Line-to-Line Fault (LLL)
    
    **How to use:**
    1. Enter the current and voltage measurements for each phase
    2. each model has own data that we have trained
    3. i recommond to debend on the model that you already enter a associated data related to the model
    4. Show the result of the other model that i try to distiguish between the two models 
    3. Click "Analyze System" to detect and classify any faults
    
    **Data Source:**
    The models are trained on standardized electrical fault data with specific characteristics.
    For accurate results, input values should be within the expected ranges for three-phase systems

    you can visit this to get more sample "https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification" .
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='subheader'>Input Measurements</h2>", unsafe_allow_html=True)
    
    # Add sample data option
    with st.expander("📊 Load Sample Data"):
        st.markdown("""
        You can load sample data from our dataset to test the system.
        Each sample represents different fault conditions or normal operation.
        """)
        
        sample_tab1, sample_tab2 = st.tabs(["Binary Model Samples", "Multiclass Model Samples"])
        
        with sample_tab1:
            binary_sample_option = st.selectbox(
                "Select binary model sample data:",
                ["Custom Input", "Normal Operation", "Fault Sample"],
                key="binary_sample"
            )
            
            if st.button("Load Binary Sample", key="load_binary"):
                # Sample data for binary model
                binary_sample_data = {
                    "Normal Operation": {"Ia":6.117669408, "Ib": -41.992780330, "Ic": 38.07057233, "Va": 0.598112442, "Vb": -0.308307458, "Vc":-0.289804985},
                    "Fault Sample": {"Ia": -642.4702436, "Ib": 271.3606456, "Ic": 373.2959406, "Va": 0.029474875, "Vb": 0.016372292, "Vc": -0.045847167},
                }
                # Set the values in the input fields
                if binary_sample_option in binary_sample_data:
                    st.session_state.Ia = binary_sample_data[binary_sample_option]["Ia"]
                    st.session_state.Ib = binary_sample_data[binary_sample_option]["Ib"]
                    st.session_state.Ic = binary_sample_data[binary_sample_option]["Ic"]
                    st.session_state.Va = binary_sample_data[binary_sample_option]["Va"]
                    st.session_state.Vb = binary_sample_data[binary_sample_option]["Vb"]
                    st.session_state.Vc = binary_sample_data[binary_sample_option]["Vc"]
                    st.success(f"Loaded {binary_sample_option} data for binary model")
        
        with sample_tab2:
            multiclass_sample_option = st.selectbox(
                "Select multiclass model sample data:",
                ["Custom Input", "Normal Operation", "LG Fault Sample", "LL Fault Sample", "LLG Fault Sample", "LLL Fault Sample"],
                key="multiclass_sample"
            )
            
            if st.button("Load Multiclass Sample", key="load_multiclass"):
                # Sample data for multiclass model
                multiclass_sample_data = {
                    "Normal Operation": {"Ia": 48.04167823, "Ib": -23.40110771, "Ic": 21.27306551, "Va": 0.367340759, "Vb": -0.56425701, "Vc": 0.196916251},
                    "LG Fault Sample": {"Ia": 183.8354578, "Ib": 57.52147714, "Ic": -48.42209166, "Va": -0.382138071, "Vb": 0.512008407, "Vc": -0.129870336},
                    "LL Fault Sample": {"Ia": 32.8837449, "Ib": -786.9246198, "Ic": 756.6458205, "Va": -0.089237547, "Vb": 0.008301792, "Vc": 0.080935755},
                    "LLG Fault Sample": {"Ia": 804.5618171, "Ib": -124.8529005, "Ic": 34.05499735, "Va":0.032684951, "Vb": -0.362137169, "Vc":0.329452218},
                    "LLL Fault Sample": {"Ia": -380.7135649, "Ib": -516.3174483, "Ic": 899.1999955, "Va": -0.040945999, "Vb": 0.019293298, "Vc":0.021652701}
                }
                # Set the values in the input fields
                if multiclass_sample_option in multiclass_sample_data:
                    st.session_state.Ia = multiclass_sample_data[multiclass_sample_option]["Ia"]
                    st.session_state.Ib = multiclass_sample_data[multiclass_sample_option]["Ib"]
                    st.session_state.Ic = multiclass_sample_data[multiclass_sample_option]["Ic"]
                    st.session_state.Va = multiclass_sample_data[multiclass_sample_option]["Va"]
                    st.session_state.Vb = multiclass_sample_data[multiclass_sample_option]["Vb"]
                    st.session_state.Vc = multiclass_sample_data[multiclass_sample_option]["Vc"]
                    st.success(f"Loaded {multiclass_sample_option} data for multiclass model")
    
    # Create columns for better layout of inputs
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**Current Measurements (A)**")
        Ia = st.number_input('Phase A Current (Ia)', format="%.6f", step=0.000001, key="Ia")
        Ib = st.number_input('Phase B Current (Ib)', format="%.6f", step=0.000001, key="Ib")
        Ic = st.number_input('Phase C Current (Ic)', format="%.6f", step=0.000001, key="Ic")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with input_col2:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**Voltage Measurements (V)**")
        Va = st.number_input('Phase A Voltage (Va)', format="%.6f", step=0.000001, key="Va")
        Vb = st.number_input('Phase B Voltage (Vb)', format="%.6f", step=0.000001, key="Vb")
        Vc = st.number_input('Phase C Voltage (Vc)', format="%.6f", step=0.000001, key="Vc")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Centered button
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        predict_btn = st.button('Analyze System')

with col2:
    st.markdown("<h2 class='subheader'>System Visualization</h2>", unsafe_allow_html=True)
    
    # Placeholder for visualization
    if 'Ia' in locals() and 'Ib' in locals() and 'Ic' in locals():
        # Create a simple visualization of the three-phase system
        fig = go.Figure()
        
        # Create time axis for sine waves
        t = np.linspace(0, 2*np.pi, 100)
        
        # Add current waveforms
        fig.add_trace(go.Scatter(x=t, y=np.sin(t) * (Ia if Ia else 1), 
                                mode='lines', name='Phase A Current', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=t, y=np.sin(t - 2*np.pi/3) * (Ib if Ib else 1), 
                                mode='lines', name='Phase B Current', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=t, y=np.sin(t + 2*np.pi/3) * (Ic if Ic else 1), 
                                mode='lines', name='Phase C Current', line=dict(color='green')))
        
        fig.update_layout(
            title='Three-Phase Current Waveforms',
            xaxis_title='Time',
            yaxis_title='Current (A)',
            legend=dict(x=0, y=1),
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Analysis logic - SEPARATED FOR EACH MODEL
if predict_btn:
    # Load models and scaler
    binary_model = joblib.load('Random Forest_model_binaryclass.pkl')
    multiclass_model = joblib.load('Random Forest_model_multiclass.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    
    # Prepare data
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
    
    # Create two columns in the sidebar for the models
    st.sidebar.title("📝 Analysis Results")
    binary_col, multiclass_col = st.sidebar.columns(2)
    
    # Binary Model Section
    with binary_col:
        st.markdown("<h3>Binary Fault Detection</h3>", unsafe_allow_html=True)
        
        # Get binary model prediction
        binary_pred = binary_model.predict(df_scaled)
        
        if binary_pred[0] == 0:
            st.success('No Fault Detected')
            st.markdown("The binary classifier indicates normal operation.")
        else:
            st.error('Fault Detected')
            st.markdown("The binary classifier indicates a fault condition.")
    
    # Multiclass Model Section
    with multiclass_col:
        st.markdown("<h3>Fault Classification</h3>", unsafe_allow_html=True)
        
        # Get multiclass model prediction
        multiclass_pred = multiclass_model.predict(df_scaled)
        
        fault_classes = {
            0: ("LG Fault", "Single Line-to-Ground"),
            1: ("LL Fault", "Line-to-Line"),
            2: ("LLG Fault", "Line-to-Line-to-Ground"),
            3: ("LLL Fault", "Line-to-Line-to-Line"),
            4: ("No Fault", "Normal Operation"),
        }
        
        fault_code, (fault_name, fault_desc) = multiclass_pred[0], fault_classes[multiclass_pred[0]]
        
        if fault_code == 4:
            st.success(f'Type: {fault_name}')
        else:
            st.error(f'Type: {fault_name}')
        
        st.info(f'Description: {fault_desc}')
    
    # Combined Analysis Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Combined Analysis")
    
    # Special case: Binary says no fault (0) but multiclass identifies a fault (not 4)
    if binary_pred[0] == 0 and multiclass_pred[0] != 4:
        st.sidebar.warning('System Status: Potential Fault Detected')
        st.sidebar.info(f"Fault Type: {fault_classes[multiclass_pred[0]][0]}")
        st.sidebar.info(f"Note: Binary model indicates normal operation, but classification model detected a fault.")
    # Normal case: Both models agree on no fault
    elif binary_pred[0] == 0 or multiclass_pred[0] == 4:
        st.sidebar.success('System Status: Normal Operation')
    # Normal case: Both models agree there is a fault
    else:
        st.sidebar.error('System Status: Fault Detected')
        
        # Only show detailed fault info if multiclass doesn't say "No Fault"
        if multiclass_pred[0] != 4:
            st.sidebar.info(f"Fault Type: {fault_classes[multiclass_pred[0]][0]}")
            st.sidebar.info(f"Description: {fault_classes[multiclass_pred[0]][1]}")

# Footer
st.markdown("---")
st.markdown("<p class='footer'>Electric Fault Detection System | Powered by XGBoost | © 2025</p>", unsafe_allow_html=True)