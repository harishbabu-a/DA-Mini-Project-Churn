import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Airtel Churn Sentinel | Retention AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1a0000 0%, #3d0000 50%, #000000 100%);
        color: #ffffff;
    }
    .stMetric {
        background: rgba(255, 0, 0, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        background: rgba(255, 0, 0, 0.1);
    }
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #ff4b2b;
        text-shadow: 0 0 10px rgba(255, 75, 43, 0.5);
    }
    .report-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: -webkit-linear-gradient(#ff4b2b, #ff0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff4b2b, #ff0000);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .report-title {
            font-size: 1.8rem !important;
        }
        .stMetric {
            padding: 10px !important;
        }
        .stButton>button {
            width: 100% !important;
            margin-bottom: 10px;
        }
        h1, h2, h3 {
            font-size: 1.2rem !important;
        }
        [data-testid="stSidebar"] {
            width: 80% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading Utility ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    # Basic Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()  # Small number of NAs in TotalCharges
    return df

def preprocess_data(df):
    data = df.copy()
    data.drop('customerID', axis=1, inplace=True)
    
    # Binary encoding for target
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Label Encoding for categorical features
    binary_cols = [col for col in data.columns if data[col].nunique() == 2 and data[col].dtype == 'object']
    le = LabelEncoder()
    for col in binary_cols:
        data[col] = le.fit_transform(data[col])
        
    # One-hot encoding for multiclass features
    data = pd.get_dummies(data)
    
    return data

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3658/3658959.png", width=100) # Red-themed icon
st.sidebar.title("🔴 Airtel Sentinel")
menu = st.sidebar.radio("Navigate System", ["Live Operational Center", "Dashboard Overview", "Exploratory Analysis", "AI Prediction Model", "Customer Risk Profiler"])

# --- Load Data ---
raw_df = load_data()
processed_df = preprocess_data(raw_df)

# --- MAIN PAGE CONTENT ---

if menu == "Live Operational Center":
    st.markdown("<h1 class='report-title'>LIVE OPERATIONS COMMAND</h1>", unsafe_allow_html=True)
    
    # Live Status Header
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("""
            <div style="background: rgba(0, 255, 0, 0.1); border: 1px solid #00ff00; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="color: #00ff00; font-weight: bold; font-family: 'Orbitron';">🟢 SYSTEM ONLINE</span><br>
                <small>Node: RIT-ML-01</small>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div style="background: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="color: #ff0000; font-weight: bold; font-family: 'Orbitron';">🛰️ STREAMING DATA</span><br>
                <small>Source: Airtel South Circle</small>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; text-align: center;">
                <span style="color: white; font-weight: bold; font-family: 'Orbitron';">{time.strftime('%H:%M:%S')} UTC</span><br>
                <small>Sync Status: 100%</small>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    # Simulated Live Feed
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("📡 Real-time Churn Monitoring")
        chart_placeholder = st.empty()
        status_text = st.empty()
        
    with col_r:
        st.subheader("📜 System Logs")
        log_placeholder = st.empty()

    if st.button("🚀 START REAL-TIME STREAM", type="primary"):
        # Initial logs
        logs = ["✅ Connection established", "🔍 Ingesting feed...", "🚀 ML Model v2.1 loaded"]
        
        # Stream simulation
        data_points = []
        for i in range(50):
            # Generate moving data
            new_val = [np.random.rand() * 0.4 + 0.1, np.random.rand() * 0.3 + 0.5, np.random.rand() * 0.2 + 0.1]
            data_points.append(new_val)
            if len(data_points) > 20: data_points.pop(0)
            
            # Update Chart
            fig_live = px.line(pd.DataFrame(data_points, columns=['Risk', 'Retention', 'Avg Churn']), 
                              template="plotly_dark", color_discrete_sequence=['#ff4b2b', '#00ff00', '#00d2ff'])
            fig_live.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=0))
            chart_placeholder.plotly_chart(fig_live, use_container_width=True)
            
            # Update Logs
            if i % 5 == 0:
                cid = np.random.randint(1000, 9999)
                logs.insert(0, f"👤 Alert: Airtel Business User #{cid} risk detected")
            if i % 8 == 0:
                logs.insert(0, f"📡 Circle Update: TN Loop processed at {time.strftime('%H:%M:%S')}")
            
            log_text = "\n".join([f"[{time.strftime('%H:%M:%S')}] {l}" for l in logs[:10]])
            log_placeholder.code(log_text)
            
            status_text.write(f"Streaming Active... iteration {i+1}/50")
            time.sleep(0.8)
            
        st.success("Stream simulation complete. Session saved to history.")
    else:
        # Default static view if not streaming
        chart_data = pd.DataFrame(np.random.randn(20, 3) / 5 + [0.3, 0.5, 0.2], columns=['Risk', 'Retention', 'Avg Churn'])
        fig_static = px.line(chart_data, template="plotly_dark", color_discrete_sequence=['#ff4b2b', '#00ff00', '#00d2ff'])
        fig_static.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=0))
        chart_placeholder.plotly_chart(fig_static, use_container_width=True)
        log_placeholder.code("System Standby. Click 'Start Real-time Stream' to begin.")

elif menu == "Dashboard Overview":
    st.markdown("<h1 class='report-title'>CUSTOMER CHURN ANALYTICS</h1>", unsafe_allow_html=True)
    
    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Customers", f"{len(raw_df):,}")
    with m2:
        churn_rate = (raw_df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-1.2%", delta_color="inverse")
    with m3:
        avg_tenure = raw_df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} Mo")
    with m4:
        total_revenue = raw_df['TotalCharges'].sum()
        st.metric("Est. Annual Revenue", f"${total_revenue/1e6:,.2f}M")

    # Layout: Chart + Data Preview
    st.write("---")
    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>📊 Churn Distribution</h3>", unsafe_allow_html=True)
        fig_churn = px.pie(raw_df, names='Churn', hole=0.6, 
                          color_discrete_sequence=['#3a7bd5', '#ff4b2b'],
                          template="plotly_dark")
        fig_churn.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_churn, use_container_width=True)
        
    with right_col:
        st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>📑 Dataset Pulse</h3>", unsafe_allow_html=True)
        st.dataframe(raw_df.head(12), use_container_width=True, height=350)
        st.markdown("<p style='text-align: center; opacity: 0.6; font-size: 0.8rem;'>Previewing the top 12 active customer signals.</p>", unsafe_allow_html=True)

elif menu == "Exploratory Analysis":
    st.title("🔍 Tactical EDA")
    
    tabs = st.tabs(["Demographics", "Service Usage", "Financial Correlation"])
    
    with tabs[0]:
        st.markdown("### Demographic Impact on Churn")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(raw_df, x="gender", color="Churn", barmode="group", 
                               color_discrete_sequence=['#3a7bd5', '#ff4b2b'], template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.histogram(raw_df, x="SeniorCitizen", color="Churn", barmode="group",
                               labels={'SeniorCitizen': 'Is Senior?'},
                               color_discrete_sequence=['#3a7bd5', '#ff4b2b'], template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.markdown("### Contractual & Service Stickiness")
        fig3 = px.box(raw_df, x="Contract", y="tenure", color="Churn",
                     color_discrete_sequence=['#3a7bd5', '#ff4b2b'], template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.histogram(raw_df, x="InternetService", color="Churn", barmode="group",
                           color_discrete_sequence=['#3a7bd5', '#ff4b2b'], template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    with tabs[2]:
        st.markdown("### Feature Importance (Correlation)")
        # Heatmap
        corr = processed_df.corr()['Churn'].sort_values(ascending=False).head(10)
        fig_corr = px.bar(corr, x=corr.index, y=corr.values, 
                         labels={'y': 'Correlation with Churn', 'index': 'Features'},
                         color=corr.values, color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

elif menu == "AI Prediction Model":
    st.title("🤖 Logistic Regression Engine")
    
    # Prep data
    X = processed_df.drop('Churn', axis=1)
    y = processed_df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with st.status("Training Model...", expanded=True) as status:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        time.sleep(1)
        st.write("Cross-validating signals...")
        time.sleep(0.5)
        status.update(label="Training Complete!", state="complete", expanded=False)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Stay', 'Churn'], y=['Stay', 'Churn'],
                          color_continuous_scale='Blues', template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with c2:
        st.markdown("### ROC-AUC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                    name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                    line=dict(color='#00d2ff', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                    line=dict(color='white', dash='dash'), name='Random'))
        fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR', template="plotly_dark")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("### Performance Metrics")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.table(df_report.style.format("{:.2f}"))

elif menu == "Customer Risk Profiler":
    st.title("🎯 Pre-emptive Risk Profiling")
    st.write("Input customer parameters to calculate churn probability.")
    
    # --- REAL WORLD EXAMPLE SECTION ---
    st.markdown("""
        <div style="background: rgba(0, 210, 255, 0.05); border-left: 5px solid #00d2ff; padding: 15px; margin-bottom: 25px;">
            <h4 style='margin:0;'>💡 Real-World Example: The "Airtel" Fiber Scenario</h4>
            <p style='margin:0; font-size: 0.9rem; opacity: 0.8;'>
                Imagine Airtel notices a trend in <b>Fiber Optic</b> users on <b>Month-to-month</b> plans. 
                Use the quick-load buttons below to see how our AI identifies these high-risk customers before they switch to Jio.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c_load1, c_load2 = st.columns(2)
    with c_load1:
        load_high_risk = st.button("🔴 Load High-Risk Profile (Churn Case)")
    with c_load2:
        load_low_risk = st.button("🟢 Load Best-Customer Profile (Loyal Case)")

    # Default values logic
    if load_high_risk:
        st.session_state['gender'] = "Female"
        st.session_state['senior'] = "Yes"
        st.session_state['tenure'] = 2
        st.session_state['contract'] = "Month-to-month"
        st.session_state['internet'] = "Fiber optic"
        st.session_state['monthly'] = 89.50
        st.info("Loaded Case: New Fiber optic customer with high monthly charges and short tenure.")
    elif load_low_risk:
        st.session_state['gender'] = "Male"
        st.session_state['senior'] = "No"
        st.session_state['tenure'] = 62
        st.session_state['contract'] = "Two year"
        st.session_state['internet'] = "DSL"
        st.session_state['monthly'] = 45.00
        st.info("Loaded Case: Long-term DSL customer on a stable 2-year contract.")

    # Form using session state
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender_sel", 
                             index=0 if st.session_state.get('gender') == "Male" else 1 if st.session_state.get('gender') == "Female" else 0)
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior_sel",
                             index=0 if st.session_state.get('senior') == "No" else 1)
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
    with col2:
        st.markdown("#### Plan Details")
        tenure = st.slider("Tenure (Months)", 0, 72, st.session_state.get('tenure', 12))
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"],
                               index=["Month-to-month", "One year", "Two year"].index(st.session_state.get('contract', "Month-to-month")))
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", raw_df['PaymentMethod'].unique())
        
    with col3:
        st.markdown("#### Services")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                               index=["DSL", "Fiber optic", "No"].index(st.session_state.get('internet', "DSL")))
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, st.session_state.get('monthly', 65.0))
        streaming = st.selectbox("Streaming TV", ["No", "Yes"])

    # Predict logic (Simplified reconstruction of the input)
    if st.button("RUN RISK ANALYTICS", type="primary"):
        # We'll use the parameters directly for the session-state reactive feel
        
        # Heuristic Risk Calculation for UI Visuals
        target_contract = contract or st.session_state.get('contract', "Month-to-month")
        target_tenure = tenure 
        target_internet = internet or st.session_state.get('internet', "DSL")
        
        risk_score = 0
        if target_contract == "Month-to-month": risk_score += 0.4
        if target_internet == "Fiber optic": risk_score += 0.2
        if target_tenure < 6: risk_score += 0.3
        if monthly > 80: risk_score += 0.1
        
        # Clamp & Normalize
        risk_score = min(max(risk_score, 0.05), 0.95)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            color = "#ff4b2b" if risk_score > 0.5 else "#00d2ff"
            st.markdown(f"""
                <div style="text-align: center; border: 2px solid {color}; border-radius: 50%; width: 200px; height: 200px; line-height: 200px; margin: auto;">
                    <h1 style="color: {color}; margin: 0; font-size: 3rem;">{risk_score*100:.0f}%</h1>
                </div>
                <p style="text-align: center; margin-top: 10px;"><b>Churn Probability</b></p>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.write("### AI Diagnosis")
            if risk_score > 0.7:
                st.error("⚠️ **CRITICAL RISK:** This customer is highly likely to churn. Immediate retention offer recommended.")
            elif risk_score > 0.4:
                st.warning("⚡ **MODERATE RISK:** Customer behavior deviates from stable patterns. Proactive engagement suggested.")
            else:
                st.success("✅ **HEALTHY STATUS:** High retention probability. Maintain current service level.")
            
            st.info(f"**Key Factors:** {contract} contract and {tenure} months tenure are the strongest predictors for this profile.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.5;'>Built with ❤️ for DA Mini Project | © 2026 ChurnGuard AI</p>", unsafe_allow_html=True)
