import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Set layout configuration
st.set_page_config(page_title="Enterprise Churn Survival Suite", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main background and text */
    .main { background-color: #f8f9fa; color: #111827; }
    .stApp { color: #111827; }
    
    /* Header styling */
    [data-testid="stHeader"] { background-color: #1a2f4f; }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8em;
        font-weight: 700;
        color: #1a2f4f;
        margin: 20px 0 15px 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    /* Tab styling */
    [role="tablist"] { gap: 5px; }
    
    /* Card styling */
    .card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 15px 0;
        color: #111827;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #0288d1;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #0f172a;
    }

    .card h1, .card h2, .card h3, .card h4, .card h5, .card h6,
    .info-box h1, .info-box h2, .info-box h3, .info-box h4, .info-box h5, .info-box h6 {
        color: inherit;
    }

    .card p, .card li, .info-box p, .info-box li {
        color: inherit;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main, .stApp {
            background-color: #0e1117;
            color: #e5e7eb;
        }

        [data-testid="stHeader"] {
            background-color: #0b1220;
        }

        .section-header {
            color: #e5e7eb;
            border-bottom-color: #7c8db5;
        }

        .card {
            background: #161b22;
            color: #e5e7eb;
            border: 1px solid #2d333b;
            box-shadow: 0 2px 10px rgba(0,0,0,0.28);
        }

        .info-box {
            background: #111827;
            color: #e5e7eb;
            border-left-color: #7dd3fc;
        }

        .stMarkdown, .stText, .stCaption {
            color: #e5e7eb;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 10px; color: white; margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 2.5em;">⏳ Enterprise Churn Survival Suite</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.95;">
            Predict customer lifetime using Cox Proportional Hazards modeling
        </p>
    </div>
""", unsafe_allow_html=True)

# Quick intro
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>📊 Real Data Analysis</strong><br>
        Powered by actual Telco customer churn data
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>⚡ Predictive Modeling</strong><br>
        Cox PH with L2 regularization (penalizer=0.1)
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-box">
        <strong>🎯 Actionable Insights</strong><br>
        Risk-based customer segmentation
        </div>
        """, unsafe_allow_html=True)

# --- DATA PROCESSING INTERNALS ---
@st.cache_data
def load_and_prepare_data():
    """Load real Telco customer churn data and prepare for modeling."""
    data_path = BASE_DIR / 'Data' / 'Telco-Customer-Churn.csv'
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()

    df = pd.read_csv(data_path)
    
    # Create Event binary variable
    df['Event'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Handle TotalCharges column robustly
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', pd.NA, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isna().any():
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Prepare model dataframe
    df_model = df.drop(columns=['customerID', 'Churn'])
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_model, drop_first=True, dtype=int)
    
    return df, df_encoded

@st.cache_data
def train_cox_model(df_encoded):
    """Train Cox Proportional Hazards model with penalizer."""
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_encoded, duration_col='tenure', event_col='Event')
    return cph

# Load data and train model
df, df_encoded = load_and_prepare_data()
cph_model = train_cox_model(df_encoded)

# --- TABS FOR DIFFERENT ANALYSES ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Global Analysis",
    "⚠️ Risk Drivers", 
    "🔮 Individual Prediction",
    "📋 Model Details"
])

with tab1:
    st.markdown('<div class="section-header">📊 Kaplan-Meier Survival Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Global Customer Survival")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        
        kmf = KaplanMeierFitter()
        kmf.fit(df['tenure'], df['Event'], label='Global Customer Survival')
        kmf.plot_survival_function(ax=ax1, linewidth=3, color='#667eea')
        ax1.set_ylabel('Survival Probability', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Tenure (Months)', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Retention by Contract Type")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        month_to_month = df[df['Contract'] == 'Month-to-month']
        two_year = df[df['Contract'] == 'Two year']
        
        kmf_monthly = KaplanMeierFitter()
        kmf_2year = KaplanMeierFitter()
        
        kmf_monthly.fit(month_to_month['tenure'], month_to_month['Event'], label='Month-to-Month')
        kmf_monthly.plot_survival_function(ax=ax2, linewidth=3, color='#f5576c')
        
        kmf_2year.fit(two_year['tenure'], two_year['Event'], label='Two year')
        kmf_2year.plot_survival_function(ax=ax2, linewidth=3, color='#38ef7d')
        
        ax2.set_title('Contract Type Impact on Retention', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Tenure (Months)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Survival Probability', fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        ax2.legend(fontsize=10, loc='lower left')
        
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header">⚡ Churn Risk Drivers</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    cph_model.plot(ax=ax3)
    ax3.set_title('Cox Proportional Hazards - Risk Factor Coefficients', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Log Hazard Ratio', fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax3.set_facecolor('#f8f9fa')
    
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📋 Detailed Model Coefficients", expanded=False):
        st.dataframe(cph_model.summary, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">👤 Customer Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Profile Parameters")
        input_contract = st.selectbox("📋 Contract Type", ["Month-to-month", "One year", "Two year"])
        input_charges = st.slider("💰 Monthly Charges", min_value=0, max_value=150, value=65, step=1)
        input_internet = st.selectbox("🌐 Internet Type", ["DSL", "Fiber optic", "No"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Build profile DataFrame matching model features
    profile_dict = {col: [0] for col in df_encoded.columns if col not in ['tenure', 'Event']}
    
    # Set contract type
    if input_contract == "Month-to-month":
        profile_dict['Contract_Month-to-month'] = [1]
    
    # Set internet type
    if input_internet == "DSL":
        profile_dict['InternetService_DSL'] = [1]
    elif input_internet == "Fiber optic":
        profile_dict['InternetService_Fiber optic'] = [1]
    
    # Set monthly charges
    profile_dict['MonthlyCharges'] = [input_charges]
    
    simulated_profile = pd.DataFrame(profile_dict)
    
    # Make sure all required columns exist
    for col in df_encoded.columns:
        if col not in ['tenure', 'Event'] and col not in simulated_profile.columns:
            simulated_profile[col] = 0
    
    # Reorder to match training data
    simulated_profile = simulated_profile[[col for col in df_encoded.columns if col not in ['tenure', 'Event']]]
    
    # --- PREDICTIVE INFERENCE ---
    individual_survival_curve = cph_model.predict_survival_function(simulated_profile)
    median_lifespan_result = cph_model.predict_median(simulated_profile)
    median_lifespan = float(median_lifespan_result.iloc[0]) if hasattr(median_lifespan_result, 'iloc') else float(median_lifespan_result)
    
    # Determine risk level
    if np.isinf(median_lifespan):
        display_lifespan = "72+ Months"
        risk_level = "Low Risk"
        risk_color = "success"
    elif median_lifespan < 15:
        display_lifespan = f"{int(median_lifespan)} Months"
        risk_level = "🚨 Critical Risk"
        risk_color = "warning"
    elif median_lifespan < 48:
        display_lifespan = f"{int(median_lifespan)} Months"
        risk_level = "⚠️ Moderate Risk"
        risk_color = "info"
    else:
        display_lifespan = f"{int(median_lifespan)} Months"
        risk_level = "✅ Low Risk"
        risk_color = "success"
    
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.markdown(f"""
        <div class="{'metric-card-success' if risk_color == 'success' else 'metric-card-warning'}">
            <div class="metric-label">PREDICTED LIFESPAN</div>
            <div class="metric-value">{display_lifespan}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RISK ASSESSMENT</div>
            <div class="metric-value" style="font-size: 1.8em;">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Survival Projection")
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        sns.set_style("whitegrid")
        
        ax4.plot(
            individual_survival_curve.index,
            individual_survival_curve.iloc[:, 0],
            color='#f5576c' if median_lifespan < 24 else '#ffa502' if median_lifespan < 48 else '#11998e',
            linewidth=4,
            label='Predicted Retention Curve'
        )
        
        ax4.axhline(0.5, color='#999', linestyle='--', alpha=0.6, linewidth=2, label='50% Threshold')
        ax4.fill_between(individual_survival_curve.index, 0, individual_survival_curve.iloc[:, 0], 
                         alpha=0.15, color='#667eea')
        ax4.set_xlabel("Future Tenure (Months)", fontsize=11, fontweight='bold')
        ax4.set_ylabel("Survival Probability", fontsize=11, fontweight='bold')
        ax4.set_xlim(0, 84)
        ax4.set_ylim(0, 1.05)
        ax4.legend(loc='lower left', fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.set_facecolor('#f8f9fa')
        
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Business recommendations
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    if median_lifespan < 15:
        with col1:
            st.markdown("""
            <div style="background: #ffebee; border-left: 5px solid #f5576c; padding: 15px; border-radius: 5px;">
            <strong style="color: #f5576c;">🚨 CRITICAL ACTION REQUIRED</strong><br>
            This customer requires immediate intervention. Recommend:
            <ul>
            <li>Proactive outreach with executive team</li>
            <li>Premium retention packages</li>
            <li>Service quality audit</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    elif median_lifespan < 48:
        with col1:
            st.markdown("""
            <div style="background: #fff3e0; border-left: 5px solid #ffa502; padding: 15px; border-radius: 5px;">
            <strong style="color: #e65100;">⚠️ MODERATE RISK - ENGAGEMENT NEEDED</strong><br>
            Proactive monitoring recommended:
            <ul>
            <li>Regular check-in calls</li>
            <li>Add-on service cross-sell</li>
            <li>Loyalty program enrollment</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        with col1:
            st.markdown("""
            <div style="background: #e8f5e9; border-left: 5px solid #11998e; padding: 15px; border-radius: 5px;">
            <strong style="color: #11998e;">✅ STABLE CUSTOMER - GROWTH FOCUS</strong><br>
            Focus on value expansion:
            <ul>
            <li>Premium add-on upsell</li>
            <li>Bundle optimization</li>
            <li>VIP customer experience</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL CUSTOMERS</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card-warning">
            <div class="metric-label">CHURN EVENTS</div>
            <div class="metric-value">{int(df['Event'].sum()):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CHURN RATE</div>
            <div class="metric-value">{(df['Event'].sum()/len(df)*100):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card-success">
            <div class="metric-label">MODEL FEATURES</div>
            <div class="metric-value">{df_encoded.shape[1] - 2}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CONCORDANCE INDEX</div>
            <div class="metric-value">{cph_model.concordance_index_:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card-success">
            <div class="metric-label">REGULARIZATION</div>
            <div class="metric-value">L2 (0.1)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Detailed Coefficients</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(cph_model.summary, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)