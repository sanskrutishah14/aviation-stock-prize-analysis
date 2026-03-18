import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import Airline  # your backend function
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import base64

# --- PAGE CONFIG 
st.set_page_config(
    page_title="American Airlines Forecast ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#styles
st.markdown("""
<style>
/* GLOBAL */
.stApp {
    background: linear-gradient(145deg, #e8edf2, #f8fafc);
    font-family: 'Poppins', sans-serif;
    color: #0d1b2a;
}

/* HEADER */
.header {
    background: linear-gradient(135deg, #003566, #ffffff 50%, #c1121f);
    color: #ffffff;
    text-align: center;
    padding: 45px 0;
    border-radius: 22px;
    box-shadow: 0 10px 35px rgba(0, 21, 41, 0.4);
    margin-bottom: 50px;
    transition: transform 0.3s ease;
    border: 2px solid rgba(255,255,255,0.3);
}

.header:hover {
    transform: translateY(-3px);
}

.header h1 {
    font-weight: 700;
    font-size: 2.5rem;
    letter-spacing: 0.5px;
    text-shadow: 0px 0px 10px rgba(0,0,0,0.3);
    color: #001d3d;
}

.header p {
    font-size: 1.05rem;
    color: #001d3d;
    margin-top: 10px;
    font-weight: 500;
}

/* INPUT FIELD */
div[data-baseweb="input"] > input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1.8px solid #003566 !important;
    border-radius: 10px !important;
    padding: 0.7em 1em !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: 0.3s ease;
}
div[data-baseweb="input"] > input:focus {
    box-shadow: 0 0 10px #0077b6 !important;
    border: 1.8px solid #0077b6 !important;
}
label[data-testid="stWidgetLabel"] {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* BUTTONS */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #001d3d, #003566);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 17px;
    padding: 0.8em 2.2em;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    transition: all 0.3s ease-in-out;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #000814, #003566);
    color: #00b4d8;
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* CARD CONTAINERS */
.card {
    background: linear-gradient(145deg, #ffffff, #f3f4f6);
    border-radius: 20px;
    padding: 35px;
    margin: 30px auto;
    width: 90%;
    box-shadow: 0 12px 24px rgba(0,0,0,0.08),
                inset 0 0 10px rgba(255,255,255,0.4);
    transition: 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 35px rgba(0,0,0,0.12),
                0 0 15px rgba(0, 119, 182, 0.2);
}

/* SECTION HEADINGS */
h3 {
    color: #001d3d !important;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.3px;
    margin-bottom: 20px;
}

/* METRICS */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, #ffffff 60%, #f8f9fa 100%) !important;
    border: 1px solid rgba(0, 53, 102, 0.2);
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.12);
    transition: transform 0.25s ease-in-out;
}
[data-testid="metric-container"]:hover {
    transform: scale(1.03);
    box-shadow: 0px 8px 28px rgba(0,0,0,0.15);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="metric-container"] * {
    color: #000000 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    letter-spacing: 0.3px;
}

/* FOOTER */
.footer {
    background: linear-gradient(90deg, #001d3d, #000814);
    color: #ffffff;
    text-align: center;
    padding: 18px;
    border-radius: 14px;
    margin-top: 60px;
    font-size: 15px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class='header'>
    <h1>✈️ American Airlines Forecast</h1>
</div>
""", unsafe_allow_html=True)

# --- INPUT SECTION ---
#st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3>📅 Enter Forecast Period</h3>", unsafe_allow_html=True)
steps = st.number_input(
    "Enter number of months to forecast ahead:",
    min_value=1,
    max_value=36,
    value=8,
    step=1,
    help="Specify how many future months you want to forecast."
)
run_button = st.button("🚀 Run Forecast")
st.markdown("</div>", unsafe_allow_html=True)

# --- PROCESSING ---
if run_button:
    with st.spinner("Running time series decomposition and forecast... Please wait ⏳"):
        try:
            # Backend returns 6 values
            ts,loadings_df, explained_variance_ratio, fig_forecast, metrics, forecast_summary = Airline("american", steps)

            mae = metrics.get("MAE", 0)
            rmse = metrics.get("RMSE", 0)
            mape = metrics.get("MAPE", 0)

            st.success("✅ Forecast & Decomposition Complete!")
        except Exception as e:
            st.error(f"⚠️ Error during forecast: {e}")
            st.stop()

    # PCA Factors
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🔍 Top 3 PCA Sensitivity Factors</h3>", unsafe_allow_html=True)
    def highlight_top3(data):
            """
            Highlight top 3 values in each column with gradient blue
            """
            df_style = pd.DataFrame('', index=data.index, columns=data.columns)
            
            for col in data.columns:
                # Get top 3 values for this column
                top3_values = data[col].nlargest(3)
                
                # Create gradient for top 3
                for idx, (index, value) in enumerate(top3_values.items()):
                    # Gradient from darkest to lightest blue
                    intensity = 1 - (idx * 0.3)  # 1.0, 0.7, 0.4
                    df_style.loc[index, col] = f'background-color: rgba(33, 150, 243, {intensity})'
        
            return df_style
    st.dataframe(loadings_df.style.apply(highlight_top3, axis=None).format(precision=3))    
    st.markdown("</div>", unsafe_allow_html=True)

    # PCA Explained Variance
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📈 PCA Explained Variance Ratio</h3>", unsafe_allow_html=True)
    fig_var, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color="#003566")
    ax.set_title("Explained Variance by PCA Components")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Variance Ratio")
    st.pyplot(fig_var)
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("MAE", f"{mae:.3f}")
    with col2: st.metric("RMSE", f"{rmse:.3f}")
    with col3: st.metric("MAPE", f"{mape:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Time Series Decomposition
    # Guard: seasonal_decompose needs at least 2*period points
    period = 12
    if len(ts) >= 2 * period:
        result = seasonal_decompose(ts, model='multiplicative', period=period)

        # Make a tall, vertical figure (one below another)
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        plt.subplots_adjust(hspace=0.4)

        # Original series
        axes[0].plot(ts, linewidth=2)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(alpha=0.3)

        # Trend
        axes[1].plot(result.trend, linewidth=2, color='tab:orange')
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Trend')
        axes[1].grid(alpha=0.3)

        # Seasonal
        axes[2].plot(result.seasonal, linewidth=2, color='tab:green')
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Seasonality')
        axes[2].grid(alpha=0.3)

        # Residual
        axes[3].plot(result.resid, linewidth=2, color='tab:red')
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Time')
        axes[3].grid(alpha=0.3)

        # IMPORTANT: don't use plt.show() in Streamlit; pass the fig to st.pyplot
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>🧩 Time Series Decomposition</h3>", unsafe_allow_html=True)
        st.markdown("Decomposing the original data into <b>Trend</b>, <b>Seasonal</b>, and <b>Residual</b> components helps reveal hidden patterns before forecasting.", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Need at least {2*period} observations for seasonal decomposition (have {len(ts)}).")

    # Forecast Visualization
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📈 Forecast Visualization</h3>", unsafe_allow_html=True)
    st.pyplot(fig_forecast)
    st.markdown("</div>", unsafe_allow_html=True)

    # Forecasted Data
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🧾 Forecasted Price Data</h3>", unsafe_allow_html=True)
    st.dataframe(forecast_summary)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 Enter a forecast period and click **Run Forecast** to start analysis.")

# --- FOOTER ---
#st.markdown("""
#<div class='footer'>
#    © 2025 Aviation Stock Forecast
#</div>
#""", unsafe_allow_html=True)