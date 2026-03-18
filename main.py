from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import itertools
from tqdm import tqdm
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pathlib
import re


import warnings
warnings.filterwarnings("ignore")

# Point to where your CSVs actually live
DATA_DIR = pathlib.Path(__file__).parent / "pages"   # <-- change to "." if CSVs are in repo root

# Accept many possible UI labels and map to the actual filenames
AIRLINE_ALIASES = {
    "american": "american.csv",
    "american airlines": "american.csv",
    "aa": "american.csv",

    "delta": "delta.csv",
    "delta airlines": "delta.csv",
    "delta air lines": "delta.csv",

    "southwest": "southwest.csv",
    "southwest airlines": "southwest.csv",

    "united": "united.csv",
    "united airlines": "united.csv",
}

def resolve_csv(name: str) -> pathlib.Path:
    """
    Normalizes a user-facing airline label into a concrete CSV path.
    Tries aliases first, then smart fallbacks, then raises a clear error.
    """
    key = str(name).strip().lower()

    # 1) direct alias hit
    if key in AIRLINE_ALIASES:
        return DATA_DIR / AIRLINE_ALIASES[key]

    # 2) strip common suffixes and retry
    key2 = (
        key.replace("airlines", "")
           .replace("air lines", "")
           .strip()
    )
    if key2 in AIRLINE_ALIASES:
        return DATA_DIR / AIRLINE_ALIASES[key2]

    # 3) fallback: try "<key>.csv" and "<key2>.csv" directly
    cand1 = DATA_DIR / f"{key}.csv"
    if cand1.exists():
        return cand1
    cand2 = DATA_DIR / f"{key2}.csv"
    if cand2.exists():
        return cand2

    # 4) give a helpful error
    valid = sorted(set(AIRLINE_ALIASES.keys()))
    raise FileNotFoundError(
        f"No CSV for airline '{name}'. Looked in {DATA_DIR}. "
        f"Try one of: {valid}"
    )


def import_and_clean_data(airline):
    csv_path = resolve_csv(airline)   # <-- use resolver
    df = pd.read_csv(csv_path)

    # Convert everything to string, remove commas
    numeric_cols = df.columns[2:]  # Price + features
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Year'] = pd.to_datetime(df['Year'], format="%Y")
    df['Month'] = pd.to_datetime(df['Month'], format="%m")
    df['Date'] = pd.to_datetime(df['Year'].dt.year.astype(str) + '-' + df['Month'].dt.month.astype(str) + '-01')

    # FIX: Only select numeric feature columns for PCA (exclude Year, Month, Date, Price)
    x = df.iloc[:, 3:].select_dtypes(include=[np.number])

    # standardize
    standard = StandardScaler()
    x = standard.fit_transform(x)
    return x, df



def sensitivity_index(airline):
    x, df = import_and_clean_data(airline)
    
    # Store the feature column names BEFORE PCA
    feature_cols = df.iloc[:, 3:].select_dtypes(include=[np.number]).columns
    
    #PCA for Sensitivity Index
    pca = PCA(0.9)
    pca.fit(x)
    pca_data = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T
    
    # FIX: Use actual feature_cols instead of df.columns[3:]
    loadings_df = pd.DataFrame(loadings, columns=["PC"+f"{i+1}" for i in range(len(explained_variance_ratio))], index=feature_cols)
    loadings_df = loadings_df.round(3)
    pc_df = pd.DataFrame(
        pca_data,
        columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
    )
    
    # Create df_out with Date FIRST, then numeric columns
    df_out = pd.DataFrame()
    df_out['Date'] = df['Date'].reset_index(drop=True)
    df_out['Price'] = df['Price'].reset_index(drop=True)
    
    # Add PC columns
    for col in pc_df.columns:
        df_out[col] = pc_df[col].values
    
    df_out = df_out.sort_values('Date').reset_index(drop=True)

    return df_out, loadings_df, explained_variance_ratio

def detect_seasonality(series, period=12, threshold=0.1):
    # Guard: seasonal_decompose needs at least 2 * period observations
    if len(series) < 2 * period:
        return False
    result = seasonal_decompose(series, model='additive', period=period)
    # seasonal strength metric
    sev = result.seasonal.var()
    resv = result.resid.var()
    seasonal_strength = sev / (sev + resv) if (sev + resv) != 0 else 0
    return seasonal_strength > threshold


def differencing(series, seasonal=False):
    """
    Returns (series_after_diff, d, D)
    series input is raw (not logged) and must be a pd.Series.
    """
    series = np.log(series)
    d = 0
    D = 0

    # Regular differencing up to 2 times
    for i in range(2):
        diff_series = series.diff().dropna()
        d = i + 1
        if adfuller(diff_series)[1] < 0.05:
            return diff_series, d, D
        series = diff_series

    # Seasonal differencing if requested
    if seasonal:
        diff_series = series.diff(12).dropna()
        D = 1
        if adfuller(diff_series)[1] < 0.05:
            return diff_series, d, D
        return diff_series, d, D

    # Return what we have if not stationary
    return series, d, D


def stationarity_check_conversion(series, seasonal=False):
    p_val = adfuller(series)[1]
    if p_val > 0.05:
        series_out, d, D = differencing(series, seasonal)
        return series_out, d, D
    else:
        return series, 0, 0


def optimize_SARIMAX(parameters_list, d, D, s, endog, exog=None):
    results = []
    for param in tqdm(parameters_list):
        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=(param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], s),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            results.append({
                '(p,q)x(P,Q)': param,
                'AIC': model.aic
            })
        except Exception:
            continue
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df


def forecast_pca_exog(df, steps=8):
    """
    Forecast each PCA component (exogenous variable) separately using auto_arima.
    Returns np.array of shape (steps, num_PCs).
    """
    # FIX: Explicitly select only PC columns
    exog = df[[col for col in df.columns if col.startswith('PC')]]
    future_pcs = []

    for col in exog.columns:
        model_pc = auto_arima(
            exog[col],
            seasonal=True,
            m=12,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        pc_forecast = model_pc.predict(n_periods=steps)
        future_pcs.append(pc_forecast)

    # shape (steps, n_PCs)
    exog_future = np.column_stack(future_pcs)
    return exog_future


def plot_price(df, forecast_index, forecast_mean, fitted_values):
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(15, 7.5))
    plt.plot(df['Date'], df['Price'], label='Actual Price', linewidth=2, marker='o', markersize=4)
    plt.plot(df['Date'], fitted_values, label='Fitted Values', linewidth=2)
    
    # CRITICAL: Connect forecast to last actual data point
    last_historical_date = df['Date'].iloc[-1]
    last_historical_price = df['Price'].iloc[-1]
    
    complete_forecast_dates = [last_historical_date] + list(forecast_index)
    complete_forecast_values = [last_historical_price] + list(forecast_mean)
    
    plt.plot(complete_forecast_dates, complete_forecast_values, 
             label='Forecasted Price', linewidth=2, marker='s', markersize=6, color='red')
    
    last_forecast_date = forecast_index[-1]
    
    plt.axvspan(last_historical_date, last_forecast_date, alpha=0.3, color='lightgrey', label='Forecast Period')
    plt.axvline(x=last_historical_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Format x-axis to show year and month
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
    plt.xticks(rotation=45, ha='right')
    
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('SARIMAX: Actual vs Fitted vs Forecasted Price Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def model_metrics(df,best_model):
    y_true = df['Price'][best_model.loglikelihood_burn:]  # skip initial NaNs
    y_pred = best_model.fittedvalues[best_model.loglikelihood_burn:]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def time_series(airline,forecast_periods):   #
    df, _,_ = sensitivity_index(airline)
    ts = pd.Series(df['Price'].values, index=df['Date'])
    
    seasonality_present = detect_seasonality(ts, period=12)

    # Stationarity + chosen d, D
    ts_stationary, d, D = stationarity_check_conversion(ts, seasonal=seasonality_present)

    # Grid search for (p,q,P,Q)
    p = P = q = Q = range(0, 3)
    parameters = list(itertools.product(p, q, P, Q))
    
    # FIX: Explicitly select only PC columns, exclude Date
    exog_vars = df[[col for col in df.columns if col.startswith('PC')]]
    
    result_table = optimize_SARIMAX(parameters, d=d, D=D, s=12, endog=df['Price'], exog=exog_vars)

    if result_table.empty:
        raise ValueError("No SARIMAX models converged in optimize_SARIMAX. Try smaller search space or change data.")

    bestvals = result_table.iloc[0, 0]
    p, q, P, Q = bestvals

    best_model = SARIMAX(endog=df['Price'], exog=exog_vars,
                         order=(p, d, q),
                         seasonal_order=(P, D, Q, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)

    # Forecast each PCA exog for the next n months
    exog_future = forecast_pca_exog(df, steps=forecast_periods)

    fitted_values = best_model.fittedvalues.copy()
    fitted_values.iloc[:max(5, int(len(fitted_values) * 0.05))] = np.nan

    forecast = best_model.get_forecast(steps=forecast_periods, exog=exog_future)
    forecast_mean = forecast.predicted_mean

    forecast_index = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=len(forecast_mean), freq='MS')

    plot_price(df, forecast_index, forecast_mean, fitted_values)
    # Print summary
    forecast_summary = pd.DataFrame({
        "Date": forecast_index,
        "Forecasted Price": forecast_mean
    })

    metrics = model_metrics(df,best_model)
    return ts, forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics


def prophet_forecast(df, forecast_periods):
    prophet_df = df[["Date", "Price"]].rename(columns={"Date": "ds", "Price": "y"})

    exog_cols = [col for col in df.columns if col.startswith("PC")]
    for col in exog_cols:
        prophet_df[col] = df[col].values

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3
    )

    for col in exog_cols:
        model.add_regressor(col)

    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=forecast_periods, freq="MS")

    #Forecast exogenous PCA components (since we need future values)
    exog_future = forecast_pca_exog(df, steps=forecast_periods)
    for i, col in enumerate(exog_cols):
        future[col] = np.concatenate([df[col].values, exog_future[:, i]])

    forecast = model.predict(future)
    
    plt.figure(figsize=(15, 7.5))
    plt.plot(df["Date"], df["Price"], label="Actual Price", linewidth=2, marker="o", markersize=4)
    plt.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast", color="red", linewidth=2, marker="s", markersize=5)
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                     color="pink", alpha=0.3, label="Confidence Interval")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Prophet (with PCA Regressors): Actual vs Forecasted Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    y_true = prophet_df["y"].values
    y_pred = forecast["yhat"][:len(y_true)].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        "Model": "Prophet (with PCA Regressors)",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    forecast_summary = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date"})

    return model, forecast_summary, metrics

def detect_model_type(series):
    mean_val, std_val = series.mean(), series.std()
    ratio = std_val / mean_val
    if ratio > 0.3:
        trend_type, seasonal_type = "mul", "mul"
    else:
        trend_type, seasonal_type = "add", "add"
    print(f"Detected model type: trend='{trend_type}', seasonal='{seasonal_type}' (ratio={ratio:.2f})")
    return trend_type, seasonal_type


def holt_winters_forecast(airline, forecast_periods=8):
    df, loadings_df, explained_variance_ratio = sensitivity_index(airline)
    ts = pd.Series(df['Price'].values, index=df['Date'])
    trend_type, seasonal_type = detect_model_type(ts)
    model = ExponentialSmoothing(ts, trend=trend_type, seasonal=seasonal_type, seasonal_periods=12).fit(optimized=True)
    fitted_values = model.fittedvalues
    forecast = model.forecast(forecast_periods)
    forecast_index = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(ts.index, ts, label='Actual', linewidth=2, marker='o')
    plt.plot(ts.index, fitted_values, label='Fitted', linestyle='--', linewidth=2)
    plt.plot(forecast_index, forecast, label='Forecast', color='red', marker='s', linewidth=2)
    plt.axvline(x=ts.index[-1], color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    plt.title("Holt-Winters Forecast (Auto Add/Mul Detection)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Metrics
    mae = mean_absolute_error(ts, fitted_values)
    rmse = np.sqrt(mean_squared_error(ts, fitted_values))
    mape = np.mean(np.abs((ts - fitted_values) / ts)) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    forecast_summary = pd.DataFrame({"Date": forecast_index, "Forecasted Price": forecast.values})
    return df, forecast_summary, metrics, loadings_df, explained_variance_ratio, trend_type, seasonal_type


def Airline(airline, forecast_periods, model="SARIMAX"):
    df, loadings_df, explained_variance_ratio = sensitivity_index(airline)
    if model.upper() == "SARIMAX":
        ts, forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics = time_series(airline, forecast_periods)
        return {
            "Model": "SARIMAX",
            "Forecast": forecast_summary,
            "Metrics": metrics,
            "Loadings": loadings_df,
            "ExplainedVar": explained_variance_ratio,
        }
    elif model.upper() == "PROPHET":
        model_prophet, forecast_summary, metrics = prophet_forecast(df, forecast_periods)
        return {
            "Model": "Prophet",
            "Forecast": forecast_summary,
            "Metrics": metrics,
            "Loadings": loadings_df,
            "ExplainedVar": explained_variance_ratio,
        }
    elif model.upper() == "HOLT-WINTERS":
        df, forecast_summary, metrics, loadings_df, explained_variance_ratio, trend_type, seasonal_type = holt_winters_forecast(airline, forecast_periods)
        return {"Model": "Holt-Winters", 
                "Forecast": forecast_summary, 
                "Metrics": metrics,
                "Trend": trend_type, 
                "Seasonal": seasonal_type,
                "Loadings": loadings_df,
                "ExplainedVar": explained_variance_ratio}
    else:
        raise ValueError("Invalid model choice. Choose either 'SARIMAX' , 'Prophet', or 'HOLT-WINTERS'.")


if __name__ == "__main__":
    Airline("american",8,model="HOLT-WINTERS")
