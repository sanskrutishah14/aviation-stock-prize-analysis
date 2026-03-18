from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import itertools
from tqdm.notebook import tqdm
import warnings

# NEW: Prophet (Facebook)
from prophet import Prophet

warnings.filterwarnings("ignore")


# 1) Data import & PCA features

def import_and_clean_data(airline):
    """
    Reads {airline}.csv and returns standardized numeric matrix X for PCA and the cleaned DataFrame df.
    Expected columns start as: Year, Month, Price, ...
    All numeric columns may contain commas (Indian-numbering style) and are sanitized.
    """
    df = pd.read_csv(f"{airline}.csv")

    # Convert everything to numeric (from col index 2 onward, i.e., Price + features)
    numeric_cols = df.columns[2:]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse time — keep Year/Month as timeline indicators
    # Expect 'Year' and 'Month' as integers (YYYY, 1..12)
    # Convert to a proper monthly "Date"
    df['Year'] = pd.to_datetime(df['Year'], format="%Y")
    df['Month'] = pd.to_datetime(df['Month'], format="%m")
    df['Date'] = pd.to_datetime(
        df['Year'].dt.year.astype(str) + '-' + df['Month'].dt.month.astype(str) + '-01'
    )

    # Only numeric feature columns for PCA (exclude Year, Month, Date, Price)
    x = df.iloc[:, 3:].select_dtypes(include=[np.number])

    # Standardize features for PCA
    standard = StandardScaler()
    x = standard.fit_transform(x)
    return x, df


def sensitivity_index(airline):
    """
    Runs PCA on standardized numeric features (except Date/Price) and returns:
    - df_out: DataFrame with Date, Price, and PC1..PCk columns
    - loadings_df: PCA loadings per original feature
    - explained_variance_ratio: list of PC explained variance ratios
    """
    x, df = import_and_clean_data(airline)

    # Feature names before PCA (for loadings)
    feature_cols = df.iloc[:, 3:].select_dtypes(include=[np.number]).columns

    # PCA retaining 90% variance
    pca = PCA(0.9)
    pca.fit(x)
    pca_data = pca.transform(x)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = np.sqrt(pca.explained_variance_) * pca.components_.T

    # Loadings dataframe
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f"PC{i+1}" for i in range(len(explained_variance_ratio))],
        index=feature_cols
    ).round(3)

    # PCs dataframe
    pc_df = pd.DataFrame(
        pca_data,
        columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
    )

    # Output with Date and Price first, then PCs
    df_out = pd.DataFrame()
    df_out['Date'] = df['Date'].reset_index(drop=True)
    df_out['Price'] = df['Price'].reset_index(drop=True)

    for col in pc_df.columns:
        df_out[col] = pc_df[col].values

    df_out = df_out.sort_values('Date').reset_index(drop=True)
    return df_out, loadings_df, explained_variance_ratio


# 2) Seasonality & stationarity utils

def detect_seasonality(series, period=12, threshold=0.1):
    """
    Returns True if seasonal strength > threshold.
    Requires at least 2*period observations.
    """
    if len(series) < 2 * period:
        return False
    result = seasonal_decompose(series, model='additive', period=period)
    sev = result.seasonal.var()
    resv = result.resid.var()
    seasonal_strength = sev / (sev + resv) if (sev + resv) != 0 else 0
    return seasonal_strength > threshold


def differencing(series, seasonal=False):
    """
    Log + difference loop to try stationarity.
    Returns (series_after_diff, d, D).
    """
    series = np.log(series)
    d = 0
    D = 0

    # Regular differencing up to 2 times
    for i in range(2):
        diff_series = series.diff().dropna()
        d = i + 1
        try:
            if adfuller(diff_series)[1] < 0.05:
                return diff_series, d, D
        except Exception:
            pass
        series = diff_series

    # Seasonal differencing if requested
    if seasonal:
        diff_series = series.diff(12).dropna()
        D = 1
        try:
            if adfuller(diff_series)[1] < 0.05:
                return diff_series, d, D
        except Exception:
            pass
        return diff_series, d, D

    return series, d, D


def stationarity_check_conversion(series, seasonal=False):
    """
    Run ADF on raw series. If non-stationary, call differencing(...).
    """
    try:
        p_val = adfuller(series)[1]
    except Exception:
        p_val = 1.0
    if p_val > 0.05:
        series_out, d, D = differencing(series, seasonal)
        return series_out, d, D
    else:
        return series, 0, 0


# 3) SARIMAX search & exog forecasting

def optimize_SARIMAX(parameters_list, d, D, s, endog, exog=None):
    results = []
    for param in tqdm(parameters_list, desc="SARIMAX grid"):
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

    exog_future = np.column_stack(future_pcs) if future_pcs else np.empty((steps, 0))
    return exog_future


# 4) SARIMAX plotting & metrics

def plot_price(df, forecast_index, forecast_mean, fitted_values):
    import matplotlib.dates as mdates

    plt.figure(figsize=(15, 7.5))
    plt.plot(df['Date'], df['Price'], label='Actual Price', linewidth=2, marker='o', markersize=4)
    plt.plot(df['Date'], fitted_values, label='Fitted Values', linewidth=2)

    # Connect forecast to last actual data point
    last_historical_date = df['Date'].iloc[-1]
    last_historical_price = df['Price'].iloc[-1]
    complete_forecast_dates = [last_historical_date] + list(forecast_index)
    complete_forecast_values = [last_historical_price] + list(forecast_mean)

    plt.plot(
        complete_forecast_dates, complete_forecast_values,
        label='Forecasted Price', linewidth=2, marker='s', markersize=6
    )

    last_forecast_date = forecast_index[-1]
    plt.axvspan(last_historical_date, last_forecast_date, alpha=0.3, color='lightgrey', label='Forecast Period')
    plt.axvline(x=last_historical_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Time'); plt.ylabel('Price')
    plt.title('SARIMAX: Actual vs Fitted vs Forecasted Price Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def model_metrics(df, best_model):
    y_true = df['Price'][best_model.loglikelihood_burn:]  # skip initial NaNs
    y_pred = best_model.fittedvalues[best_model.loglikelihood_burn:]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# 5) SARIMAX pipeline
def time_series(airline, forecast_periods):
    df, _, _ = sensitivity_index(airline)
    ts = pd.Series(df['Price'].values, index=df['Date'])

    # Seasonality
    seasonality_present = detect_seasonality(ts, period=12)

    # Stationarity + chosen d, D
    ts_stationary, d, D = stationarity_check_conversion(ts, seasonal=seasonality_present)

    # Grid search for (p, q, P, Q)
    p = P = q = Q = range(0, 3)
    parameters = list(itertools.product(p, q, P, Q))

    # Exogenous = PCs
    exog_vars = df[[col for col in df.columns if col.startswith('PC')]]

    result_table = optimize_SARIMAX(parameters, d=d, D=D, s=12, endog=df['Price'], exog=exog_vars)
    if result_table.empty:
        raise ValueError("No SARIMAX models converged in optimize_SARIMAX. Try smaller search space or change data.")

    bestvals = result_table.iloc[0, 0]
    p, q, P, Q = bestvals

    best_model = SARIMAX(
        endog=df['Price'], exog=exog_vars,
        order=(p, d, q),
        seasonal_order=(P, D, Q, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # Forecast future PCA regressors
    exog_future = forecast_pca_exog(df, steps=forecast_periods)

    fitted_values = best_model.fittedvalues.copy()
    fitted_values.iloc[:max(5, int(len(fitted_values) * 0.05))] = np.NaN

    forecast = best_model.get_forecast(steps=forecast_periods, exog=exog_future)
    forecast_mean = forecast.predicted_mean
    forecast_index = pd.date_range(
        start=df['Date'].iloc[-1] + pd.DateOffset(months=1),
        periods=len(forecast_mean), freq='MS'
    )

    plot_price(df, forecast_index, forecast_mean, fitted_values)

    forecast_summary = pd.DataFrame({
        "Date": forecast_index,
        "Forecasted Price": forecast_mean
    })

    metrics = model_metrics(df, best_model)
    return ts, forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics


def Airline(airline, forecast_periods):
    """
    Wrapper to run SARIMAX for an airline. Returns series, PCA loadings, explained variance,
    a matplotlib figure, metrics, and forecast summary.
    """
    df, loadings_df, explained_variance_ratio = sensitivity_index(airline)
    df['seq_index'] = range(len(df))
    ts, forecast_summary, forecast_index, forecast_mean, fitted_values, df, best_model, metrics = time_series(airline, forecast_periods)

    fig_forecast, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['Date'], df['Price'], label='Actual Price', linewidth=2)
    ax1.plot(df['Date'], fitted_values, label='Fitted', linestyle='--')
    ax1.plot(forecast_index, forecast_mean, label='Forecast', color='red', marker='o')
    ax1.legend()
    ax1.set_title(f'Forecasted vs Actual Prices (SARIMAX) — {airline.title()}')
    plt.tight_layout()

    return ts, loadings_df, explained_variance_ratio, fig_forecast, metrics, forecast_summary


# 6) Prophet integration (with PCA regressors)
def prepare_prophet_frame(df):
    """
    df: output of sensitivity_index(airline) -> has Date, Price, PC1..PCk
    Returns dfp with Prophet's columns: ds (datetime), y (float), plus PC regressors
    """
    dfp = df.copy().sort_values("Date").reset_index(drop=True)
    dfp.rename(columns={"Date": "ds", "Price": "y"}, inplace=True)

    # Ensure numeric and no NaNs in y
    dfp["y"] = pd.to_numeric(dfp["y"], errors="coerce")
    pc_cols = [c for c in dfp.columns if c.startswith("PC")]
    for c in pc_cols:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = dfp.dropna(subset=["y"])  # ensure y exists
    return dfp, pc_cols


def prophet_with_pca(df, forecast_periods=8, yearly_seasonality=True):
    """
    df: output of sensitivity_index(airline) (Date, Price, PC1..)
    Returns: model, forecast_df (with yhat), fitted_df, metrics
    """
    # Prepare
    dfp, pc_cols = prepare_prophet_frame(df)

    # Build Prophet model
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.8
    )

    # Add PCA regressors
    for c in pc_cols:
        m.add_regressor(c)

    # Fit Prophet (needs all regressors in the dfp passed to fit)
    m.fit(dfp[["ds", "y"] + pc_cols])

    # In-sample fitted values
    fitted_df = m.predict(dfp[["ds"] + pc_cols])
    fitted_df = fitted_df[["ds", "yhat"]].rename(columns={"yhat": "fitted"})

    # Future dates (monthly steps)
    last_ds = dfp["ds"].max()
    future_dates = pd.date_range(
        start=last_ds + pd.DateOffset(months=1),
        periods=forecast_periods, freq="MS"
    )

    # Forecast future PCs using your ARIMA helper
    exog_future = forecast_pca_exog(df, steps=forecast_periods)  # shape (steps, n_PCs)

    future = pd.DataFrame({"ds": future_dates})
    for i, c in enumerate(pc_cols):
        if exog_future.shape[1] > i:
            future[c] = exog_future[:, i]
        else:
            # Edge case: if no PCs exist (shouldn't happen if PCA was run)
            future[c] = 0.0

    # Forecast
    forecast_df = m.predict(future)
    forecast_df = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_df.rename(columns={"yhat": "Forecasted Price"}, inplace=True)

    # Metrics on in-sample fit
    merged = dfp.merge(fitted_df, on="ds", how="left")
    y_true = merged["y"].values
    y_pred = merged["fitted"].values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    return m, forecast_df, fitted_df, metrics


def plot_prophet_price(df, fitted_df, forecast_df, title_suffix=""):
    import matplotlib.dates as mdates

    dfp, _ = prepare_prophet_frame(df)
    plt.figure(figsize=(15, 7.5))
    plt.plot(dfp["ds"], dfp["y"], label="Actual Price", linewidth=2, marker="o", markersize=4)

    # Fitted
    plt.plot(fitted_df["ds"], fitted_df["fitted"], label="Fitted (Prophet)", linewidth=2)

    # Connect forecast to last actual
    last_hist_ds = dfp["ds"].max()
    last_hist_y = dfp.loc[dfp["ds"] == last_hist_ds, "y"].iloc[0]
    complete_forecast_dates = [last_hist_ds] + list(forecast_df["ds"])
    complete_forecast_values = [last_hist_y] + list(forecast_df["Forecasted Price"])

    plt.plot(
        complete_forecast_dates, complete_forecast_values,
        label="Forecasted Price (Prophet)", linewidth=2, marker="s", markersize=6
    )

    # Shade forecast window
    plt.axvspan(last_hist_ds, forecast_df["ds"].max(), alpha=0.3, label="Forecast Period")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Time"); plt.ylabel("Price")
    plt.title(f"Prophet: Actual vs Fitted vs Forecasted Price {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def Airline_Prophet(airline, forecast_periods=8):
    """
    Prophet wrapper mirroring Airline(SARIMAX). Returns consistent artifacts.
    """
    df, loadings_df, explained_variance_ratio = sensitivity_index(airline)
    model_p, forecast_df, fitted_df, metrics = prophet_with_pca(df, forecast_periods=forecast_periods)

    # Build a simple figure to return
    fig, ax = plt.subplots(figsize=(10, 5))
    dfp, _ = prepare_prophet_frame(df)
    ax.plot(dfp["ds"], dfp["y"], label="Actual Price", linewidth=2)
    ax.plot(fitted_df["ds"], fitted_df["fitted"], label="Fitted (Prophet)", linestyle="--")
    ax.plot(forecast_df["ds"], forecast_df["Forecasted Price"], label="Forecast", marker="o")
    ax.legend()
    ax.set_title(f"Prophet Forecast vs Actual Prices — {airline.title()}")
    plt.tight_layout()

    # Harmonize the return signature with Airline()
    # (series, loadings_df, explained_variance_ratio, fig, metrics, forecast_summary)
    return dfp["y"], loadings_df, explained_variance_ratio, fig, metrics, forecast_df[["ds","Forecasted Price","yhat_lower","yhat_upper"]]


