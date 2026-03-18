# Aviation Stock Price Analysis and Forecasting

This project provides a comprehensive framework for analyzing and predicting the stock prices of major airline carriers, including American, Delta, Southwest, and United. By combining dimensionality reduction techniques with advanced time-series modeling, the system identifies key market drivers and provides multi-model price forecasts.

## Core Features
- Data Engineering: Automated cleaning of financial datasets, including currency normalization and timeline alignment.
- Sensitivity Analysis: Utilizes Principal Component Analysis (PCA) to condense numerous technical indicators into high-variance Sensitivity Indices, maintaining 90% of the original data's information.
- Comparative Forecasting: Implementation of three distinct modeling approaches:
    - SARIMAX: For seasonal trends with exogenous variable support.
    - Prophet: For robust handling of outliers and holiday effects.
    - Holt-Winters: For triple exponential smoothing to capture trend and seasonality.
- Interactive Interface: A Streamlit dashboard designed for visualizing historical trends and comparing future projections.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sanskrutishah14/aviation-stock-prize-analysis.git
   cd aviation-stock-prize-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- app.py: The primary script for launching the Streamlit web interface.
- main.py: The central engine for data processing and model coordination.
- prophet_pipeline.py / holt_winters.py: Specialized modules for individual forecasting algorithms.
- anchor.ipynb: A research notebook containing the initial exploratory data analysis and model validation.

## Performance Evaluation
Model accuracy is measured and compared using standard statistical metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Author
Sanskruti Shah B.Tech Data Science | NMIMS
