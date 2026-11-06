# Quantitative Financial Forecasting: Bidirectional LSTM

Lightweight PyTorch pipeline designed for financial time series analysis, asset log-return modeling, and multi-step recursive price forecasting. 

## Overview
This repository implements a deep learning architecture tailored for financial data, shifting away from raw price levels to stationary log-returns. It features a custom training loop with early stopping, sliding-window sequence generation, and a recursive forecasting engine to project future paths.

## Key Architecture & Pipeline
1. **Data Ingestion & Stationarity**: Fetches historical data via `yfinance` and converts closing prices into log-returns (ln(P_t / P_{t-1})) to ensure statistical stability.
2. **Sequence Engineering**: Applies a sliding window (configurable sequence length, e.g., 180 steps) with a robust `StandardScaler` fitted strictly on the training split to prevent look-ahead bias.
3. **Deep Learning Model**: 
   - **Backbone**: 2-layer Bidirectional LSTM (`hidden_size=64`, `dropout=0.2`) to capture temporal dependencies in both forward and backward directions.
   - **Criterion & Optimization**: L1 Loss (`nn.L1Loss`) for outlier resilience, optimized via Adam.
   - **Regularization**: Custom `EarlyStopping` class tracking training loss convergence.
4. **Recursive Multi-Step Projection**: Feeds model predictions back into the rolling window tensor to iteratively forecast a custom horizon (e.g., N business days ahead).
5. **Diagnostics**: Automated visualization suite plotting training RMSE convergence alongside actual vs. reconstructed test data and future projections.

## Tech Stack
- **Core**: Python
- **Deep Learning**: PyTorch (`torch.nn`, `torch.utils.data`)
- **Data Processing**: Pandas, NumPy, Scikit-Learn
- **Market Data**: `yfinance`
- **Visualization**: Matplotlib

## Usage

Clone the repository and run the main script:

```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib
python Beta_model.py
