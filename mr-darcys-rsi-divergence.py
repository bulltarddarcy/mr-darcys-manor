import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    """
    Standard RSI calculation. 
    Calculating on the full series ensures the 'smoothing' aspect of RSI 
    is consistent with desktop trading platforms.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(df, rsi_period=14, ema_period=50):
    """
    The 'Recommended Fix' implementation:
    1. Calculate indicators on the RAW dataframe.
    2. Slice the data ONLY after indicators are populated.
    """
    # Create a copy to avoid SettingWithCopy warnings
    data = df.copy()

    # --- STEP 1: Calculate Indicators on the FULL dataframe ---
    # This preserves the historical context needed for EMA and RSI warm-up
    data['RSI'] = calculate_rsi(data['close'], period=rsi_period)
    data['EMA'] = data['close'].ewm(span=ema_period, adjust=False).mean()

    # --- STEP 2: Drop NaN values or Slice for Signal Lookback ---
    # We drop only after the calculations are done to ensure the first 
    # visible signal has a 'warm' indicator value.
    final_df = data.dropna().copy()

    # Logic for signals (Example: RSI Cross)
    final_df['signal'] = 0
    final_df.loc[(final_df['RSI'] < 30) & (final_df['close'] > final_df['EMA']), 'signal'] = 1  # Buy
    final_df.loc[(final_df['RSI'] > 70), 'signal'] = -1 # Sell

    return final_df

# Example usage with mock data
if __name__ == "__main__":
    # Simulate 200 days of price data
    dates = pd.date_range(start='2023-01-01', periods=200)
    prices = np.cumsum(np.random.randn(200)) + 100
    mock_data = pd.DataFrame({'close': prices}, index=dates)

    # Generate signals using the fix
    processed_data = generate_signals(mock_data)

    print("First 5 rows of processed data (indicators are now 'warm'):")
    print(processed_data[['close', 'RSI', 'EMA', 'signal']].head())
