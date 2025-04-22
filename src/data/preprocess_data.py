#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to preprocess stock market data.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess stock market data')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol to preprocess')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--input-dir', type=str, default='../../data/raw',
                        help='Input directory containing raw data')
    parser.add_argument('--output-dir', type=str, default='../../data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Window size for time series data')
    return parser.parse_args()

def load_data(file_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for feature engineering."""
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI) - 14 period
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Logarithmic Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Remove rows with NaN values resulting from calculations
    df = df.dropna()
    
    return df

def create_sequences(data, window_size):
    """Create sequences for time series prediction."""
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)

def normalize_data(df):
    """Normalize the data using MinMaxScaler."""
    scaler = MinMaxScaler()
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Scale numeric data
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled, scaler

def main():
    """Main function to preprocess stock data."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct input file path
    input_file = os.path.join(args.input_dir, f"{args.symbol}_{args.interval}.csv")
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return
        
    print(f"Loaded data from {input_file}, shape: {df.shape}")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    print(f"Calculated technical indicators, shape: {df.shape}")
    
    # Save processed data with indicators
    output_features_file = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_features.csv")
    df.to_csv(output_features_file, index=False)
    print(f"Saved processed data with features to {output_features_file}")
    
    # Normalize data
    df_normalized, scaler = normalize_data(df)
    
    # Select features for sequence creation
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'MA_5', 'MA_20', 'RSI', 'MACD', 'Log_Return']
    
    # Filter DataFrame to selected features
    df_selected = df_normalized[features]
    
    # Create sequences
    X, y = create_sequences(df_selected.values, args.window_size)
    
    # Save sequences for model training
    output_seq_file = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_sequences_{args.window_size}.npz")
    np.savez(output_seq_file, X=X, y=y)
    print(f"Saved sequences to {output_seq_file}, X shape: {X.shape}, y shape: {y.shape}")
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main() 