#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to make predictions using trained stock market models.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make stock price predictions')
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'arima', 'random_forest', 'xgboost'],
                        help='Model type to use for prediction')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol to predict')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to predict into the future')
    parser.add_argument('--model-dir', type=str, default='../../models',
                        help='Directory containing trained models')
    parser.add_argument('--data-dir', type=str, default='../../data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='../../results',
                        help='Output directory for predictions')
    return parser.parse_args()

def load_latest_data(symbol, days=100, interval='1d'):
    """Load the latest stock data for prediction."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        return df
    except Exception as e:
        print(f"Error loading latest data: {e}")
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
    
    # Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Logarithmic Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Remove rows with NaN values resulting from calculations
    df = df.dropna()
    
    return df

def prepare_data_for_lstm(df, window_size=60):
    """Prepare data for LSTM prediction."""
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'MA_5', 'MA_20', 'RSI', 'MACD', 'Log_Return']
    
    # Ensure all required features are present
    for feature in features:
        if feature not in df.columns:
            print(f"Missing feature: {feature}")
            return None
    
    # Filter DataFrame to selected features
    df_selected = df[features]
    
    # Normalize data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected),
        columns=df_selected.columns
    )
    
    # Create a single sequence for prediction
    sequence = df_scaled.values[-window_size:].reshape(1, window_size, len(features))
    
    return sequence, scaler, df

def predict_lstm(model_path, sequence, scaler, original_df, days=30):
    """Make predictions using the LSTM model."""
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Get the last date in the original data
        last_date = original_df['Date'].iloc[-1]
        
        # Initialize predictions list
        predictions = []
        current_sequence = sequence.copy()
        
        # Generate predictions for the next 'days'
        for i in range(days):
            # Make prediction
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0])
            
            # Update the sequence by removing the first element and adding the prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = pred[0]
        
        # Convert predictions to a DataFrame
        pred_dates = [(last_date + timedelta(days=i+1)) for i in range(days)]
        
        # Extract the Close price (index 3 in our feature list)
        close_index = 3  # Index of Close in the features list
        pred_close = [pred[close_index] for pred in predictions]
        
        # Create a DataFrame with the predictions
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Close': pred_close
        })
        
        return pred_df
    
    except Exception as e:
        print(f"Error making LSTM predictions: {e}")
        return None

def predict_arima(model_path, original_df, days=30):
    """Make predictions using the ARIMA model."""
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Get the last date in the original data
        last_date = original_df['Date'].iloc[-1]
        
        # Generate predictions
        forecast, conf_int = model.predict(n_periods=days, return_conf_int=True)
        
        # Create a DataFrame with the predictions
        pred_dates = [(last_date + timedelta(days=i+1)) for i in range(days)]
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Close': forecast,
            'Lower_CI': conf_int[:, 0],
            'Upper_CI': conf_int[:, 1]
        })
        
        return pred_df
    
    except Exception as e:
        print(f"Error making ARIMA predictions: {e}")
        return None

def predict_rf_or_xgb(model_path, df, window_size=60, days=30, model_type='rf'):
    """Make predictions using the Random Forest or XGBoost model."""
    try:
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA_5', 'MA_20', 'RSI', 'MACD', 'Log_Return']
        
        # Ensure all required features are present
        for feature in features:
            if feature not in df.columns:
                print(f"Missing feature: {feature}")
                return None
        
        # Filter DataFrame to selected features
        df_selected = df[features]
        
        # Normalize data
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_selected),
            columns=df_selected.columns
        )
        
        # Get the last date in the original data
        last_date = df['Date'].iloc[-1]
        
        # Load the model
        if model_type == 'rf':
            model = joblib.load(model_path)
        else:  # xgboost
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_path)
        
        # Create the sequence
        latest_sequence = df_scaled.values[-window_size:].flatten().reshape(1, -1)
        
        # Initialize predictions list
        predictions = []
        current_sequence = latest_sequence.copy()
        
        # Generate predictions for the next 'days'
        for i in range(days):
            # Make prediction (the model predicts only the Close price)
            pred = model.predict(current_sequence)[0]
            predictions.append(pred)
            
            # Update the sequence by removing the first 10 elements (1 row of features)
            # and adding the new prediction at the end
            # Note: This is a simplified approach; ideally, you'd update all features
            num_features = len(features)
            current_sequence = np.roll(current_sequence, -num_features)
            current_sequence[0, -1] = pred
        
        # Create a DataFrame with the predictions
        pred_dates = [(last_date + timedelta(days=i+1)) for i in range(days)]
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Close': predictions
        })
        
        return pred_df
    
    except Exception as e:
        print(f"Error making {model_type.upper()} predictions: {e}")
        return None

def main():
    """Main function to make stock price predictions."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the latest data
    df = load_latest_data(args.symbol, days=100, interval=args.interval)
    if df is None:
        return
    
    print(f"Loaded latest data for {args.symbol}, shape: {df.shape}")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    print(f"Calculated technical indicators, shape: {df.shape}")
    
    # Construct model path
    model_path = os.path.join(args.model_dir, f"{args.symbol}_{args.interval}_{args.model}")
    if args.model == 'lstm':
        model_path += '.h5'
    elif args.model == 'arima':
        model_path += '.pkl'
    elif args.model == 'random_forest':
        model_path = model_path.replace('random_forest', 'rf') + '.pkl'
    elif args.model == 'xgboost':
        model_path = model_path.replace('xgboost', 'xgb') + '.json'
    
    # Make predictions based on model type
    if args.model == 'lstm':
        # Prepare data for LSTM
        sequence, scaler, df = prepare_data_for_lstm(df, window_size=60)
        if sequence is None:
            return
        
        # Make predictions
        predictions = predict_lstm(model_path, sequence, scaler, df, days=args.days)
    
    elif args.model == 'arima':
        # Make predictions
        predictions = predict_arima(model_path, df, days=args.days)
    
    elif args.model == 'random_forest':
        # Make predictions
        predictions = predict_rf_or_xgb(model_path, df, window_size=60, days=args.days, model_type='rf')
    
    elif args.model == 'xgboost':
        # Make predictions
        predictions = predict_rf_or_xgb(model_path, df, window_size=60, days=args.days, model_type='xgb')
    
    if predictions is None:
        print(f"Failed to make predictions with {args.model} model")
        return
    
    print(f"Generated predictions for the next {args.days} days")
    
    # Save predictions to CSV
    output_file = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_{args.model}_predictions.csv")
    predictions.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")
    
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    
    # Plot the historical data
    last_n_days = min(30, len(df))
    plt.plot(df['Date'].iloc[-last_n_days:], df['Close'].iloc[-last_n_days:], label='Historical Close')
    
    # Plot the predictions
    plt.plot(predictions['Date'], predictions['Predicted_Close'], label='Predicted Close', color='red')
    
    # Add confidence intervals if available (ARIMA)
    if 'Lower_CI' in predictions.columns and 'Upper_CI' in predictions.columns:
        plt.fill_between(
            predictions['Date'],
            predictions['Lower_CI'],
            predictions['Upper_CI'],
            color='pink', alpha=0.3
        )
    
    plt.title(f'{args.symbol} Stock Price Prediction ({args.model.upper()})')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_{args.model}_predictions.png")
    plt.savefig(plot_file)
    print(f"Saved prediction plot to {plot_file}")
    
    print("Prediction completed!")

if __name__ == "__main__":
    main() 