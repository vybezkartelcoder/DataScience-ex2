#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train models for stock market prediction.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train stock price prediction models')
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'arima', 'random_forest', 'xgboost'],
                        help='Model type to train')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol to train on')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Window size for time series data')
    parser.add_argument('--input-dir', type=str, default='../../data/processed',
                        help='Input directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='../../models',
                        help='Output directory for trained models')
    parser.add_argument('--features', nargs='+', 
                        default=['open', 'high', 'low', 'close', 'volume'],
                        help='Features to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for deep learning models')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for deep learning models')
    return parser.parse_args()

def load_data(file_path):
    """Load preprocessed sequence data."""
    try:
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def build_lstm_model(input_shape, output_shape=10):
    """Build an LSTM model for stock price prediction."""
    model = Sequential()
    
    # First LSTM layer with return sequences for stacking
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(units=25))
    model.add(Dense(units=output_shape))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(X, y, epochs=100, batch_size=32, output_dir='../../models', symbol='STOCK', interval='1d'):
    """Train an LSTM model and save it."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]
    
    # Build model
    model = build_lstm_model(input_shape, output_shape)
    
    # Define callbacks
    model_path = os.path.join(output_dir, f"{symbol}_{interval}_lstm.h5")
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min', verbose=1)
    
    # Train model
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Evaluate model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{symbol}_{interval}_lstm_history.npy")
    np.save(history_path, history.history)
    
    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, f"{symbol}_{interval}_lstm_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
    
    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")
    print(f"Metrics saved to {metrics_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{symbol}_{interval}_lstm_loss.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    
    return model, history

def main():
    """Main function to train stock price prediction models."""
    args = parse_args()
    
    # Construct input file path
    input_file = os.path.join(args.input_dir, f"{args.symbol}_{args.interval}_sequences_{args.window_size}.npz")
    
    # Load data
    X, y = load_data(input_file)
    if X is None or y is None:
        return
        
    print(f"Loaded data from {input_file}, X shape: {X.shape}, y shape: {y.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model based on selected type
    if args.model == 'lstm':
        model, history = train_lstm_model(
            X, y, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            output_dir=args.output_dir,
            symbol=args.symbol,
            interval=args.interval
        )
    elif args.model == 'arima':
        # Import here to avoid dependencies if not using this model
        from pmdarima import auto_arima
        
        # For ARIMA, we need the original time series data, not sequences
        # Load the feature data instead
        feature_file = os.path.join(args.input_dir, f"{args.symbol}_{args.interval}_features.csv")
        df = pd.read_csv(feature_file)
        
        # Use only the Close price for ARIMA
        data = df['Close'].values
        
        # Split data into train and test sets
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        
        # Train ARIMA model
        print("Training ARIMA model...")
        model = auto_arima(
            train, 
            seasonal=True, 
            m=5,
            suppress_warnings=True, 
            error_action="ignore", 
            trace=True
        )
        
        # Make predictions
        n_periods = len(test)
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
        
        # Calculate metrics
        mse = mean_squared_error(test, forecast)
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mse)
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_arima.pkl")
        joblib.dump(model, model_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_arima_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
        
        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(train)), train, label='Training Data')
        plt.plot(range(len(train), len(train) + len(test)), test, label='Actual Values')
        plt.plot(range(len(train), len(train) + len(test)), forecast, label='Forecast', color='red')
        plt.fill_between(
            range(len(train), len(train) + len(test)),
            conf_int[:, 0], conf_int[:, 1],
            color='pink', alpha=0.3
        )
        plt.title('ARIMA Forecast')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_arima_forecast.png")
        plt.savefig(plot_path)
        print(f"Forecast plot saved to {plot_path}")
        
    elif args.model == 'random_forest':
        # Import here to avoid dependencies if not using this model
        from sklearn.ensemble import RandomForestRegressor
        
        # For Random Forest, reshape the data
        X_rf = X.reshape(X.shape[0], -1)  # Flatten the sequences
        
        # Use only the first column of y (Close price)
        y_rf = y[:, 0]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, shuffle=False)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_rf.pkl")
        joblib.dump(model, model_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_rf_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
        
        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Values')
        plt.plot(y_pred, label='Predictions', color='red')
        plt.title('Random Forest Predictions')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_rf_predictions.png")
        plt.savefig(plot_path)
        print(f"Predictions plot saved to {plot_path}")
        
    elif args.model == 'xgboost':
        # Import here to avoid dependencies if not using this model
        import xgboost as xgb
        
        # For XGBoost, reshape the data
        X_xgb = X.reshape(X.shape[0], -1)  # Flatten the sequences
        
        # Use only the first column of y (Close price)
        y_xgb = y[:, 0]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)
        
        # Train XGBoost model
        print("Training XGBoost model...")
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_xgb.json")
        model.save_model(model_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_xgb_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
        
        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Values')
        plt.plot(y_pred, label='Predictions', color='red')
        plt.title('XGBoost Predictions')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(args.output_dir, f"{args.symbol}_{args.interval}_xgb_predictions.png")
        plt.savefig(plot_path)
        print(f"Predictions plot saved to {plot_path}")
    
    print("Model training completed!")

if __name__ == "__main__":
    main() 