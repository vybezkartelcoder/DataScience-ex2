#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to visualize stock market data and predictions.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize stock market data and predictions')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--model', type=str, default='lstm',
                        help='Model type (lstm, arima, rf, xgb)')
    parser.add_argument('--input-dir', type=str, default='../../data/processed',
                        help='Input directory containing processed data')
    parser.add_argument('--pred-dir', type=str, default='../../results',
                        help='Directory containing prediction results')
    parser.add_argument('--output-dir', type=str, default='../../results',
                        help='Output directory for visualizations')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive Plotly visualizations')
    return parser.parse_args()

def load_data(file_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_predictions(file_path):
    """Load prediction data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

def plot_stock_history(df, symbol, output_dir):
    """Plot historical stock data."""
    plt.figure(figsize=(14, 8))
    
    # Plot stock price
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.plot(df['Date'], df['MA_20'], label='20-day MA', alpha=0.7)
    plt.plot(df['Date'], df['MA_50'], label='50-day MA', alpha=0.7)
    plt.title(f'{symbol} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot volume
    plt.subplot(2, 1, 2)
    plt.bar(df['Date'], df['Volume'], alpha=0.7, color='blue')
    plt.title(f'{symbol} Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{symbol}_history.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved stock history plot to {output_file}")
    
    return output_file

def plot_technical_indicators(df, symbol, output_dir):
    """Plot technical indicators."""
    plt.figure(figsize=(14, 12))
    
    # RSI
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.title(f'{symbol} RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MACD
    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['Date'], df['MACD_signal'], label='Signal Line', color='red')
    plt.title(f'{symbol} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log Returns
    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['Log_Return'], label='Log Return', color='green')
    plt.title(f'{symbol} Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{symbol}_technical.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved technical indicators plot to {output_file}")
    
    return output_file

def plot_predictions_comparison(df, predictions, symbol, model, output_dir):
    """Plot comparison between actual and predicted prices."""
    # Find overlapping dates
    merged_df = pd.merge(df, predictions, on='Date', how='inner')
    
    if merged_df.empty:
        print("No overlapping dates between historical data and predictions")
        # Plot just the latest historical data and future predictions
        last_date = df['Date'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        
        plt.figure(figsize=(14, 7))
        
        # Plot the last 30 days of historical data
        last_n_days = min(30, len(df))
        plt.plot(df['Date'].iloc[-last_n_days:], 
                df['Close'].iloc[-last_n_days:], 
                label='Historical Data', color='blue')
        
        # Add a point connecting historical data to predictions
        dates_to_plot = [last_date] + predictions['Date'].tolist()
        values_to_plot = [last_close] + predictions['Predicted_Close'].tolist()
        plt.plot(dates_to_plot, values_to_plot, 
                label=f'Predicted Data ({model.upper()})', 
                color='red', linestyle='--')
        
        plt.title(f'{symbol} Stock Price Prediction ({model.upper()})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    else:
        # Plot comparison
        plt.figure(figsize=(14, 7))
        plt.plot(merged_df['Date'], merged_df['Close'], label='Actual', color='blue')
        plt.plot(merged_df['Date'], merged_df['Predicted_Close'], 
                label=f'Predicted ({model.upper()})', 
                color='red', linestyle='--')
        
        # Add confidence intervals if available
        if 'Lower_CI' in merged_df.columns and 'Upper_CI' in merged_df.columns:
            plt.fill_between(
                merged_df['Date'],
                merged_df['Lower_CI'],
                merged_df['Upper_CI'],
                color='red', alpha=0.2,
                label='95% Confidence Interval'
            )
        
        plt.title(f'{symbol} Stock Price - Actual vs Predicted ({model.upper()})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{symbol}_{model}_comparison.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved prediction comparison plot to {output_file}")
    
    return output_file

def create_interactive_candlestick(df, predictions, symbol, model, output_dir):
    """Create an interactive candlestick chart with predictions."""
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       subplot_titles=(f'{symbol} Stock Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Add candlestick chart for historical data
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Historical Data'
    ), row=1, col=1)
    
    # Add MA lines
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA_20'],
        name='20-day MA',
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA_50'],
        name='50-day MA',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    # Add predictions
    # Connect last historical point to first prediction
    last_date = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    connect_dates = [last_date] + predictions['Date'].tolist()
    connect_values = [last_close] + predictions['Predicted_Close'].tolist()
    
    fig.add_trace(go.Scatter(
        x=connect_dates,
        y=connect_values,
        name=f'Predictions ({model.upper()})',
        line=dict(color='red', width=2, dash='dash')
    ), row=1, col=1)
    
    # Add confidence interval if available
    if 'Lower_CI' in predictions.columns and 'Upper_CI' in predictions.columns:
        # Add lower and upper bounds
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Upper_CI'],
            name='Upper CI',
            line=dict(width=0),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Lower_CI'],
            name='Lower CI',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            showlegend=True
        ), row=1, col=1)
    
    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker=dict(color='blue', opacity=0.5)
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price with {model.upper()} Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save as HTML
    output_file = os.path.join(output_dir, f"{symbol}_{model}_interactive.html")
    fig.write_html(output_file)
    print(f"Saved interactive chart to {output_file}")
    
    return output_file

def main():
    """Main function to visualize stock data and predictions."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct file paths
    data_file = os.path.join(args.input_dir, f"{args.symbol}_{args.interval}_features.csv")
    pred_file = os.path.join(args.pred_dir, f"{args.symbol}_{args.interval}_{args.model}_predictions.csv")
    
    # Load data
    df = load_data(data_file)
    if df is None:
        print(f"Could not load data from {data_file}")
        return
    
    predictions = load_predictions(pred_file)
    if predictions is None:
        print(f"Could not load predictions from {pred_file}")
        return
    
    print(f"Loaded data and predictions for {args.symbol}")
    
    # Generate visualizations
    plot_stock_history(df, args.symbol, args.output_dir)
    plot_technical_indicators(df, args.symbol, args.output_dir)
    plot_predictions_comparison(df, predictions, args.symbol, args.model, args.output_dir)
    
    # Generate interactive visualization if requested
    if args.interactive:
        create_interactive_candlestick(df, predictions, args.symbol, args.model, args.output_dir)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main() 