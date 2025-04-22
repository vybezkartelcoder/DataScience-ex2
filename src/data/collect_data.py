#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download historical stock market data using yfinance.
"""

import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download historical stock data')
    parser.add_argument('--symbols', nargs='+', required=True, 
                        help='List of stock symbols to download')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                        help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--output-dir', type=str, default='../../data/raw',
                        help='Output directory for data')
    return parser.parse_args()

def download_stock_data(symbol, start_date, end_date, interval='1d'):
    """Download historical stock data for a single symbol."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure the DataFrame has expected columns
        if 'Date' in df.columns and 'Open' in df.columns:
            print(f"Successfully downloaded data for {symbol}")
            return df
        else:
            print(f"Error: Retrieved data for {symbol} doesn't have expected columns")
            return None
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def main():
    """Main function to download and save stock data."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for symbol in args.symbols:
        # Download data
        df = download_stock_data(symbol, args.start_date, args.end_date, args.interval)
        
        if df is not None:
            # Save to CSV
            output_file = os.path.join(args.output_dir, f"{symbol}_{args.interval}.csv")
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            
    print("Data collection completed!")

if __name__ == "__main__":
    main() 