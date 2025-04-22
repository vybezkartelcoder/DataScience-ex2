#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the stock market prediction pipeline.
"""

import os
import argparse
import subprocess
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run stock market prediction pipeline')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol to predict')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                        help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to predict')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'arima', 'random_forest', 'xgboost', 'all'],
                        help='Model to train and use for prediction')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection step')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive visualizations')
    return parser.parse_args()

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'-' * 80}")
    print(f"STEP: {description}")
    print(f"{'-' * 80}")
    print(f"Running command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False

def main():
    """Main function to run the stock prediction pipeline."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define models to train/predict
    models = []
    if args.model == 'all':
        models = ['lstm', 'arima', 'random_forest', 'xgboost']
    else:
        models = [args.model]
    
    # Start pipeline
    start_time = time.time()
    print(f"\n{'=' * 80}")
    print(f"STOCK MARKET PREDICTION PIPELINE - {args.symbol}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    
    # Step 1: Data Collection
    if not args.skip_collection:
        command = f"python src/data/collect_data.py --symbols {args.symbol} --start-date {args.start_date} --interval {args.interval}"
        if not run_command(command, "Data Collection"):
            return
    else:
        print("\nSkipping data collection step...")
    
    # Step 2: Data Preprocessing
    if not args.skip_preprocessing:
        command = f"python src/data/preprocess_data.py --symbol {args.symbol} --interval {args.interval}"
        if not run_command(command, "Data Preprocessing"):
            return
    else:
        print("\nSkipping data preprocessing step...")
    
    # Step 3: Model Training
    if not args.skip_training:
        for model in models:
            command = f"python src/models/train.py --model {model} --symbol {args.symbol} --interval {args.interval}"
            if not run_command(command, f"Training {model.upper()} Model"):
                continue
    else:
        print("\nSkipping model training step...")
    
    # Step 4: Make Predictions
    for model in models:
        command = f"python src/models/predict.py --model {model} --symbol {args.symbol} --interval {args.interval} --days {args.days}"
        if not run_command(command, f"Making Predictions with {model.upper()} Model"):
            continue
    
    # Step 5: Visualize Results
    for model in models:
        interactive_flag = "--interactive" if args.interactive else ""
        command = f"python src/visualization/visualize.py --symbol {args.symbol} --interval {args.interval} --model {model} {interactive_flag}"
        if not run_command(command, f"Visualizing Results for {model.upper()} Model"):
            continue
    
    # Pipeline completed
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n{'=' * 80}")
    print(f"PIPELINE COMPLETED - {args.symbol}")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"{'=' * 80}")
    
    # Print summary
    print("\nSummary of Results:")
    print(f"- Stock Symbol: {args.symbol}")
    print(f"- Data Interval: {args.interval}")
    print(f"- Prediction Horizon: {args.days} days")
    print(f"- Models Used: {', '.join([m.upper() for m in models])}")
    print("\nResults are available in the 'results' directory.")
    if args.interactive:
        print("Interactive visualizations are available as HTML files.")

if __name__ == "__main__":
    main() 