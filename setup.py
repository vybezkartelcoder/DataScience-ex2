#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Stock Market Price Prediction project.
"""

from setuptools import setup, find_packages

setup(
    name="stock_price_prediction",
    version="0.1.0",
    description="A data science project for predicting stock market prices",
    author="User",
    author_email="user@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "scikit-learn>=1.2.2",
        "statsmodels>=0.14.0",
        "tensorflow>=2.12.0",
        "keras>=2.12.0",
        "yfinance>=0.2.18",
        "pandas-datareader>=0.10.0",
        "plotly>=5.14.1",
        "jupyter>=1.0.0",
        "pmdarima>=2.0.3",
        "xgboost>=1.7.5",
    ],
    python_requires=">=3.8",
) 