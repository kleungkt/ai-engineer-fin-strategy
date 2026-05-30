"""
Data Loader Module
==================

This module provides utilities for loading stock market data from various sources.

Supported data sources:
- Tushare: Chinese stock market data
- AKShare: Chinese and international stock market data
- Local CSV files

Usage:
    from shared.data_loader import load_stock_data, load_from_csv
    from shared.data_loader.stock_data import TushareLoader, AKShareLoader
"""

from .stock_data import (
    load_stock_data,
    load_from_csv,
    load_from_tushare,
    load_from_akshare,
    TushareLoader,
    AKShareLoader,
    StockDataLoader,
)

__all__ = [
    "load_stock_data",
    "load_from_csv",
    "load_from_tushare",
    "load_from_akshare",
    "TushareLoader",
    "AKShareLoader",
    "StockDataLoader",
]
