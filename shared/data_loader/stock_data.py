"""
Stock Data Loader Module
========================

This module provides utilities for loading stock market data from various sources.

Supported sources:
- Tushare: Chinese A-share market data (requires API token)
- AKShare: Free alternative for Chinese stock data
- CSV files: Load data from local CSV files

All loaders return pandas DataFrames with standardized column names:
- date: Trading date
- open: Opening price
- high: Highest price
- low: Lowest price
- close: Closing price
- volume: Trading volume
- amount: Trading amount (optional)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime, date
from abc import ABC, abstractmethod
import logging

# 配置日志
logger = logging.getLogger(__name__)


class StockDataLoader(ABC):
    """
    Abstract base class for stock data loaders.

    All data loaders should inherit from this class and implement
    the `load` method.
    """

    @abstractmethod
    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load stock data for a given symbol.

        Args:
            symbol: Stock symbol (e.g., '000001.SZ' for Tushare, '000001' for AKShare)
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Stock data with standardized columns
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and standardize DataFrame columns.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: Validated and standardized DataFrame
        """
        # 必需列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 确保日期列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        return df


class TushareLoader(StockDataLoader):
    """
    Tushare data loader for Chinese A-share market data.

    Requires Tushare API token. Install with: pip install tushare

    Example:
        >>> loader = TushareLoader(token='your_token_here')
        >>> df = loader.load('000001.SZ', start_date='2024-01-01')
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize Tushare loader.

        Args:
            token: Tushare API token (optional, can be set via environment variable)
        """
        self.token = token
        self._pro = None

    def _get_pro_api(self):
        """获取 Tushare Pro API 实例"""
        if self._pro is None:
            try:
                import tushare as ts
                if self.token:
                    ts.set_token(self.token)
                self._pro = ts.pro_api()
            except ImportError:
                raise ImportError(
                    "tushare is not installed. Install with: pip install tushare"
                )
        return self._pro

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load stock data from Tushare.

        Args:
            symbol: Stock symbol (e.g., '000001.SZ')
            start_date: Start date in 'YYYYMMDD' or 'YYYY-MM-DD' format
            end_date: End date in 'YYYYMMDD' or 'YYYY-MM-DD' format
            **kwargs: Additional parameters for Tushare API

        Returns:
            pd.DataFrame: Stock data with standardized columns
        """
        pro = self._get_pro_api()

        # 格式化日期
        start = start_date.replace('-', '') if start_date else None
        end = end_date.replace('-', '') if end_date else None

        # 获取日线数据
        df = pro.daily(
            ts_code=symbol,
            start_date=start,
            end_date=end,
            **kwargs
        )

        if df is None or df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # 重命名列
        df = df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume',
        })

        # 选择需要的列
        columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        available_columns = [col for col in columns if col in df.columns]
        df = df[available_columns]

        return self.validate_dataframe(df)


class AKShareLoader(StockDataLoader):
    """
    AKShare data loader for Chinese stock market data.

    Free alternative to Tushare. Install with: pip install akshare

    Example:
        >>> loader = AKShareLoader()
        >>> df = loader.load('000001', start_date='2024-01-01')
    """

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load stock data from AKShare.

        Args:
            symbol: Stock symbol (e.g., '000001')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Stock data with standardized columns
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError(
                "akshare is not installed. Install with: pip install akshare"
            )

        # 获取日线数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period='daily',
            start_date=start_date.replace('-', '') if start_date else '19900101',
            end_date=end_date.replace('-', '') if end_date else '20500101',
            adjust='qfq',  # 前复权
            **kwargs
        )

        if df is None or df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
        })

        # 选择需要的列
        columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        available_columns = [col for col in columns if col in df.columns]
        df = df[available_columns]

        return self.validate_dataframe(df)


def load_stock_data(
    source: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load stock data from specified source.

    Args:
        source: Data source ('tushare', 'akshare', or 'csv')
        symbol: Stock symbol or file path
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        **kwargs: Additional parameters

    Returns:
        pd.DataFrame: Stock data with standardized columns

    Example:
        >>> df = load_stock_data('akshare', '000001', start_date='2024-01-01')
    """
    loaders = {
        'tushare': TushareLoader,
        'akshare': AKShareLoader,
    }

    if source not in loaders:
        raise ValueError(f"Unknown source: {source}. Available: {list(loaders.keys())}")

    loader = loaders[source](**kwargs)
    return loader.load(symbol, start_date=start_date, end_date=end_date)


def load_from_csv(
    file_path: Union[str, Path],
    date_column: str = 'date',
    **kwargs
) -> pd.DataFrame:
    """
    Load stock data from CSV file.

    Args:
        file_path: Path to CSV file
        date_column: Name of the date column (default: 'date')
        **kwargs: Additional parameters for pd.read_csv

    Returns:
        pd.DataFrame: Stock data with standardized columns

    Example:
        >>> df = load_from_csv('data/stock_data.csv')
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 读取 CSV 文件
    df = pd.read_csv(file_path, **kwargs)

    # 如果日期列名不同，重命名
    if date_column != 'date' and date_column in df.columns:
        df = df.rename(columns={date_column: 'date'})

    # 标准化列名（小写）
    df.columns = df.columns.str.lower()

    return df


def load_from_tushare(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load data from Tushare.

    Args:
        symbol: Stock symbol (e.g., '000001.SZ')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        token: Tushare API token
        **kwargs: Additional parameters

    Returns:
        pd.DataFrame: Stock data with standardized columns
    """
    loader = TushareLoader(token=token)
    return loader.load(symbol, start_date=start_date, end_date=end_date, **kwargs)


def load_from_akshare(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load data from AKShare.

    Args:
        symbol: Stock symbol (e.g., '000001')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        **kwargs: Additional parameters

    Returns:
        pd.DataFrame: Stock data with standardized columns
    """
    loader = AKShareLoader()
    return loader.load(symbol, start_date=start_date, end_date=end_date, **kwargs)
