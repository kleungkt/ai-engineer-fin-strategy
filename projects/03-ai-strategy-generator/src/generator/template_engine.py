"""
Template Engine
===============

Pre-built strategy templates for common quantitative trading patterns.

Each template is a complete, runnable Python strategy class that follows
the standard interface: `generate_signals(self, data: pd.DataFrame) -> pd.DataFrame`

Supported strategy types:
- momentum_breakout: 动量突破策略
- dual_ma_crossover: 双均线交叉策略
- bollinger_rsi: 布林通道+RSI策略
- rsi_extreme: RSI极端值策略
- macd_signal: MACD信号策略
- mean_reversion: 均值回归策略
- turtle_trading: 海龟交易策略
"""

from typing import Dict, List
from string import Template


# ============================================
# 策略模板定义
# ============================================

STRATEGY_TEMPLATES: Dict[str, str] = {
    # ------------------------------------------
    # 1. 动量突破策略
    # ------------------------------------------
    "momentum_breakout": '''\
import pandas as pd
import numpy as np


class MomentumBreakout:
    """动量突破策略

    当价格突破N日最高价时买入，跌破M日最低价时卖出。
    使用ATR进行动态止损。

    参数:
        entry_period: 突破周期 (默认: $entry_period)
        exit_period: 退出周期 (默认: $exit_period)
        stop_loss_atr: ATR止损倍数 (默认: $stop_loss_atr)
        atr_period: ATR计算周期 (默认: $atr_period)
    """

    def __init__(self):
        self.params = {
            "entry_period": $entry_period,
            "exit_period": $exit_period,
            "stop_loss_atr": $stop_loss_atr,
            "atr_period": $atr_period,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame,
                  需要列: open, high, low, close, volume

        Returns:
            添加了 'signal' 列的DataFrame
                signal: 1=买入, -1=卖出, 0=持有
        """
        df = data.copy()

        # 计算N日最高价和最低价
        high_n = df["high"].rolling(window=self.params["entry_period"]).max()
        low_n = df["low"].rolling(window=self.params["exit_period"]).min()

        # 计算ATR用于止损
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": abs(df["high"] - df["close"].shift(1)),
            "lc": abs(df["low"] - df["close"].shift(1)),
        }).max(axis=1)
        atr = tr.rolling(window=self.params["atr_period"]).mean()

        # 初始化信号列
        df["signal"] = 0

        # 买入信号: 收盘价突破N日最高价
        df.loc[df["close"] > high_n.shift(1), "signal"] = 1

        # 卖出信号: 收盘价跌破M日最低价
        df.loc[df["close"] < low_n.shift(1), "signal"] = -1

        # 保存ATR用于止损计算
        df["atr"] = atr
        df["stop_loss_distance"] = atr * self.params["stop_loss_atr"]

        return df
''',

    # ------------------------------------------
    # 2. 双均线交叉策略
    # ------------------------------------------
    "dual_ma_crossover": '''\
import pandas as pd
import numpy as np


class DualMACrossover:
    """双均线交叉策略

    当短期均线上穿长期均线时买入(金叉)，
    当短期均线下穿长期均线时卖出(死叉)。

    参数:
        fast_period: 短期均线周期 (默认: $fast_period)
        slow_period: 长期均线周期 (默认: $slow_period)
        ma_type: 均线类型 'sma' 或 'ema' (默认: $ma_type)
    """

    def __init__(self):
        self.params = {
            "fast_period": $fast_period,
            "slow_period": $slow_period,
            "ma_type": "$ma_type",
        }

    def _calc_ma(self, series: pd.Series, period: int) -> pd.Series:
        """计算移动平均线"""
        if self.params["ma_type"] == "ema":
            return series.ewm(span=period, adjust=False).mean()
        return series.rolling(window=period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame

        Returns:
            添加了 'signal' 列的DataFrame
        """
        df = data.copy()

        # 计算快慢均线
        df["ma_fast"] = self._calc_ma(df["close"], self.params["fast_period"])
        df["ma_slow"] = self._calc_ma(df["close"], self.params["slow_period"])

        # 初始化信号列
        df["signal"] = 0

        # 金叉: 快线从下方穿越慢线 → 买入
        cross_up = (df["ma_fast"] > df["ma_slow"]) & (
            df["ma_fast"].shift(1) <= df["ma_slow"].shift(1)
        )
        df.loc[cross_up, "signal"] = 1

        # 死叉: 快线从上方穿越慢线 → 卖出
        cross_down = (df["ma_fast"] < df["ma_slow"]) & (
            df["ma_fast"].shift(1) >= df["ma_slow"].shift(1)
        )
        df.loc[cross_down, "signal"] = -1

        return df
''',

    # ------------------------------------------
    # 3. 布林通道 + RSI 策略
    # ------------------------------------------
    "bollinger_rsi": '''\
import pandas as pd
import numpy as np


class BollingerRSI:
    """布林通道 + RSI 组合策略

    当价格触及布林通道下轨且RSI处于超卖区域时买入，
    当价格触及布林通道上轨且RSI处于超买区域时卖出。

    参数:
        bb_period: 布林通道周期 (默认: $bb_period)
        bb_std: 标准差倍数 (默认: $bb_std)
        rsi_period: RSI计算周期 (默认: $rsi_period)
        rsi_oversold: RSI超卖阈值 (默认: $rsi_oversold)
        rsi_overbought: RSI超买阈值 (默认: $rsi_overbought)
    """

    def __init__(self):
        self.params = {
            "bb_period": $bb_period,
            "bb_std": $bb_std,
            "rsi_period": $rsi_period,
            "rsi_oversold": $rsi_oversold,
            "rsi_overbought": $rsi_overbought,
        }

    def _calc_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame

        Returns:
            添加了 'signal' 列的DataFrame
        """
        df = data.copy()
        period = self.params["bb_period"]
        std_mult = self.params["bb_std"]

        # 计算布林通道
        df["bb_mid"] = df["close"].rolling(window=period).mean()
        bb_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_mid"] + std_mult * bb_std
        df["bb_lower"] = df["bb_mid"] - std_mult * bb_std

        # 计算RSI
        df["rsi"] = self._calc_rsi(df["close"], self.params["rsi_period"])

        # 初始化信号列
        df["signal"] = 0

        # 买入条件: 价格 <= 下轨 且 RSI < 超卖阈值
        buy_cond = (df["close"] <= df["bb_lower"]) & (
            df["rsi"] < self.params["rsi_oversold"]
        )
        df.loc[buy_cond, "signal"] = 1

        # 卖出条件: 价格 >= 上轨 且 RSI > 超买阈值
        sell_cond = (df["close"] >= df["bb_upper"]) & (
            df["rsi"] > self.params["rsi_overbought"]
        )
        df.loc[sell_cond, "signal"] = -1

        return df
''',

    # ------------------------------------------
    # 4. RSI 极端值策略
    # ------------------------------------------
    "rsi_extreme": '''\
import pandas as pd
import numpy as np


class RSIExtreme:
    """RSI 极端值策略

    当RSI低于超卖阈值时买入，高于超买阈值时卖出。
    使用中间区域作为中性区域，不产生信号。

    参数:
        rsi_period: RSI计算周期 (默认: $rsi_period)
        oversold: 超卖阈值 (默认: $oversold)
        overbought: 超买阈值 (默认: $overbought)
        exit_middle: 退出中位线 (默认: $exit_middle)
    """

    def __init__(self):
        self.params = {
            "rsi_period": $rsi_period,
            "oversold": $oversold,
            "overbought": $overbought,
            "exit_middle": $exit_middle,
        }

    def _calc_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame

        Returns:
            添加了 'signal' 列的DataFrame
        """
        df = data.copy()

        # 计算RSI
        df["rsi"] = self._calc_rsi(df["close"], self.params["rsi_period"])

        # 初始化信号列
        df["signal"] = 0

        # 买入信号: RSI低于超卖阈值
        df.loc[df["rsi"] < self.params["oversold"], "signal"] = 1

        # 卖出信号: RSI高于超买阈值
        df.loc[df["rsi"] > self.params["overbought"], "signal"] = -1

        return df
''',

    # ------------------------------------------
    # 5. MACD 信号策略
    # ------------------------------------------
    "macd_signal": '''\
import pandas as pd
import numpy as np


class MACDSignal:
    """MACD 信号策略

    当MACD线上穿信号线时买入，下穿时卖出。
    结合零轴位置判断趋势方向。

    参数:
        fast_period: 快速EMA周期 (默认: $fast_period)
        slow_period: 慢速EMA周期 (默认: $slow_period)
        signal_period: 信号线周期 (默认: $signal_period)
        above_zero_only: 是否仅在零轴上方做多 (默认: $above_zero_only)
    """

    def __init__(self):
        self.params = {
            "fast_period": $fast_period,
            "slow_period": $slow_period,
            "signal_period": $signal_period,
            "above_zero_only": $above_zero_only,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame

        Returns:
            添加了 'signal' 列的DataFrame
        """
        df = data.copy()

        # 计算MACD
        ema_fast = df["close"].ewm(
            span=self.params["fast_period"], adjust=False
        ).mean()
        ema_slow = df["close"].ewm(
            span=self.params["slow_period"], adjust=False
        ).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(
            span=self.params["signal_period"], adjust=False
        ).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 初始化信号列
        df["signal"] = 0

        # 金叉: MACD线上穿信号线
        cross_up = (df["macd"] > df["macd_signal"]) & (
            df["macd"].shift(1) <= df["macd_signal"].shift(1)
        )
        # 死叉: MACD线下穿信号线
        cross_down = (df["macd"] < df["macd_signal"]) & (
            df["macd"].shift(1) >= df["macd_signal"].shift(1)
        )

        if self.params["above_zero_only"]:
            # 仅在零轴上方做多，零轴下方做空
            df.loc[cross_up & (df["macd"] > 0), "signal"] = 1
            df.loc[cross_down & (df["macd"] < 0), "signal"] = -1
        else:
            df.loc[cross_up, "signal"] = 1
            df.loc[cross_down, "signal"] = -1

        return df
''',

    # ------------------------------------------
    # 6. 均值回归策略
    # ------------------------------------------
    "mean_reversion": '''\
import pandas as pd
import numpy as np


class MeanReversion:
    """均值回归策略

    当价格偏离均值超过N个标准差时入场，
    回归均值时出场。使用分批建仓和ATR动态止损。

    参数:
        lookback: 均值回看周期 (默认: $lookback)
        entry_std: 入场标准差倍数 (默认: $entry_std)
        exit_std: 出场标准差倍数 (默认: $exit_std)
        atr_period: ATR周期 (默认: $atr_period)
        atr_stop: ATR止损倍数 (默认: $atr_stop)
    """

    def __init__(self):
        self.params = {
            "lookback": $lookback,
            "entry_std": $entry_std,
            "exit_std": $exit_std,
            "atr_period": $atr_period,
            "atr_stop": $atr_stop,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame

        Returns:
            添加了 'signal' 列的DataFrame
        """
        df = data.copy()
        lookback = self.params["lookback"]

        # 计算均值和标准差
        df["mean"] = df["close"].rolling(window=lookback).mean()
        df["std"] = df["close"].rolling(window=lookback).std()
        df["z_score"] = (df["close"] - df["mean"]) / df["std"]

        # 计算ATR
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": abs(df["high"] - df["close"].shift(1)),
            "lc": abs(df["low"] - df["close"].shift(1)),
        }).max(axis=1)
        df["atr"] = tr.rolling(window=self.params["atr_period"]).mean()

        # 初始化信号列
        df["signal"] = 0

        # 买入: 价格偏离均值下方超过 entry_std 个标准差
        df.loc[df["z_score"] < -self.params["entry_std"], "signal"] = 1

        # 卖出: 价格偏离均值上方超过 entry_std 个标准差
        df.loc[df["z_score"] > self.params["entry_std"], "signal"] = -1

        # 止损距离
        df["stop_loss_distance"] = df["atr"] * self.params["atr_stop"]

        return df
''',

    # ------------------------------------------
    # 7. 海龟交易策略
    # ------------------------------------------
    "turtle_trading": '''\
import pandas as pd
import numpy as np


class TurtleTrading:
    """海龟交易策略

    经典趋势跟踪策略：
    - 入场: 价格突破N日最高价
    - 出场: 价格跌破M日最低价
    - 止损: 基于ATR的动态止损
    - 仓位: 基于ATR的头寸规模管理

    参数:
        entry_period: 入场突破周期 (默认: $entry_period)
        exit_period: 出场突破周期 (默认: $exit_period)
        atr_period: ATR计算周期 (默认: $atr_period)
        risk_per_trade: 每笔交易风险比例 (默认: $risk_per_trade)
        stop_loss_atr: ATR止损倍数 (默认: $stop_loss_atr)
    """

    def __init__(self):
        self.params = {
            "entry_period": $entry_period,
            "exit_period": $exit_period,
            "atr_period": $atr_period,
            "risk_per_trade": $risk_per_trade,
            "stop_loss_atr": $stop_loss_atr,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 包含 OHLCV 数据的DataFrame

        Returns:
            添加了 'signal' 列的DataFrame
        """
        df = data.copy()

        # 计算唐奇安通道（Donchian Channel）
        df["dc_upper"] = df["high"].rolling(
            window=self.params["entry_period"]
        ).max()
        df["dc_lower"] = df["low"].rolling(
            window=self.params["exit_period"]
        ).min()
        df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2

        # 计算ATR
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": abs(df["high"] - df["close"].shift(1)),
            "lc": abs(df["low"] - df["close"].shift(1)),
        }).max(axis=1)
        df["atr"] = tr.rolling(window=self.params["atr_period"]).mean()

        # 初始化信号列
        df["signal"] = 0

        # 入场: 收盘价突破N日最高价
        df.loc[
            df["close"] > df["dc_upper"].shift(1), "signal"
        ] = 1

        # 出场: 收盘价跌破M日最低价
        df.loc[
            df["close"] < df["dc_lower"].shift(1), "signal"
        ] = -1

        # 止损距离和头寸规模
        df["stop_loss_distance"] = df["atr"] * self.params["stop_loss_atr"]
        df["position_size"] = self.params["risk_per_trade"] / (
            df["stop_loss_distance"] / df["close"]
        )

        return df
''',
}


# ============================================
# 模板参数默认值
# ============================================

DEFAULT_PARAMS: Dict[str, Dict[str, str]] = {
    "momentum_breakout": {
        "entry_period": "20",
        "exit_period": "10",
        "stop_loss_atr": "2.0",
        "atr_period": "14",
    },
    "dual_ma_crossover": {
        "fast_period": "10",
        "slow_period": "30",
        "ma_type": "ema",
    },
    "bollinger_rsi": {
        "bb_period": "20",
        "bb_std": "2.0",
        "rsi_period": "14",
        "rsi_oversold": "30",
        "rsi_overbought": "70",
    },
    "rsi_extreme": {
        "rsi_period": "14",
        "oversold": "30",
        "overbought": "70",
        "exit_middle": "50",
    },
    "macd_signal": {
        "fast_period": "12",
        "slow_period": "26",
        "signal_period": "9",
        "above_zero_only": "True",
    },
    "mean_reversion": {
        "lookback": "20",
        "entry_std": "2.0",
        "exit_std": "0.5",
        "atr_period": "14",
        "atr_stop": "2.0",
    },
    "turtle_trading": {
        "entry_period": "20",
        "exit_period": "10",
        "atr_period": "20",
        "risk_per_trade": "0.01",
        "stop_loss_atr": "2.0",
    },
}


def render_template(template_name: str, params: Dict[str, str]) -> str:
    """Fill a strategy template with the given parameters.

    Args:
        template_name: Name of the template (e.g. 'momentum_breakout')
        params: Parameter values to substitute into the template

    Returns:
        Complete strategy code string with parameters filled in

    Raises:
        KeyError: If template_name is not found in STRATEGY_TEMPLATES
        ValueError: If required template variables are missing from params

    Example:
        >>> code = render_template("dual_ma_crossover", {
        ...     "fast_period": "5",
        ...     "slow_period": "20",
        ...     "ma_type": "sma",
        ... })
    """
    if template_name not in STRATEGY_TEMPLATES:
        available = ", ".join(sorted(STRATEGY_TEMPLATES.keys()))
        raise KeyError(
            f"Template '{template_name}' not found. "
            f"Available templates: {available}"
        )

    # 使用默认参数填充缺失值
    merged_params = dict(DEFAULT_PARAMS.get(template_name, {}))
    merged_params.update(params)

    template = Template(STRATEGY_TEMPLATES[template_name])
    try:
        return template.substitute(merged_params)
    except KeyError as e:
        raise ValueError(
            f"Missing required parameter {e} for template '{template_name}'"
        ) from e


def list_templates() -> List[str]:
    """List all available strategy template names.

    Returns:
        Sorted list of template names

    Example:
        >>> templates = list_templates()
        >>> print(templates)
        ['bollinger_rsi', 'dual_ma_crossover', 'macd_signal', ...]
    """
    return sorted(STRATEGY_TEMPLATES.keys())


def get_template(template_name: str) -> str:
    """Get the raw template string without parameter substitution.

    Args:
        template_name: Name of the template

    Returns:
        Raw template string

    Raises:
        KeyError: If template_name is not found
    """
    if template_name not in STRATEGY_TEMPLATES:
        available = ", ".join(sorted(STRATEGY_TEMPLATES.keys()))
        raise KeyError(
            f"Template '{template_name}' not found. "
            f"Available templates: {available}"
        )
    return STRATEGY_TEMPLATES[template_name]


def get_default_params(template_name: str) -> Dict[str, str]:
    """Get default parameters for a template.

    Args:
        template_name: Name of the template

    Returns:
        Dictionary of default parameter values

    Raises:
        KeyError: If template_name is not found
    """
    if template_name not in STRATEGY_TEMPLATES:
        raise KeyError(f"Template '{template_name}' not found.")
    return dict(DEFAULT_PARAMS.get(template_name, {}))
