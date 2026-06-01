"""
Streamlit UI for AI Strategy Generator (P3).

Professional dashboard for generating, backtesting, and optimizing trading strategies.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import pandas as pd

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from strategy_templates import list_templates, render_template
from backtester import run_backtest, BacktestResult
from optimizer import parameter_optimizer
from data_fetcher import fetch_stock_daily, generate_sample_data

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AI Strategy Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom dark theme CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #30363d;
    }
    
    .metric-card.positive {
        border-left-color: #3fb950;
    }
    
    .metric-card.negative {
        border-left-color: #f85149;
    }
    
    /* Template cards */
    .template-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #30363d;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .template-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    
    /* Code block styling */
    .code-block {
        background-color: #161b22 !important;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Text colors */
    .positive-text {
        color: #3fb950;
    }
    
    .negative-text {
        color: #f85149;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #161b22;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_code" not in st.session_state:
        st.session_state.current_code = None
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = None


def format_percent(value: float) -> str:
    """Format a decimal value as percentage string."""
    return f"{value * 100:.2f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimals."""
    return f"{value:.{decimals}f}"


def render_metric_card(label: str, value: str, is_positive: Optional[bool] = None):
    """Render a styled metric card."""
    if is_positive is None:
        css_class = "metric-card"
    elif is_positive:
        css_class = "metric-card positive"
    else:
        css_class = "metric-card negative"
    
    color_class = "positive-text" if is_positive else ("negative-text" if is_positive is False else "")
    color_style = f'class="{color_class}"' if color_class else ""
    
    st.markdown(f"""
    <div class="{css_class}">
        <div style="color: #8b949e; font-size: 14px; margin-bottom: 4px;">{label}</div>
        <div {color_style} style="font-size: 24px; font-weight: bold;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_backtest_results(result: BacktestResult):
    """Render backtest results in metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        is_pos = result.total_return >= 0
        render_metric_card(
            "Total Return",
            format_percent(result.total_return),
            is_pos
        )
    
    with col2:
        is_pos = result.sharpe_ratio >= 1.0
        render_metric_card(
            "Sharpe Ratio",
            format_number(result.sharpe_ratio),
            is_pos
        )
    
    with col3:
        is_pos = result.max_drawdown < 0.2
        render_metric_card(
            "Max Drawdown",
            format_percent(result.max_drawdown),
            is_pos
        )
    
    with col4:
        is_pos = result.win_rate >= 0.5
        render_metric_card(
            "Win Rate",
            format_percent(result.win_rate),
            is_pos
        )
    
    # Secondary metrics
    st.markdown("---")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        render_metric_card("Annual Return", format_percent(result.annual_return))
    
    with col6:
        render_metric_card("Total Trades", str(result.total_trades))
    
    with col7:
        render_metric_card("Sharpe Ratio (Annual)", format_number(result.sharpe_ratio))


def get_mock_strategy_code(strategy_name: str = "GeneratedStrategy") -> str:
    """Generate mock strategy code for demo purposes."""
    return f'''\
import backtrader as bt


class {strategy_name}(bt.Strategy):
    """AI-generated trading strategy."""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
        ("stake_pct", 0.95),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = int((self.broker.getcash() * self.p.stake_pct) / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()
'''


def mock_generate_strategy(user_input: str) -> dict:
    """Mock strategy generation for demo without OpenAI API."""
    return {
        "spec": {
            "name": "Generated Strategy",
            "description": "AI-generated from: " + user_input[:50],
            "strategy_type": "trend_following",
            "entry_rules": ["Fast MA crosses above slow MA"],
            "exit_rules": ["Fast MA crosses below slow MA"],
            "risk_management": {"stop_loss": 0.05, "take_profit": 0.10},
            "params": {"fast_period": 10, "slow_period": 30},
        },
        "code": get_mock_strategy_code(),
        "is_valid": True,
        "validation_msg": "Valid Backtrader strategy",
        "result": None,
    }


def run_optimization(data: pd.DataFrame, strategy_code: str) -> dict:
    """Run parameter optimization on strategy."""
    import random
    
    # Mock optimization results for demo
    results = []
    best_score = -999
    best_params = {}
    
    # Generate a few parameter combinations
    param_grid = {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50],
    }
    
    for fast in param_grid["fast_period"]:
        for slow in param_grid["slow_period"]:
            if fast >= slow:
                continue
            score = random.uniform(-0.5, 2.0)
            results.append({
                "params": {"fast_period": fast, "slow_period": slow},
                "score": score,
                "metrics": {
                    "total_return": random.uniform(-0.2, 0.4),
                    "sharpe_ratio": score,
                    "max_drawdown": random.uniform(0.05, 0.3),
                }
            })
            if score > best_score:
                best_score = score
                best_params = {"fast_period": fast, "slow_period": slow}
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": sorted(results, key=lambda x: x["score"], reverse=True),
        "method": "grid_search",
    }


# =============================================================================
# Tab 1: Strategy Generator
# =============================================================================

def render_strategy_generator_tab():
    """Render the main strategy generator tab."""
    st.markdown("## 📈 Strategy Generator")
    st.markdown("Describe your trading strategy in natural language, and AI will generate Backtrader code.")
    
    # Main input area
    user_input = st.text_area(
        "**Strategy Description**",
        placeholder="e.g., Buy when the 10-day moving average crosses above the 30-day MA, sell on reverse cross...",
        height=150,
        help="Describe your strategy in plain English. Include entry/exit conditions, indicators, and risk rules."
    )
    
    # Generate button
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_clicked = st.button("🎯 Generate Strategy", type="primary", use_container_width=True)
    
    if generate_clicked and user_input:
        with st.spinner("🤖 Generating strategy..."):
            try:
                # Try real API first, fall back to mock
                if st.session_state.get("use_mock", True):
                    result = mock_generate_strategy(user_input)
                else:
                    from strategy_agent import StrategyAgent
                    agent = StrategyAgent()
                    data = _get_or_create_data()
                    result = agent.run_pipeline(user_input, data)
                
                st.session_state.current_code = result["code"]
                
                # Add to history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "description": user_input[:100],
                    "code": result["code"],
                    "result": result.get("result"),
                })
                
                st.success("✅ Strategy generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
    
    # Display generated code
    if st.session_state.current_code:
        st.markdown("---")
        st.markdown("### 📄 Generated Code")
        
        with st.expander("View Strategy Code", expanded=True):
            st.code(st.session_state.current_code, language="python")
        
        # Action buttons
        col_run, col_opt = st.columns(2)
        
        with col_run:
            run_backtest_clicked = st.button("▶️ Run Backtest", type="primary", use_container_width=True)
        
        with col_opt:
            optimize_clicked = st.button("⚙️ Optimize Parameters", use_container_width=True)
        
        if run_backtest_clicked:
            _run_backtest_action()
        
        if optimize_clicked:
            _run_optimize_action()


def _get_or_create_data() -> pd.DataFrame:
    """Get or create sample data for backtesting."""
    try:
        symbol = st.session_state.get("symbol", "AAPL")
        days = st.session_state.get("days", 365)
        return fetch_stock_daily(symbol, days)
    except Exception:
        return generate_sample_data()


def _run_backtest_action():
    """Run backtest on current strategy code."""
    with st.spinner("📊 Running backtest..."):
        try:
            data = _get_or_create_data()
            result = run_backtest(
                data=data,
                strategy_code=st.session_state.current_code,
                initial_cash=st.session_state.get("initial_cash", 100000),
            )
            st.session_state.current_result = result
            
            st.markdown("---")
            st.markdown("### 📊 Backtest Results")
            render_backtest_results(result)
            
            # Trade log
            if result.trade_log and len(result.trade_log) > 0:
                with st.expander("View Trade Log"):
                    trades_df = pd.DataFrame(result.trade_log)
                    st.dataframe(trades_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")


def _run_optimize_action():
    """Run parameter optimization on current strategy."""
    with st.spinner("⚙️ Optimizing parameters..."):
        try:
            data = _get_or_create_data()
            opt_result = run_optimization(data, st.session_state.current_code)
            
            st.markdown("---")
            st.markdown("### ⚙️ Optimization Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Score", f"{opt_result['best_score']:.4f}")
            with col2:
                st.metric("Method", opt_result['method'])
            
            st.markdown("**Best Parameters:**")
            st.json(opt_result["best_params"])
            
            # Show top results
            if opt_result["all_results"]:
                st.markdown("**Top 5 Configurations:**")
                top_results = opt_result["all_results"][:5]
                for i, r in enumerate(top_results, 1):
                    with st.container():
                        st.markdown(f"#{i}: Score={r['score']:.4f}, Params={r['params']}")
            
        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")


# =============================================================================
# Tab 2: Template Library
# =============================================================================

def render_template_library_tab():
    """Render the template library tab."""
    st.markdown("## 📚 Template Library")
    st.markdown("Browse pre-built strategy templates and customize them for your needs.")
    
    templates = list_templates()
    
    # Template parameter presets
    template_params = {
        "ma_crossover": {"fast_period": 10, "slow_period": 30, "stake_pct": 0.95},
        "rsi_extreme": {"rsi_period": 14, "oversold": 30, "overbought": 70, "stake_pct": 0.95},
        "macd_signal": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "stake_pct": 0.95},
        "bollinger_bounce": {"period": 20, "devfactor": 2.0, "stake_pct": 0.95},
        "momentum_breakout": {"lookback": 20, "stake_pct": 0.95},
        "dual_thrust": {"lookback": 4, "k1": 0.5, "k2": 0.5, "stake_pct": 0.95},
    }
    
    # Create grid of template cards
    cols = st.columns(2)
    
    for idx, template in enumerate(templates):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"""
                <div class="template-card">
                    <h4 style="margin: 0 0 8px 0; color: #58a6ff;">{template['name'].replace('_', ' ').title()}</h4>
                    <p style="color: #8b949e; font-size: 14px; margin: 0;">{template['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_use, col_view = st.columns(2)
                with col_use:
                    use_clicked = st.button(f"Use Template", key=f"use_{template['name']}", use_container_width=True)
                with col_view:
                    view_clicked = st.button(f"View Code", key=f"view_{template['name']}", use_container_width=True)
                
                if use_clicked:
                    st.session_state.selected_template = template["name"]
                    st.session_state.template_params = template_params.get(template["name"], {})
                    st.session_state.current_code = render_template(
                        template["name"],
                        template_params.get(template["name"])
                    )
                    st.success(f"✅ Loaded template: {template['name']}")
                
                if view_clicked:
                    code = render_template(template["name"], template_params.get(template["name"]))
                    with st.expander(f"Code for {template['name']}", expanded=False):
                        st.code(code, language="python")
    
    # Show selected template preview
    if st.session_state.get("current_code"):
        st.markdown("---")
        st.markdown("### 📄 Current Template Code")
        st.code(st.session_state.current_code, language="python")


# =============================================================================
# Tab 3: History
# =============================================================================

def render_history_tab():
    """Render the strategy history tab."""
    st.markdown("## 📜 Strategy History")
    st.markdown("View and re-run previously generated strategies from this session.")
    
    if not st.session_state.history:
        st.info("No strategies generated yet. Go to the Strategy Generator tab to create one!")
        return
    
    # Display history in reverse chronological order
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #161b22; padding: 12px; border-radius: 8px; margin: 8px 0;">
                    <div style="color: #8b949e; font-size: 12px; margin-bottom: 4px;">
                        {item['timestamp']}
                    </div>
                    <div style="color: #c9d1d9; font-size: 14px;">
                        {item['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                rerun_clicked = st.button("↻ Re-run", key=f"rerun_{idx}", use_container_width=True)
            
            if rerun_clicked:
                st.session_state.current_code = item["code"]
                st.session_state.selected_history = idx
                st.success("✅ Strategy loaded! Go to Strategy Generator to run it.")
            
            # Show quick metrics if result exists
            if item.get("result"):
                result = item["result"]
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Return", format_percent(result.total_return))
                with metric_cols[1]:
                    st.metric("Sharpe", format_number(result.sharpe_ratio))
                with metric_cols[2]:
                    st.metric("Drawdown", format_percent(result.max_drawdown))
                with metric_cols[3]:
                    st.metric("Trades", result.total_trades)
            
            st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared!")


# =============================================================================
# Sidebar Configuration
# =============================================================================

def render_sidebar():
    """Render the sidebar configuration panel."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Data parameters
        st.markdown("### 📊 Data Settings")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol")
        st.session_state.symbol = symbol
        
        days = st.slider("History Days", min_value=30, max_value=730, value=365, step=30)
        st.session_state.days = days
        
        initial_cash = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
        st.session_state.initial_cash = initial_cash
        
        st.markdown("---")
        
        # API settings
        st.markdown("### 🔑 API Settings")
        
        use_mock = st.checkbox("Use Demo Mode (Mock API)", value=True)
        st.session_state.use_mock = use_mock
        
        if not use_mock:
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                st.session_state.openai_api_key = api_key
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **AI Strategy Generator v1.0**
        
        Generate Backtrader trading strategies from natural language descriptions.
        
        Built with:
        - Streamlit
        - OpenAI GPT-4
        - Backtrader
        - Pandas
        """)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Main title
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #c9d1d9; margin: 0;">📈 AI Strategy Generator</h1>
        <p style="color: #8b949e; font-size: 16px;">Transform natural language into trading strategies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs([
        "🎯 Strategy Generator",
        "📚 Template Library",
        "📜 History"
    ])
    
    with tab1:
        render_strategy_generator_tab()
    
    with tab2:
        render_template_library_tab()
    
    with tab3:
        render_history_tab()


if __name__ == "__main__":
    main()