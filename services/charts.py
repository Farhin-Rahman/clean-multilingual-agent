# services/charts.py - Financial charts with Plotly

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for a price DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional technical indicators
    """
    if df.empty or "Close" not in df.columns:
        return df
    
    result = df.copy()
    
    try:
        # Simple Moving Averages
        result["SMA_20"] = result["Close"].rolling(window=20).mean()
        result["SMA_50"] = result["Close"].rolling(window=50).mean()
        result["SMA_200"] = result["Close"].rolling(window=200).mean()
        
        # RSI
        delta = result["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result["RSI"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result["BB_Middle"] = result["Close"].rolling(window=20).mean()
        bb_std = result["Close"].rolling(window=20).std()
        result["BB_Upper"] = result["BB_Middle"] + (bb_std * 2)
        result["BB_Lower"] = result["BB_Middle"] - (bb_std * 2)
        
        # MACD
        exp1 = result["Close"].ewm(span=12).mean()
        exp2 = result["Close"].ewm(span=26).mean()
        result["MACD"] = exp1 - exp2
        result["MACD_Signal"] = result["MACD"].ewm(span=9).mean()
        result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]
        
        # Volume indicators
        if "Volume" in result.columns:
            result["Volume_SMA"] = result["Volume"].rolling(window=20).mean()
            result["Volume_Ratio"] = result["Volume"] / result["Volume_SMA"]
        
        # Price change indicators
        result["Price_Change"] = result["Close"].pct_change()
        result["Volatility"] = result["Price_Change"].rolling(window=20).std() * np.sqrt(252)
        
        # Drawdown
        result["Peak"] = result["Close"].expanding().max()
        result["Drawdown"] = (result["Close"] - result["Peak"]) / result["Peak"]
        
    except Exception as e:
        logger.warning(f"Failed to compute technical indicators: {e}")
    
    return result

def candle_with_sma(symbol: str) -> Optional[Any]:
    """
    Create a candlestick chart with moving averages.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Plotly Figure or None if failed
    """
    try:
        from services.market_data import get_history
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get historical data
        hist = get_history(symbol, period="1y", interval="1d")
        if hist.empty:
            logger.warning(f"No data available for {symbol}")
            return None
        
        # Compute indicators
        hist = compute_indicators(hist)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol} Price & Moving Averages", "Volume"),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"] if "Open" in hist.columns else hist["Close"],
                high=hist["High"] if "High" in hist.columns else hist["Close"],
                low=hist["Low"] if "Low" in hist.columns else hist["Close"],
                close=hist["Close"],
                name="Price",
                increasing_line_color="#00ff00",
                decreasing_line_color="#ff0000"
            ),
            row=1, col=1
        )
        
        # Moving averages
        if "SMA_20" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["SMA_20"],
                    name="SMA 20",
                    line=dict(color="blue", width=1)
                ),
                row=1, col=1
            )
        
        if "SMA_50" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["SMA_50"],
                    name="SMA 50",
                    line=dict(color="orange", width=1)
                ),
                row=1, col=1
            )
        
        if "SMA_200" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["SMA_200"],
                    name="SMA 200",
                    line=dict(color="purple", width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        if "Volume" in hist.columns:
            fig.add_trace(
                go.Bar(
                    x=hist.index,
                    y=hist["Volume"],
                    name="Volume",
                    marker_color="lightblue",
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Price Chart with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        # Remove range slider
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        logger.info(f"Created candlestick chart for {symbol}")
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create candlestick chart for {symbol}: {e}")
        return None

def rsi_chart(symbol: str) -> Optional[Any]:
    """
    Create an RSI chart.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Plotly Figure or None if failed
    """
    try:
        from services.market_data import get_history
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get historical data
        hist = get_history(symbol, period="1y", interval="1d")
        if hist.empty:
            return None
        
        # Compute indicators
        hist = compute_indicators(hist)
        
        if "RSI" not in hist.columns:
            return None
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol} Price", "RSI (14)"),
            row_width=[0.7, 0.3]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                name="Price",
                line=dict(color="black", width=2)
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["RSI"],
                name="RSI",
                line=dict(color="purple", width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - RSI Analysis",
            xaxis_title="Date",
            height=500,
            showlegend=True,
            template="plotly_white"
        )
        
        # Set y-axis ranges
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        logger.info(f"Created RSI chart for {symbol}")
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create RSI chart for {symbol}: {e}")
        return None

def drawdown_chart(symbol: str) -> Optional[Any]:
    """
    Create a drawdown chart.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Plotly Figure or None if failed
    """
    try:
        from services.market_data import get_history
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get historical data
        hist = get_history(symbol, period="1y", interval="1d")
        if hist.empty:
            return None
        
        # Compute indicators
        hist = compute_indicators(hist)
        
        if "Drawdown" not in hist.columns:
            return None
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol} Price", "Drawdown"),
            row_width=[0.7, 0.3]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                name="Price",
                line=dict(color="black", width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Drawdown"] * 100,  # Convert to percentage
                name="Drawdown",
                line=dict(color="red", width=2),
                fill="tozeroy"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Drawdown Analysis",
            xaxis_title="Date",
            height=500,
            showlegend=True,
            template="plotly_white"
        )
        
        # Set y-axis ranges
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        logger.info(f"Created drawdown chart for {symbol}")
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create drawdown chart for {symbol}: {e}")
        return None

def macd_chart(symbol: str) -> Optional[Any]:
    """
    Create a MACD chart.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Plotly Figure or None if failed
    """
    try:
        from services.market_data import get_history
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get historical data
        hist = get_history(symbol, period="1y", interval="1d")
        if hist.empty:
            return None
        
        # Compute indicators
        hist = compute_indicators(hist)
        
        if "MACD" not in hist.columns:
            return None
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol} Price", "MACD"),
            row_width=[0.7, 0.3]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                name="Price",
                line=dict(color="black", width=2)
            ),
            row=1, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["MACD"],
                name="MACD",
                line=dict(color="blue", width=2)
            ),
            row=2, col=1
        )
        
        # MACD Signal
        if "MACD_Signal" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist["MACD_Signal"],
                    name="Signal",
                    line=dict(color="red", width=2)
                ),
                row=2, col=1
            )
        
        # MACD Histogram
        if "MACD_Histogram" in hist.columns:
            colors = ["green" if x >= 0 else "red" for x in hist["MACD_Histogram"]]
            fig.add_trace(
                go.Bar(
                    x=hist.index,
                    y=hist["MACD_Histogram"],
                    name="Histogram",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - MACD Analysis",
            xaxis_title="Date",
            height=500,
            showlegend=True,
            template="plotly_white"
        )
        
        # Set y-axis ranges
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        logger.info(f"Created MACD chart for {symbol}")
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create MACD chart for {symbol}: {e}")
        return None

def create_all_charts(symbol: str) -> Dict[str, Any]:
    """
    Create all available charts for a symbol.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dict with chart figures
    """
    charts = {}
    
    try:
        charts["candlestick"] = candle_with_sma(symbol)
        charts["rsi"] = rsi_chart(symbol)
        charts["drawdown"] = drawdown_chart(symbol)
        charts["macd"] = macd_chart(symbol)
        
        # Remove None values
        charts = {k: v for k, v in charts.items() if v is not None}
        
        logger.info(f"Created {len(charts)} charts for {symbol}")
        
    except Exception as e:
        logger.error(f"Failed to create charts for {symbol}: {e}")
    
    return charts
