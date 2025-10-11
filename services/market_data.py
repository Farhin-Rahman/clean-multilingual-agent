# services/market_data.py - Real market data with caching and Stooq fallback

import os
import time
import logging
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Install cache at module import time
try:
    import requests_cache
    requests_cache.install_cache("market_cache", backend="sqlite", expire_after=6*3600)
    logger.info("Market data cache installed")
except Exception as e:
    logger.warning(f"Failed to install market cache: {e}")

# Cache configuration
CACHE_TTL_HOURS = 6

def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Get historical price data with caching and Stooq fallback.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        period: Time period ('1y', '6mo', '3mo', etc.)
        interval: Data interval ('1d', '1h', etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Try yfinance first with caching
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval, auto_adjust=True)
        
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            logger.info(f"Got {len(hist)} rows for {symbol} from yfinance")
            return hist
        
    except Exception as e:
        logger.warning(f"yfinance failed for {symbol}: {e}")
    
    # Fallback to Stooq
    try:
        from pandas_datareader import data as pdr
        
        # Convert symbol for Stooq (add .US suffix if needed)
        stooq_symbol = symbol
        if not stooq_symbol.startswith("^") and "." not in stooq_symbol:
            stooq_symbol = f"{stooq_symbol}.US"
        
        # Map period to days for Stooq
        period_days = {
            "1y": 365,
            "6mo": 180,
            "3mo": 90,
            "1mo": 30,
            "5d": 5
        }.get(period, 365)
        
        # Get data from Stooq
        df = pdr.DataReader(stooq_symbol, "stooq")
        
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Sort by date and filter to requested period
            df = df.sort_index()
            cutoff_date = datetime.now() - timedelta(days=period_days)
            df = df.loc[df.index >= cutoff_date]
            
            logger.info(f"Got {len(df)} rows for {symbol} from Stooq")
            return df
        
    except Exception as e:
        logger.warning(f"Stooq fallback failed for {symbol}: {e}")
    
    # Last resort: return empty DataFrame
    logger.error(f"All data sources failed for {symbol}")
    return pd.DataFrame()

def get_last_price(symbol: str) -> float:
    """
    Get the most recent price for a symbol.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Last price as float, or 0.0 if unavailable
    """
    try:
        hist = get_history(symbol, period="5d", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to get last price for {symbol}: {e}")
    
    return 0.0

def get_quote(symbol: str) -> dict:
    """
    Get current quote information for a symbol.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dict with quote information
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Try fast_info first for speed
        try:
            fast_info = ticker.fast_info
            quote = {
                "symbol": symbol,
                "price": fast_info.get("lastPrice", 0.0),
                "change": fast_info.get("regularMarketChange", 0.0),
                "change_percent": fast_info.get("regularMarketChangePercent", 0.0),
                "volume": fast_info.get("volume", 0),
                "market_cap": fast_info.get("marketCap", 0),
                "pe_ratio": 0.0,  # Not in fast_info
                "dividend_yield": 0.0,  # Not in fast_info
                "beta": 1.0,  # Not in fast_info
                "sector": "Unknown",  # Not in fast_info
                "industry": "Unknown",  # Not in fast_info
                "name": symbol
            }
        except Exception:
            # Fallback to regular info
            info = ticker.info
            quote = {
                "symbol": symbol,
                "price": info.get("currentPrice", info.get("regularMarketPrice", 0.0)),
                "change": info.get("regularMarketChange", 0.0),
                "change_percent": info.get("regularMarketChangePercent", 0.0),
                "volume": info.get("volume", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0.0),
                "dividend_yield": info.get("dividendYield", 0.0),
                "beta": info.get("beta", 1.0),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "name": info.get("longName", symbol)
            }
        
        return quote
        
    except Exception as e:
        logger.warning(f"Failed to get quote for {symbol}: {e}")
        return {
            "symbol": symbol,
            "price": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "volume": 0,
            "market_cap": 0,
            "pe_ratio": 0.0,
            "dividend_yield": 0.0,
            "beta": 1.0,
            "sector": "Unknown",
            "industry": "Unknown",
            "name": symbol
        }

def get_multiple_quotes(symbols: list) -> dict:
    """
    Get quotes for multiple symbols efficiently.
    
    Args:
        symbols: List of stock symbols
    
    Returns:
        Dict mapping symbols to quote information
    """
    quotes = {}
    
    try:
        import yfinance as yf
        
        # Use yfinance's multi-ticker download for efficiency
        tickers = yf.Tickers(" ".join(symbols))
        
        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                
                quotes[symbol] = {
                    "symbol": symbol,
                    "price": info.get("currentPrice", info.get("regularMarketPrice", 0.0)),
                    "change": info.get("regularMarketChange", 0.0),
                    "change_percent": info.get("regularMarketChangePercent", 0.0),
                    "volume": info.get("volume", 0),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0.0),
                    "dividend_yield": info.get("dividendYield", 0.0),
                    "beta": info.get("beta", 1.0),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "name": info.get("longName", symbol)
                }
                
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                quotes[symbol] = get_quote(symbol)  # Fallback to individual lookup
    
    except Exception as e:
        logger.warning(f"Multi-quote lookup failed: {e}")
        # Fallback to individual lookups
        for symbol in symbols:
            quotes[symbol] = get_quote(symbol)
    
    return quotes

def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol exists and is tradeable.
    
    Args:
        symbol: Stock symbol to validate
    
    Returns:
        True if symbol is valid, False otherwise
    """
    try:
        quote = get_quote(symbol)
        return quote["price"] > 0
    except Exception:
        return False

def get_market_status() -> dict:
    """
    Get overall market status and indices.
    
    Returns:
        Dict with market status information
    """
    try:
        indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
        status = {}
        
        for index in indices:
            try:
                quote = get_quote(index)
                status[index] = {
                    "price": quote["price"],
                    "change": quote["change"],
                    "change_percent": quote["change_percent"]
                }
            except Exception as e:
                logger.warning(f"Failed to get status for {index}: {e}")
                status[index] = {"price": 0, "change": 0, "change_percent": 0}
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get market status: {e}")
        return {}

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators for a price DataFrame.
    
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
        logger.warning(f"Failed to calculate technical indicators: {e}")
    
    return result
