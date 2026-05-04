"""
Tool functions used by the LangChain agent.

This module provides two capabilities that the agent can invoke:

1. Stock price retrieval -- fetches recent OHLCV data from Stooq via
   pandas-datareader and returns a concise summary of latest close and
   five-day price change.

2. News sentiment analysis -- passes pre-collected financial headlines
   through the FinBERT deep learning model and returns a structured
   sentiment summary.

Both functions include robust fallback logic so the agent demonstration
never crashes due to external data-source failures.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as pdr

from agent.sentiment import analyze_sentiment

# ---------------------------------------------------------------------------
# Sample headlines -- used when a live news API is not available.
# In production these would be fetched from a financial news endpoint.
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load_sample_headlines() -> dict:
    """
    Load sample headlines from the JSON file in the data directory.

    Returns:
        A dict mapping ticker symbols to lists of headline strings.
    """
    path = os.path.join(_DATA_DIR, "sample_headlines.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SAMPLE_HEADLINES: dict = _load_sample_headlines()


# ---------------------------------------------------------------------------
# Stock price tool
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str, days: int = 10) -> dict:
    """
    Fetch recent stock price data for a given ticker using Stooq.

    Retrieves up to the last 5 available trading days and computes a
    percentage price change between the oldest and newest close in that
    window. If the Stooq request fails for any reason, fallback sample
    data is returned so the agent can continue operating.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA).
        days:   Number of calendar days to look back for trading data.

    Returns:
        A dict with keys: ticker, latest_close, price_change_5d, status.
        status is 'success' when live data was used, or 'fallback' otherwise.
    """
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)

        # Stooq requires a .US suffix for US equity tickers
        stooq_ticker = f"{ticker.upper()}.US"
        df = pdr.get_data_stooq(stooq_ticker, start=start_date, end=end_date)

        if df.empty:
            raise ValueError("No data returned from Stooq.")

        # Sort ascending and keep only the most recent 5 trading days
        df = df.sort_index(ascending=True).tail(5)

        # Percentage change from oldest to newest close in the window
        price_change = (
            (df["Close"].iloc[-1] - df["Close"].iloc[0])
            / df["Close"].iloc[0]
        ) * 100

        return {
            "ticker": ticker.upper(),
            "latest_close": round(df["Close"].iloc[-1], 2),
            "price_change_5d": round(price_change, 2),
            "status": "success",
        }

    except Exception as e:
        # Fallback data keeps the agent functional during demos
        print(f"Stooq fetch failed: {e}. Using fallback sample data.")
        return {
            "ticker": ticker.upper(),
            "latest_close": 189.45,
            "price_change_5d": 1.82,
            "status": "fallback",
        }


# ---------------------------------------------------------------------------
# News sentiment tool
# ---------------------------------------------------------------------------

def analyze_sentiment_for_ticker(ticker: str) -> dict:
    """
    Run FinBERT sentiment classification on headlines for a given ticker.

    Headlines are sourced from the pre-collected sample set. If the
    requested ticker is not in the sample set, AAPL headlines are used
    as a default so the agent always returns a result.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA).

    Returns:
        A dict with keys: ticker, dominant_sentiment, average_confidence,
        breakdown, headlines_analyzed.
    """
    ticker = ticker.upper()

    # Default to AAPL headlines if the ticker is not in the sample set
    headlines = SAMPLE_HEADLINES.get(ticker, SAMPLE_HEADLINES["AAPL"])

    # Delegate classification to the FinBERT sentiment module
    result = analyze_sentiment(headlines)

    # Attach the ticker to the result for downstream consumers
    result["ticker"] = ticker
    return result
