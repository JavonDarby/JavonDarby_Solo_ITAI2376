"""
LangChain agent module for the AI Trading Assistant.

This module assembles the complete agent by connecting two tools -- stock
price retrieval and FinBERT news sentiment analysis -- with a large language
model reasoning brain hosted on OpenRouter. LangChain orchestrates the
reasoning loop, manages tool calls, and structures the agent's
decision-making process.

The LLM receives structured outputs from both tools, determines risk level,
and generates a plain-language explanation. It does not perform sentiment
classification -- that is FinBERT's role.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools import get_stock_data, analyze_sentiment_for_ticker


# ---------------------------------------------------------------------------
# System prompt -- defines the agent persona and output format
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a professional AI Trading Assistant designed to help
beginner investors make more informed decisions. You have access to two tools:
one that retrieves stock price data and one that analyzes news sentiment using
a deep learning model called FinBERT.

When a user asks about a stock, always call both tools before responding.
After gathering the data, provide your response in exactly this format:

Ticker:           [TICKER]
News Sentiment:   [Positive / Negative / Neutral] ([confidence]% confidence)
Price Trend:      [Up / Down / Flat] [X]% over the last 5 trading days
Risk Level:       [Low / Medium / High]
Suggestion:       [Buy / Hold / Sell]
Explanation:      [2 to 3 sentences in plain language explaining your reasoning,
                   referencing both the price data and the sentiment score.]

Always remind the user that this is an educational tool and not professional
financial advice."""


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------

@tool
def stock_price_tool(ticker: str) -> str:
    """
    Retrieves the last 5 trading days of stock price data for the given ticker.
    Use this tool when you need current price trend information for a stock.
    Input: a stock ticker symbol such as AAPL or TSLA.
    Returns: latest closing price and 5-day percentage price change.
    """
    data = get_stock_data(ticker)
    return (
        f"Ticker: {data['ticker']} | "
        f"Latest Close: ${data['latest_close']} | "
        f"5-Day Price Change: {data['price_change_5d']}%"
    )


@tool
def news_sentiment_tool(ticker: str) -> str:
    """
    Analyzes recent financial news headlines for the given ticker using FinBERT,
    a deep learning model fine-tuned on financial text.
    Use this tool when you need to understand market sentiment around a stock.
    Input: a stock ticker symbol such as AAPL or TSLA.
    Returns: dominant sentiment label, average confidence score, and breakdown.
    """
    data = analyze_sentiment_for_ticker(ticker)
    return (
        f"Ticker: {data['ticker']} | "
        f"Sentiment: {data['dominant_sentiment']} | "
        f"Confidence: {data['average_confidence']:.2%} | "
        f"Breakdown: {data['breakdown']}"
    )


# ---------------------------------------------------------------------------
# Agent assembly
# ---------------------------------------------------------------------------

def build_agent(api_key: str,
                base_url: str = "https://openrouter.ai/api/v1",
                model: str = "openai/gpt-4o-mini") -> AgentExecutor:
    """
    Build and return a LangChain AgentExecutor wired to both tools.

    Args:
        api_key:  OpenRouter API key.
        base_url: OpenRouter base URL.
        model:    Model identifier on OpenRouter.

    Returns:
        A ready-to-invoke AgentExecutor instance with verbose output enabled.
    """
    # Initialize the LLM via OpenRouter using LangChain's OpenAI-compatible class
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.3,
    )

    # Prompt template with a scratchpad for intermediate reasoning steps
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Register both tools with the agent
    tools = [stock_price_tool, news_sentiment_tool]
    agent = create_openai_tools_agent(llm, tools, prompt)

    # verbose=True surfaces the internal reasoning loop during execution
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def run_agent(question: str, api_key: str,
              base_url: str = "https://openrouter.ai/api/v1",
              model: str = "openai/gpt-4o-mini") -> str:
    """
    Convenience function: build the agent and invoke it with a single question.

    Args:
        question: The user's plain-language trading question.
        api_key:  OpenRouter API key.
        base_url: OpenRouter base URL.
        model:    Model identifier on OpenRouter.

    Returns:
        The agent's final response as a string.
    """
    executor = build_agent(api_key, base_url, model)
    response = executor.invoke({"input": question})
    return response["output"]
