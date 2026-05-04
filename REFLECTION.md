# Reflection

## What Worked Well

The separation between FinBERT and the LLM agent brain turned out to be the strongest
design decision in this project. By assigning sentiment classification entirely to
FinBERT and reserving reasoning and response generation for the LLM, each component
had a clearly defined role with no overlap. This made the system easy to debug: if
the sentiment scores looked wrong, the issue was in FinBERT or the headlines; if the
final recommendation was off, the issue was in the LLM prompt or tool output formatting.

LangChain's tool-calling interface worked smoothly once the tools were properly defined
with descriptive docstrings. The agent consistently called both tools before generating
a response, which was the intended behavior. Setting verbose=True made every reasoning
step visible, which was valuable both for development and for demonstrating the agent's
internal process.

The fallback mechanism for stock price data also proved essential. Stooq occasionally
returns empty responses depending on the time of day and server load. Having automatic
fallback data meant the agent could always complete its analysis without crashing.

## What Did Not Work

The original plan used the free `meta-llama/llama-3.1-8b-instruct:free` model on
OpenRouter. During testing, this model occasionally failed to follow the structured
output format defined in the system prompt and sometimes skipped tool calls entirely.
Switching to `openai/gpt-4o-mini` on OpenRouter resolved both issues immediately. The
model reliably calls both tools and follows the output format every time.

Stooq's stock data API was unreliable during development. It returned empty dataframes
for valid tickers on multiple occasions, likely due to rate limiting or regional access
restrictions from the Colab environment. This is why the fallback mechanism was added
early in development rather than as an afterthought.

## Biggest Technical Challenge

The biggest challenge was getting the LangChain agent to reliably call both tools in
sequence before generating a response. Early versions of the system prompt were too
vague, and the LLM would sometimes answer the user's question directly without calling
any tools. The solution was twofold: first, making the system prompt explicitly state
"always call both tools before responding," and second, writing very specific tool
docstrings that clearly describe when and why each tool should be used. The combination
of a directive system prompt and descriptive tool metadata solved the reliability issue.

## Path Choice

This project follows Option A: Single AI Agent, which is the same path chosen in the
Midterm blueprint. A single agent was the right fit because the problem -- analyzing one
stock at a time using two data sources -- does not require multiple agents with distinct
roles. One agent with two tools handles the workflow cleanly. Adding a second agent would
have introduced unnecessary complexity without improving the user experience.

## What I Would Build Next

With more time, I would add three capabilities. First, a live news API integration
(such as NewsAPI or Finnhub) to replace the static sample headlines with real-time
financial news, making the sentiment analysis reflect current market conditions. Second,
support for a broader set of tickers beyond AAPL and TSLA, including automatic ticker
detection from natural language queries. Third, a simple web interface using Streamlit
or Gradio so the agent could be used by non-technical users without needing a notebook
environment.
