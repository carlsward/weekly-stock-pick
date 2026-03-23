import yfinance as yf

symbol = "AAPL"
ticker = yf.Ticker(symbol)

try:
    print("Testing quote...")
    q = ticker.history(period="1d")
    print("Quote rows:", len(q))

    print("Testing news...")
    news = ticker.news
    print("News items:", len(news) if news else 0)

except Exception as e:
    print("ERROR:", e)
