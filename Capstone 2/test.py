import requests

API_KEY = "UUNFUFCQ239ZTTRM"
response = requests.get("https://www.alphavantage.co/query", params={
    "function": "OVERVIEW",
    "symbol": "AAPL",
    "apikey": API_KEY
})
print(response.json())


