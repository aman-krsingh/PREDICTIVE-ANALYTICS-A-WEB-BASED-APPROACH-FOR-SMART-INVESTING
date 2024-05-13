import yfinance as yf

ticker='ADANIENT.NS'
res = yf.Ticker(f'{ticker}')

prd= 12*25
df =yf.download(f'{ticker}', period=f'{prd}mo')
df.to_csv(f'./data/{ticker}.csv')

# df = res.history(period="1mo")
# df.to_csv('./data/apl.csv')