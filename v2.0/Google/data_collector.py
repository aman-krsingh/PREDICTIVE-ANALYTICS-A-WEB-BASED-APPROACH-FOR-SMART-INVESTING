import yfinance as yf

ticker='GOOG'
res = yf.Ticker(f'{ticker}')

prd= 12*25
df =yf.download(f'{ticker}', period=f'{prd}mo')
df.to_csv(f'./data/{ticker}.csv')

