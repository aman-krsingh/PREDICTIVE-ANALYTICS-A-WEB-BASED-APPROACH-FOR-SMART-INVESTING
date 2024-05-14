import yfinance as yf

ticker='AAPL'
prd= 12*25

df =yf.download(f'{ticker}', period=f'{prd}mo')
df.to_csv(f'./data/{ticker}.csv')


