import yfinance as yf

ticker='TCS.NS'

prd= 12*25
df =yf.download(f'{ticker}', period=f'{prd}mo')
df.to_csv(f'./data/{ticker}.csv')
