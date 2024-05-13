from alpha_vantage.timeseries import TimeSeries

ticker='AAPL'

API_key = 'WSKF50ODKWY4WP1O'
ts = TimeSeries(key= API_key, output_format='pandas')
res = ts.get_daily(ticker, outputsize='full')

df=res[0]
df.to_csv(f'./data/{ticker}.csv')


