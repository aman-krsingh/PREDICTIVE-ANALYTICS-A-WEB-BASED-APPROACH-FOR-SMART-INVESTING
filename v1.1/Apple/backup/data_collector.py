from alpha_vantage.timeseries import TimeSeries

ticker='AAPL'
API_key = 'WSKF50ODKWY4WP1O'
ts = TimeSeries(key= API_key, output_format='pandas')
res = ts.get_daily(ticker, outputsize='full')

df=res[0]
df.to_csv('./AAPL_data.csv')

# size = len(df)
# year = 365 * 5
# df = df[size - year:]

# data = df.reset_index()['4. close']
# data.to_csv('AAPL_data.csv')
# #data.to_csv('./data/AAPL_5-yrs.csv')

