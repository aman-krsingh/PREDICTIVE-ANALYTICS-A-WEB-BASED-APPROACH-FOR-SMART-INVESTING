from alpha_vantage.timeseries import TimeSeries

#ticker='AAPL'
ticker_list = ['AAPL', 'META', 'GOOG']

API_key = 'WSKF50ODKWY4WP1O'

for ticker in ticker_list:
    ts = TimeSeries(key= API_key, output_format='pandas')
    res = ts.get_daily(ticker, outputsize='full')
    df=res[0]
    df.to_csv(f'./data/{ticker}2.csv')


