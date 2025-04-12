import pandas as pd
from prophet import Prophet
from datetime import datetime

class TimeSeriesForecaster:
    def __init__(self):
        self.model = Prophet()

    def prepare_data(self, log_df):
        # Count number of errors per minute
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
        log_df['count'] = 1
        grouped = log_df[log_df['level'].isin(['error', 'critical'])].groupby(pd.Grouper(key='timestamp', freq='1min')).count().reset_index()
        grouped = grouped.rename(columns={"timestamp": "ds", "count": "y"})
        return grouped

    def train(self, log_df):
        ts_data = self.prepare_data(log_df)
        self.model.fit(ts_data)

    def forecast(self, periods=30):
        future = self.model.make_future_dataframe(periods=periods, freq='min')
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

if __name__ == '__main__':
    logs = pd.read_csv("realtime_logs.csv")
    forecaster = TimeSeriesForecaster()
    forecaster.train(logs)
    future_df = forecaster.forecast(30)
    print(future_df.tail())
