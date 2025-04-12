import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import io
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log/forecaster.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TimeForecaster")


class TimeSeriesForecaster:
    def __init__(self):
        self.model = None
        self.is_trained = False
        logger.info("TimeSeriesForecaster initialized")

    def prepare_data(self, log_df):
        """Chuẩn bị dữ liệu từ log để huấn luyện Prophet"""
        if log_df is None or log_df.empty:
            logger.warning("Empty DataFrame provided for forecasting")
            return None

        try:
            # Đảm bảo timestamp là datetime
            if 'timestamp' in log_df.columns:
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            else:
                logger.error("DataFrame missing 'timestamp' column")
                return None

            # Đếm số lượng lỗi theo phút
            log_df['count'] = 1

            # Lọc các log cấp độ error hoặc critical
            error_logs = log_df[log_df['level'].isin(['error', 'critical'])]

            if error_logs.empty:
                logger.warning("No error or critical logs found")
                # Tạo bộ dữ liệu giả với số lượng bằng 0
                now = datetime.now()
                dates = pd.date_range(now - timedelta(hours=1), now, freq='1min')
                return pd.DataFrame({
                    'ds': dates,
                    'y': np.zeros(len(dates))
                })

            # Nhóm theo phút
            grouped = error_logs.groupby(pd.Grouper(key='timestamp', freq='1min')).count()[['count']].reset_index()

            # Đổi tên cột để phù hợp với Prophet
            prophet_df = grouped.rename(columns={"timestamp": "ds", "count": "y"})

            # Điền các khoảng thời gian không có lỗi bằng giá trị 0
            min_time = prophet_df['ds'].min()
            max_time = prophet_df['ds'].max()

            # Tạo index đầy đủ theo phút
            full_timerange = pd.date_range(min_time, max_time, freq='1min')

            # Reindex và điền giá trị khuyết là 0
            full_df = pd.DataFrame({'ds': full_timerange})
            prophet_df = pd.merge(full_df, prophet_df, on='ds', how='left').fillna(0)

            logger.info(f"Prepared time series data with {len(prophet_df)} intervals")
            return prophet_df

        except Exception as e:
            logger.error(f"Error preparing forecast data: {e}")
            return None

    def train(self, log_df=None, ts_data=None):
        """Huấn luyện mô hình dự đoán với dữ liệu từ log hoặc dữ liệu chuẩn bị sẵn"""
        try:
            # Nếu được cung cấp log, chuẩn bị dữ liệu trước
            if ts_data is None and log_df is not None:
                ts_data = self.prepare_data(log_df)

            if ts_data is None or ts_data.empty:
                logger.warning("No data available for training")
                return False

            # Khởi tạo và huấn luyện mô hình
            self.model = Prophet(
                seasonality_mode='multiplicative',
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )

            self.model.fit(ts_data)
            self.is_trained = True
            logger.info("Model trained successfully")
            return True

        except Exception as e:
            logger.error(f"Error training forecasting model: {e}")
            self.is_trained = False
            return False

    def forecast(self, df=None, periods=30):
        """Dự đoán xu hướng lỗi trong tương lai"""
        # Nếu được cung cấp log, huấn luyện mô hình trước
        if df is not None:
            self.train(log_df=df)

        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, cannot forecast")
            return {
                "success": False,
                "message": "Model not trained"
            }

        try:
            # Tạo DataFrame cho dự đoán tương lai
            future = self.model.make_future_dataframe(periods=periods, freq='min')
            forecast = self.model.predict(future)

            # Lấy dữ liệu các dự đoán
            forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

            # Tính số lượng lỗi dự kiến
            total_errors = forecast_results['yhat'].sum()
            max_errors_minute = forecast_results['yhat'].max()

            # Tạo biểu đồ
            fig = self.create_forecast_chart(forecast)

            # Convert chart to base64 string
            chart_data = self._fig_to_base64(fig)

            logger.info(f"Generated forecast for next {periods} minutes")

            return {
                "success": True,
                "forecast_data": forecast_results.to_dict(orient='records'),
                "summary": {
                    "total_predicted_errors": round(total_errors, 1),
                    "max_errors_per_minute": round(max_errors_minute, 1),
                    "forecast_period_minutes": periods,
                },
                "chart": chart_data
            }

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {
                "success": False,
                "message": f"Error generating forecast: {str(e)}"
            }

    def create_forecast_chart(self, forecast_df):
        """Tạo biểu đồ dự đoán"""
        fig = self.model.plot(forecast_df)
        plt.title('Error Forecast')
        plt.xlabel('Time')
        plt.ylabel('Number of Errors')
        plt.tight_layout()
        return fig

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str


if __name__ == '__main__':
    # Test the forecaster with sample data
    # Create a sample dataset
    now = datetime.now()
    dates = pd.date_range(now - timedelta(hours=2), now, freq='1min')

    # Generate some synthetic error counts with a pattern
    y_values = np.zeros(len(dates))
    # Add some random errors
    for i in range(len(dates)):
        # More errors every 15 minutes
        if i % 15 == 0:
            y_values[i] = np.random.poisson(5)
        else:
            # Random baseline errors
            y_values[i] = np.random.poisson(0.8)

    # Create the dataframe
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'level': ['error' if y > 0 else 'info' for y in y_values],
        'message': ['Sample error message' if y > 0 else 'Sample info message' for y in y_values]
    })

    # Test the forecaster
    forecaster = TimeSeriesForecaster()
    forecaster.train(log_df=sample_df)
    future_prediction = forecaster.forecast(periods=30)

    print("Forecast Summary:")
    print(f"Total predicted errors: {future_prediction['summary']['total_predicted_errors']}")
    print(f"Max errors per minute: {future_prediction['summary']['max_errors_per_minute']}")