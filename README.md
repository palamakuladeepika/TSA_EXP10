# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date:

### AIM:

To implement SARIMA model using python.

### ALGORITHM:

1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions

### PROGRAM:
```
Developed By : Palamakula Deepika
Reg. No. : 212221240035
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Temperature.csv')
data['date'] = pd.to_datetime(data['date'])

plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['temp'])

plot_acf(data['temp'])
plt.show()
plot_pacf(data['temp'])
plt.show()

sarima_model = SARIMAX(data['temp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['temp'][:train_size], data['temp'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT:
![image](https://github.com/Pavan-Gv/TSA_EXP10/assets/94827772/b555ed04-90b2-4b08-9f9c-37e77034c389)

![image](https://github.com/Pavan-Gv/TSA_EXP10/assets/94827772/d429374a-6423-4f6a-8806-02c1dcd0eff2)

![image](https://github.com/Pavan-Gv/TSA_EXP10/assets/94827772/b0bce165-7a67-43a2-8b89-39d78f1e820d)

![image](https://github.com/Pavan-Gv/TSA_EXP10/assets/94827772/5c26e7ed-37f0-4789-84df-5cec9658e1d1)


### RESULT:
Thus the program run successfully based on the SARIMA model.
