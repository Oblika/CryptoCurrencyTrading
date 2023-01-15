#Libraries or Modules
#Pandas
#yfinance
#datetime
#pyplot


import  warnings
warnings.filterwarnings("ignore")

try:
    import yfinance
except:
    import yfinance

import  yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta

#We are going to download data for BITCOIN, ETHEREUM, NASDAQ 100.


df = yf.download(
    'BTC-USD',
    start = '2018-01-01',
    end = date.today(),
    progress=False,
)

df2 = yf.download(
    'ETH-USD',
    start = '2018-01-01',
    end = date.today(),
    progress=False,
)

df3 = yf.download(
    '^NDX',
    start = '2018-01-01',
    end = date.today(),
    progress=False,
)


df.plot(y='Close', title="BTC-USD")
df2.plot(y='Close', title='ETH-USD')
df3.plot(y='Close', title='Nasdaq 100')
plt.show()

#Create Technical Analysis Indicators

try:
    import pandas_ta as ta
except:
    import pandas_ta as ta

df['RSI(2)'] = ta.rsi(close=df['Close'],length=2)
df['RSI(2) of DF2'] = ta.rsi(close=df2['Close'],length=2)
df['RSI(2) of DF3'] = ta.rsi(close=df3['Close'],length=2)
df['RSI(7)'] = ta.rsi(close=df3['Close'],length=7)
df['RSI(7) of DF2'] = ta.rsi(close=df3['Close'],length=7)
df['RSI(7) of DF3'] = ta.rsi(close=df3['Close'],length=7)
df['Close / Moving Average(14)'] = df['Close'] / ta.sma(close=df['Close'],length=14)
df['Close / Moving Average(14) of DF2'] = df2['Close'] / ta.sma(close=df2['Close'],length=14)
df['Close / Moving Average(14) of DF3'] = df3['Close'] / ta.sma(close=df3['Close'],length=14)
df['Close / Moving Average(30)'] = df['Close'] / ta.sma(close=df['Close'],length=30)
df['Close / Moving Average(30) of DF2'] = df2['Close'] / ta.sma(close=df2['Close'],length=30)
df['Close / Moving Average(30) of DF3'] = df3['Close'] / ta.sma(close=df3['Close'],length=30)
df['WILLR(10)'] = ta.willr(high=df['High'],close=df['Close'],low=df['Low'],length=10)
df['WILLR(10) of DF2'] = ta.willr(high=df2['High'],close=df2['Close'],low=df2['Low'],length=10)
df['WILLR(10) of DF2'] = ta.willr(high=df3['High'],close=df3['Close'],low=df3['Low'],length=10)

df = df.dropna()

df.head()

#Plotting Indicators
df.plot(y='Close / Moving Average(14)')
df.plot(y='Close / Moving Average(14) of DF2')
df.plot(y='Close / Moving Average(14) of DF3')
plt.plot()


#Labeling our data
import numpy as np

df['LABEL'] = np.where (df['Open'].shift(-2)/df['Open'].shift(-1)).gt(1.0025)

df.head()


try:
    import wittgenstein as lw
except:
    import wittgenstein as lw

from sklearn.model_selection import train_test_split

x = df[df.columns[6:-1]]
y = df['LABEL']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

ripper_clf = lw.RIPPER(max_rules)
ripper_clf.fit(x_train, y_train,pos_class="1")
print(ripper_clf.out_model())


print("OOS Accuracy Score: ", ripper_clf.score(x_test, y_test))

predict_train = ripper_clf.predict(x_train)
predict_test = ripper_clf.predict(x_test)

df['Prediction'] = np.apprend(predict_train,predict_test)

df['Strategy Returns'] = np.where( df['Prediction'].eq(True),df['Open'].shift(-2)-df['Open'].shift(-1),0)

base_capital = 1000

df['Strategy Returns'] = np.where( df['Strategy Returns'].eq(0),0, base_capital*df['Strategy Returns']-base_capital).cumsum()

df.plot(y='Strategy Returns')
plt.plot()
