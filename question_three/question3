# import yfinance as yf
# import pandas as pd
# import matplotlib as plt
# from IPython.display import display
# from tabulate import tabulate
# import IPython
#
# teslaStockTicker = 'TSLA'
#
# teslaStockData = yf.Ticker(teslaStockTicker)
#
# teslaStockDF = teslaStockData.history(period='1d', start='2022-10-10', end='2022-11-20')
#
# column_headers = list(teslaStockDF.columns)
#
# teslaStockDF.to_csv('teslaData.csv')
# print(column_headers)
# print(teslaStockDF)
#
# plt.plot(teslaStockDF)
#
# teslaStockDF.plot(teslaStockDF["Date"], teslaStockDF["Close"])
# display(teslaStockDF)

from sklearn.linear_model import SGDClassifier
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from matplotlib import pyplot as plt

# calculating the start date
Start = date.today() - timedelta(40)
Start.strftime('%Y-%m-%d')

# calculating the end date
End = date.today()
# print(date.today())
# print(End)
End.strftime('%Y-%m-%d')

# func accepts ticker, return df of date & closing price


def closing_price(ticker):
    asset = pd.DataFrame(yf.download(ticker, start=Start, end=End)['Adj Close'])
    return asset


TESLA = closing_price('TSLA')                  # CALL THE FUNCTION

# print(TESLA)
plt.plot(TESLA)
plt.title('TESLA Performance')
plt.ylabel('Price ($)')
plt.xlabel('Date')
plt.show()

X = [[1], [20], [40]]
y = [0]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
SGDClassifier(max_iter=5)
print(clf.predict[3])

# def stochasticGradientDescent():
