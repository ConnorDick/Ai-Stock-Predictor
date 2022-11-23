from sklearn.linear_model import SGDClassifier
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from matplotlib import pyplot as plt
import numpy as np

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
    testData = []
    asset = pd.DataFrame(yf.download(ticker, start=Start, end=End)['Adj Close'])
    for index in asset.index:
        #print(asset['Adj Close'][index])
        testData.append(asset['Adj Close'][index])
    return asset, testData

TESLA, testData = closing_price('TSLA')                  # CALL THE FUNCTION
testData = np.array(testData).reshape(-1, 1)
#testData.reshape(-1, 1)
targetValues = []
for i in range(28):
    targetValues.append(i)
print(testData)

# print(TESLA)
plt.plot(TESLA)
plt.title('TESLA Performance')
plt.ylabel('Price ($)')
plt.xlabel('Date')
plt.show()

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
clf.fit(testData, targetValues)
for i in range(90):
    test = clf.predict([[]])
    print(test)
