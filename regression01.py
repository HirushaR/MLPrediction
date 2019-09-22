import pandas as pd
import math
import quandl
import numpy as np
from sklearn import preprocessing,svm,model_selection
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
# print(df.head())
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)
forecast_Out = int(math.ceil(0.01*len(df)))
#print(forecast_Out)

df['label'] = df[forecast_col].shift(-forecast_Out)
df.dropna(inplace=True)
#print(df.head())
#print(df.tail())

X = np.array(df.drop(['label'],1))
Y = np.array(df['label'])
X = preprocessing.scale(X)
Y = np.array(df['label'])

#print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
clf = LinearRegression(n_jobs=10)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

print(accuracy)

