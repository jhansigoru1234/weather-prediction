from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv('temps2.csv', sep=';')
data.head(5)
data.describe()
data.shape
data = data[['Total Precipitation','Wind Speed','Wind Direction','Temperature']]
data.head()
data.isnull()
x = data[['Total Precipitation','Wind Speed','Wind Direction']]
x = np.array(x)
x = x.reshape(len(x),3)
x.shape
y = data[['Temperature']]
y = np.array(y)
y = y.reshape(len(y),1)
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size = 0.2,random_state=4)
x_train
x_test
y_train
y_test
rf = RandomForestRegressor(n_estimators=1000,random_state=4)
rf.fit(x_train,y_train)print('Mean Absolute Error:',round(np.mean(errors),2),)
pred = rf.predict(x_test)
print(pred)
errors = abs(pred-y_test)
print('Mean Absolute Error:',round(np.mean(errors),2),)