#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from numpy import math
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv(r"C:\PhD Work\PhD Papers\About Third Paper\Sieved\Poro_Data.csv")
data.head(50)


# In[ ]:


# Using Data for Analysis: This comprise Synthetic Data generated from same location as Well 15/9-F-11-B


# In[4]:


# For Multiple linear regression model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data.loc[:, data.columns != 'KW']
Y = data.loc[:, 'KW']
Model = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 5, test_size=0.10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[5]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[7]:


from pylab import rcParams
import matplotlib

pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Multiple Linear Regression Prediction")
rcParams['figure.figsize'] = 16, 14


# In[614]:


# For Extreme Gradient Boost 

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data.loc[:, data.columns != 'KW']
Y = data.loc[:, 'KW']
Model = XGBRegressor(verbosity=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[615]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[616]:


from pylab import rcParams
import matplotlib

pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Extreme Gradient Boost")
rcParams['figure.figsize'] = 16, 14


# In[12]:


# For Neural Network

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data.loc[:, data.columns != 'KW']
Y = data.loc[:, 'KW']
model = Sequential()
# Add the first hidden layer
model. add(Dense(64, activation='sigmoid', input_dim=5))
# Add the second hidden layer
model.add(Dense(16, activation='relu'))
# Add the third hidden layer
model.add(Dense(4, activation='relu'))
# Add the fourth hidden layer
#model.add(Dense(25, activation='relu'))
# Add the output layer
model.add(Dense(1, activation='relu'))
# compile the model
model.compile(optimizer='adam',loss='mse')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 3, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10)
# Train the model for 200 epochs
model.fit(X_train, Y_train, epochs=100, batch_size = 8, validation_data=(X_val, Y_val))
predictions = model.predict({'Predicted':X_test})
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[13]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[14]:


df1 = pd.DataFrame(Y_test)
df2 = pd.DataFrame(predictions)
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Diagram = plt.scatter(df1, df2, color='r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# In[15]:


from pylab import rcParams
import matplotlib

pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Classic Regression Prediction")
rcParams['figure.figsize'] = 16, 14


# In[73]:


# For decision tree analysis

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sklearn.ensemble as ml

X = data.drop(['KW'], axis=1)
y = data['KW']
model = DecisionTreeRegressor(max_depth=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size=0.05)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[74]:


df1 = pd.DataFrame({'Actual': y_test})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[76]:


from pylab import rcParams

plt.plot(predictions, color='g', label='Predicted')
plt.plot(y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Decision Tree", fontsize=28)
rcParams['figure.figsize'] = 16, 14
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)


# In[68]:


# for Random forest Analysis

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data.drop(['KW'], axis=1)
y = data['KW']
models = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.05)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
model.fit(X_train, y_train)
models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[69]:


df1 = pd.DataFrame({'Actual': y_test})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[72]:


from pylab import rcParams

plt.plot(predictions, color='g', label='Predicted')
plt.plot(y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Random Forest", fontsize=28)
rcParams['figure.figsize'] = 16, 14
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)


# In[77]:


# for K-Nearest Neighbor Analysis

import sklearn.svm as ml
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

models = KNeighborsRegressor(n_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.05)
model.fit(X_train, y_train)

models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[78]:


df1 = pd.DataFrame({'Actual': y_test})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[79]:


from pylab import rcParams

plt.plot(predictions, color='g', label='Predicted')
plt.plot(y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("K-Nearest Neighbor", fontsize=28)
rcParams['figure.figsize'] = 16, 14
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)


# In[147]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data.plot.box(figsize=(16,12))


# In[148]:


from pandas.plotting import scatter_matrix

data.corr()
data.corr()['KW'].sort_values(ascending=False)
scatter_matrix(data, figsize=(16,12), color='r')
plt.show()
plt.yticks(fontsize=14)


# In[151]:


shade = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(shade)
shade[triangle_indices] = True
shade


# In[152]:


plt.figure(figsize=(16,10))
sns.heatmap(data.corr(), mask=shade, annot=True, annot_kws={"size":20})
plt.show
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Spearman's Correlation Heatmap", fontsize=30)
plt.show()


# In[ ]:


# Part Two


# In[ ]:


# Using Data2 for Analysis: This comprise Synthetic Data generated from same location as Well 15/9-F-11-T2


# In[33]:


data2 = pd.read_csv(r"C:\PhD Work\PhD Papers\About Third Paper\Sieved\Poro_Data2.csv")
data2.head(50)


# In[35]:


# For Multiple linear regression model

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data2.loc[:, data.columns != 'KW']
Y = data2.loc[:, 'KW']
Model = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[36]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[37]:


from pylab import rcParams
import matplotlib

pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Multiple Linear Regression Prediction")
rcParams['figure.figsize'] = 16, 14


# In[38]:


# For Neural Network analysis

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

X = data2.loc[:, data.columns != 'KW']
Y = data2.loc[:, 'KW']
model = Sequential()
# Add the first hidden layer
model. add(Dense(200, activation='relu', input_dim=5))
# Add the second hidden layer
model.add(Dense(100, activation='relu'))
# Add the third hidden layer
model.add(Dense(50, activation='relu'))
# Add the fourth hidden layer
model.add(Dense(25, activation='relu'))
# Add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam',loss='mse')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.05)
# Train the model for 200 epochs
model.fit(X_train, Y_train, epochs=200, batch_size = 8, validation_data=(X_val, Y_val))
from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict({'Predicted':X_test})
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[39]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[40]:


from pylab import rcParams
import matplotlib

pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network Prediction")
rcParams['figure.figsize'] = 16, 14


# In[320]:


# For Extreme Gradient Boost 

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data2.loc[:, data2.columns != 'KW']
Y = data2.loc[:, 'KW']
Model = XGBRegressor(verbosity=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[321]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[322]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Extreme Gradient Boosting Prediction")
rcParams['figure.figsize'] = 16, 14


# In[41]:


# For decision tree analysis

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sklearn.ensemble as ml

X = data2.drop(['KW'], axis=1)
y = data2['KW']
model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size=0.05)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[42]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[43]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Decision Tree Prediction")
rcParams['figure.figsize'] = 16, 14


# In[533]:


# for K-Nearest Neighbor Analysis

import sklearn.svm as ml
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

models = KNeighborsRegressor(n_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.1)
model.fit(X_train, y_train)

models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[534]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[228]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Decision Tree Prediction")
rcParams['figure.figsize'] = 16, 14


# In[536]:


# for Random forest Analysis

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data2.loc[:, data2.columns != 'KW']
y = data2.loc[:, 'KW']
models = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
model.fit(X_train, y_train)
models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[537]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[283]:


# for SVM Analysis

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data2.loc[:, data2.columns != 'KW']
y = data2.loc[:, 'KW']

models = SVR(kernel='rbf', C=1e2, gamma=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model.fit(X_train, y_train)

models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[284]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[285]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("SVM Prediction")
rcParams['figure.figsize'] = 16, 14


# In[229]:


plt.figure(figsize=(16,10))
sns.heatmap(data2.corr(), mask=shade, annot=True, annot_kws={"size":20})
plt.show
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Spearman's Correlation Heatmap", fontsize=30)
plt.show()


# In[ ]:


# Using Data 3 for Analysis: This comprise Synthetic Data of density-porosity using a CONSTANT Bulk density for F/9-11-B.


# In[45]:


data3 = pd.read_csv(r"C:\PhD Work\PhD Papers\About Third Paper\Sieved\Poro_Data3.csv")
data3.head(50)


# In[46]:


# For Multiple linear regression model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data3.loc[:, data3.columns != 'KW']
Y = data3.loc[:, 'KW']
Model = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[578]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[579]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("SVM Prediction")
rcParams['figure.figsize'] = 16, 14


# In[80]:


# For Extreme Gradient Boost 

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data3.loc[:, data3.columns != 'KW']
Y = data3.loc[:, 'KW']
Model = XGBRegressor(verbosity=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[81]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[82]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Random Forest", fontsize=28)
rcParams['figure.figsize'] = 16, 14
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)


# In[83]:


# For Neural Network analysis

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.layers import Dense

X = data3.loc[:, data3.columns != 'KW']
Y = data3.loc[:, 'KW']
model = Sequential()
# Add the first hidden layer
model. add(Dense(200, activation='relu', input_dim=3))
# Add the second hidden layer
model.add(Dense(100, activation='relu'))
# Add the third hidden layer
model.add(Dense(50, activation='relu'))
# Add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam',loss='mse')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.05)
# Train the model for 200 epochs
model.fit(X_train, Y_train, epochs=200, batch_size = 8, validation_data=(X_val, Y_val))
predictions = model.predict({'Predicted':X_test})
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[84]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[86]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network", fontsize=28)
rcParams['figure.figsize'] = 16, 14
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)


# In[91]:


# For Extreme Gradient Boost 

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data3.loc[:, data3.columns != 'KW']
Y = data3.loc[:, 'KW']
Model = XGBRegressor(verbosity=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[92]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[93]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Extreme Gradient Boost", fontsize=28)
rcParams['figure.figsize'] = 16, 14
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)


# In[94]:


# for Random forest Analysis

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data3.loc[:, data3.columns != 'KW']
y = data3.loc[:, 'KW']
models = RandomForestRegressor(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.05)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
model.fit(X_train, y_train)
models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[95]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[96]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Random Forest")
rcParams['figure.figsize'] = 16, 14


# In[545]:


# For decision tree analysis

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sklearn.ensemble as ml

X = data3.drop(['KW'], axis=1)
y = data3['KW']
#model = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error,mean_absolute_error
#predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))
gbmt = ml.GradientBoostingRegressor(max_depth=10, 
                           min_samples_leaf=0.1, 
                           random_state=2)
gbmt.fit(X_test, y_test)
predictions = gbmt.predict(X_test)


# In[546]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[345]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Extreme Gradient Boost Prediction")
rcParams['figure.figsize'] = 16, 14


# In[548]:


# for K-Nearest Neighbor Analysis

import sklearn.svm as ml
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

X = data3.drop(['KW'], axis=1)
y = data3['KW']
models = KNeighborsRegressor(n_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.1)
model.fit(X_train, y_train)

models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[549]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[466]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Extreme Gradient Boost Prediction")
rcParams['figure.figsize'] = 16, 14


# In[346]:


data3.corr()
data3.corr()['KW'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
scatter_matrix(data3, figsize=(16,12), color='r', s=50)
plt.yticks(fontsize=14)


# In[347]:


shade = np.zeros_like(data3.corr())
triangle_indices = np.triu_indices_from(shade)
shade[triangle_indices] = True
shade


# In[348]:


plt.figure(figsize=(16,10))
sns.heatmap(data3.corr(), mask=shade, annot=True, annot_kws={"size":20})
plt.show
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Spearman's Correlation Heatmap", fontsize=30)
plt.show()


# In[404]:


# Using Data 4 for Analysis: This comprise Synthetic Data of density-porosity using a CONSTANT Bulk density for F/9-11-T2.


# In[550]:


data4 = pd.read_csv(r"C:\PhD Work\PhD Papers\About Third Paper\Sieved\Poro_Data4.csv")
data4.head(50)


# In[551]:


# For Multiple linear regression model

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data4.loc[:, data4.columns != 'KW']
Y = data4.loc[:, 'KW']
Model = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[552]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[385]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network Prediction")
rcParams['figure.figsize'] = 16, 14


# In[554]:


# for K-Nearest Neighbor Analysis

import sklearn.svm as ml
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

X = data4.drop(['KW'], axis=1)
y = data4['KW']
models = KNeighborsRegressor(n_neighbors=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, test_size=0.1)
model.fit(X_train, y_train)

models.fit(X_test, y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[555]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[389]:


# For Extreme Gradient Boost 

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data4.loc[:, data4.columns != 'KW']
Y = data4.loc[:, 'KW']
Model = XGBRegressor(verbosity=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
model = Model.fit(X_train, Y_train)
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[391]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)


# In[392]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network Prediction")
rcParams['figure.figsize'] = 16, 14


# In[556]:


# For Neural Network analysis

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

X = data4.loc[:, data4.columns != 'KW']
Y = data4.loc[:, 'KW']
model = Sequential()
# Add the first hidden layer
model. add(Dense(100, activation='relu', input_dim=3))
# Add the second hidden layer
model.add(Dense(50, activation='relu'))
# Add the third hidden layer
model.add(Dense(25, activation='relu'))
# Add the fourth hidden layer
model.add(Dense(5, activation='relu'))
# Add the output layer
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam',loss='mse')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)
# Train the model for 200 epochs
model.fit(X_train, Y_train, epochs=200, batch_size = 8, validation_data=(X_val, Y_val))
from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict({'Predicted':X_test})
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[557]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[364]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network Prediction")
rcParams['figure.figsize'] = 16, 14


# In[558]:


# For decision tree analysis

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sklearn.ensemble as ml

X = data4.loc[:, data4.columns != 'KW']
Y = data4.loc[:, 'KW']
model = DecisionTreeRegressor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.12)
model.fit(X_train, Y_train)
from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[559]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[436]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network Prediction")
rcParams['figure.figsize'] = 16, 14


# In[560]:


# for Random forest Analysis

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = data4.loc[:, data4.columns != 'KW']
Y = data4.loc[:, 'KW']
models = RandomForestRegressor(n_estimators=100)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
model.fit(X_train, Y_train)
models.fit(X_test, Y_test)
predictions = models.predict(X_test)
metric_mae = mean_absolute_error(Y_test,predictions)
metric_rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print('MAE:{}, RMSE:{}'.format(metric_mae, metric_rmse))


# In[561]:


df1 = pd.DataFrame({'Actual': round(Y_test,2)})
df2 = pd.DataFrame(predictions)
df2.columns = ['Predicted']
Y= pd.concat([d.reset_index(drop=True) for d in [df1, df2]], axis=1)
Y
pd.set_option('display.max_rows', 100)
Y.head(100)


# In[423]:


pred = plt.plot(predictions, color='g', label='Predicted')
Act = plt.plot(Y_test.values, color='r', label='Actual')
plt.legend(loc="upper left")
plt.title("Neural Network Prediction")
rcParams['figure.figsize'] = 16, 14


# In[437]:


data4.corr()
data4.corr()['KW'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
scatter_matrix(data3, figsize=(16,12), color='r', s=50)
plt.yticks(fontsize=14)


# In[438]:


shade = np.zeros_like(data4.corr())
triangle_indices = np.triu_indices_from(shade)
shade[triangle_indices] = True
shade


# In[439]:


plt.figure(figsize=(16,10))
sns.heatmap(data4.corr(), mask=shade, annot=True, annot_kws={"size":20})
plt.show
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Spearman's Correlation Heatmap", fontsize=30)
plt.show()

