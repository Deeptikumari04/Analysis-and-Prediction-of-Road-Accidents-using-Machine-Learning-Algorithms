# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:14:58 2020

@author: PANKAJKUMAR
"""

import pandas as pd 
import numpy as np

df = pd.read_csv("C:/Users/PANKAJKUMAR/Documents/Machine Learning/motornew.csv",index_col=0)

# To convert into Samples
df1 = pd.read_csv("C:/Users/PANKAJKUMAR/Documents/Machine Learning/motornew.csv",index_col=0, nrows=2000)
#df1.to_csv("C:/Users/PANKAJKUMAR/Documents/Machine Learning/Samplemotornew1.csv")
df2= df1.drop(["Traffic Control Device"], axis = 1)


dum_df = pd.get_dummies(df2, drop_first=True)
x= dum_df.drop(columns=["Number of Vehicles Involved"], axis=1)
## Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std_scaled=scaler.fit_transform(x)
x= pd.DataFrame(x_std_scaled, index=x.index, columns=x.columns)

y = dum_df.iloc[:,1] 
  

 ############## knn Regressor ############
# Create training and test sets
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.3, 
                                                    random_state=2020)

knn = KNeighborsRegressor(n_neighbors=9)
knn.fit( x_train , y_train )
y_pred = knn.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


########### Linear Regressor #############

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

########## Ridge Regression ############

from sklearn.linear_model import Ridge
clf = Ridge(alpha=2)
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

######### Lasso Regression ###########

from sklearn.linear_model import Lasso
clf = Lasso(alpha=2)
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

############# ElasticNet Regression ############

from sklearn.linear_model import ElasticNet
clf = ElasticNet(alpha=2, l1_ratio=0.6)
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_test)

print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

################  Regression Trees ##########

from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_depth=3,random_state=2020)
clf2 = clf.fit(x_train, y_train)
y_pred = clf2.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) )
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

#############  Random Forest Regressor ##########

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(random_state=2020)
model_rf.fit( x_train , y_train )
y_pred = model_rf.predict(x_test)

print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

parameters = {'max_features': np.arange(1,11)}

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2020,shuffle=True)


cv = GridSearchCV(model_rf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit( x , y )

results_df = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

############# XG Boost Regressor #################

from xgboost import XGBRegressor
clf = XGBRegressor()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))



############### For Testing ##############################


df3 = pd.read_csv("C:/Users/PANKAJKUMAR/Documents/Machine Learning/eg1.csv",index_col=0)

df4= df1.drop(["Traffic Control Device"], axis = 1)

dum_df = pd.get_dummies(df3)

# Standard Scaler
x= dum_df
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_std_scaled=scaler.fit_transform(x)

a1= pd.DataFrame(x_std_scaled,  index=x.index, columns=x.columns )
y_pred = knn.predict(a1)
a= x_test.head(1)
test=knn.predict(a)
test

## Random forest
test1=model_rf.predict(a)
test1

## XG Boost Regressor
test2=clf.predict(a)
test2

## Regression Trees
test2=clf2.predict(a)
test2






