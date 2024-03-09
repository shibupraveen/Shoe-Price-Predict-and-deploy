import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df=pd.read_csv('product_data.csv')
print("Shape of the dataframe of reviews :",df.shape)
print("data type of reviews \n",df.dtypes)


def missing_values(x):
    print(x.isnull().sum())
print('missing values = ',missing_values(df))


def duplicated_values(x):
    print(x.duplicated().sum())
print('Duplicated values = ',duplicated_values(df))


df = df.drop_duplicates()
print("Shape of the dataframe of reviews post removing duplicates:",df.shape)


def mapping_values(x):
    if x == 'credit card':
        return 1
    elif x == 'debit card':
        return 2
    elif x == 'mobile payments':
        return 3
    else:
        return 4
df['payment_type'] = df['payment_type'].map(mapping_values)


y = df['price'].values
df.drop('price', axis=1, inplace=True)
X=df.copy()


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('Linear_regression_score',r2_score(y_test,y_pred)*100)

RF=RandomForestRegressor()
RF.fit(X_train,y_train)
y_pred=RF.predict(X_test)
print('Random forest score',r2_score(y_test,y_pred)*100)


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(max_depth=7)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('Decision tree score',r2_score(y_test,y_pred)*100)

from sklearn.linear_model import Lasso
reg = Lasso(alpha=1.0)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('Lasso',r2_score(y_test,y_pred)*100)


pickle.dump(RF, open('rf_model.pkl','wb'))
print("dumping complete")

model = pickle.load(open('rf_model.pkl','rb'))

