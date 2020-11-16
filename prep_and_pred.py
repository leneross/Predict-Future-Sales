import pandas as pd
import numpy as np

# Read CSV and define dataframe
original = pd.read_csv('sales_train.csv')
df = original.copy()
df.columns

df1 = df[['date_block_num',
          'shop_id', 
          'item_id',
          'item_cnt_day',
          'item_price'
          ]]

df1['revenue'] = df1['item_cnt_day']*df1['item_price']
del df1['item_price']

# Drop returns
df1.loc[df1['item_cnt_day'] < 0].count()
df1.drop((df1.loc[(df1['item_cnt_day'] <0 )].index), inplace=True)

df1.loc[df1['revenue'] < 0 ].count()
df1.drop((df1.loc[(df1['revenue'] < 0 )].index), inplace= True)

# Delete outlayers
df1.drop((df1.loc[(df1['item_cnt_day'] > 500)].index), inplace=True)
df1['item_cnt_day'].describe()

# Summarize item_cnt_day into month
df2 = df1.groupby(['date_block_num', 'shop_id', 'item_id' ]).sum()
df2.rename(columns={'item_cnt_day' : 'item_cnt_month'}, inplace = True)
df2 = df2.reset_index()

# Create lags for items_count 
for number in range(1,5):
    tmp = (df2.sort_values(by=['date_block_num'], ascending=True).groupby(['shop_id', 'item_id'])
           ['item_cnt_month'].shift(number))
    df2[f"items_cnt_lag{number}"] = tmp.copy()

#Create lags for revenue
for number in range(1,5):
    tmp = (df2.sort_values(by=['date_block_num'], ascending=True).groupby(['shop_id', 'item_id'])
           ['revenue'].shift(number))
    df2[f"revenue_lag{number}"] = tmp.copy()

# Delete monts used for lags
for num in range(4): 
    df2.drop((df2.loc[(df2['date_block_num'] == num)].index), inplace=True)
   
#Convert nanvalues to zero
for num in range(1,5): 
    df2[f"items_cnt_lag{num}"].fillna(0,inplace=True)

for num in range(1,5): 
    df2[f"revenue_lag{num}"].fillna(0,inplace=True)

# Reset index
df2 = df2.set_index(['date_block_num', 'shop_id', 'item_id'])

  
# Divide into features and label

X = df2[['items_cnt_lag1', 'items_cnt_lag2', 'items_cnt_lag3', 'items_cnt_lag4', 'revenue_lag1',
         'revenue_lag2', 'revenue_lag3', 'revenue_lag4']]

y = df2['item_cnt_month']

# Split into train og test
X = X.reset_index()
y = y.reset_index()

x_train = X[(X['date_block_num'] != 33)]
y_train = y[(y['date_block_num'] != 33)]
x_test = X[(X['date_block_num'] == 33)]
y_test = y[(y['date_block_num'] == 33)]

# New index
x_train = x_train.set_index(['date_block_num', 'shop_id', 'item_id'])
y_train = y_train.set_index(['date_block_num', 'shop_id', 'item_id'])
x_test = x_test.set_index(['date_block_num', 'shop_id', 'item_id'])
y_test = y_test.set_index(['date_block_num', 'shop_id', 'item_id'])
    
# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_model = LinearRegression()
lr_model.fit(X=x_train, y=y_train)

y_train_pred = lr_model.predict(x_train)
y_test_pred = lr_model.predict(x_test)

# Testing train
print(mean_absolute_error(y_train,y_train_pred)) 
print(np.sqrt(mean_squared_error(y_train,y_train_pred))) 
                           
# Testing test
print(mean_absolute_error(y_test,y_test_pred))
print(np.sqrt(mean_squared_error(y_test,y_test_pred)))
                        
linearcsv = y_test.copy()
linearcsv['y_test_pred'] = y_test_pred
linearcsv.to_csv('lr_mod.csv')


# XGBoost
import xgboost as xgb

xg_reg = xgb.XGBRegressor() 
xg_reg.fit(X=x_train, y=y_train)
xg_train_pred = xg_reg.predict(x_train)
xg_test_pred = xg_reg.predict(x_test)

# Testing train
print(mean_absolute_error(y_train,xg_train_pred)) 
print(np.sqrt(mean_squared_error(y_train,xg_train_pred)))   
                           
# Testing test
print(mean_absolute_error(y_test,xg_test_pred))  
print(np.sqrt(mean_squared_error(y_test,xg_test_pred))) 
                           
xgboostcsv = y_test.copy()
xgboostcsv['xg_test_pred'] = xg_test_pred
xgboostcsv.to_csv('xg_12layers.csv')


