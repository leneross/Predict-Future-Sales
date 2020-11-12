import pandas as pd
import numpy as np

#%% Read CSV and define dataframe

df = pd.read_csv('sales_train.csv')

df1 = df[['date_block_num',
          'shop_id',
          'item_id',
          'item_cnt_day'
          ]]

#Summarize item_cnt_day --> month
df2 = df1.groupby(['date_block_num', 'shop_id', 'item_id' ]).sum()
df2.rename(columns={'item_cnt_day' : 'item_cnt_month'}, inplace = True)
df2 = df2.reset_index()

#%% Find outlayers in PowerBi and change value to median

item20949 = df2.loc[(df2['item_id']==20949)]
item20949.describe()
df2.loc[(df2['item_id']== 20949)&(df2['item_cnt_month'] >150),'item_cnt_month'] = 167
item20949 = df2.loc[(df2['item_id']==20949)]

item3732 = df2.loc[(df2['item_id']==3732)]
item3732.describe()
df2.loc[(df2['item_id']== 3732)&(df2['item_cnt_month'] >41),'item_cnt_month'] = 41
item3732 = df2.loc[(df2['item_id']==3732)]
item3732.describe()

item3731 = df2.loc[(df2['item_id']==3731)]
item3731.describe()
df2.loc[(df2['item_id']== 3731)&(df2['item_cnt_month'] >81),'item_cnt_month'] = 81
item3731 = df2.loc[(df2['item_id']==3731)]
item3731.describe()


#%% Make lags

df2['items_cnt_lag1'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(1))

df2['items_cnt_lag2'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(2))

df2['items_cnt_lag3'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(3))

df2['items_cnt_lag4'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(4))

df2['items_cnt_lag5'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(5))

df2['items_cnt_lag6'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(6))

df2['items_cnt_lag7'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(7))

df2['items_cnt_lag8'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(8))

df2['items_cnt_lag9'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(9))

df2['items_cnt_lag10'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(10))

df2['items_cnt_lag11'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(11))

df2['items_cnt_lag12'] = (df2.sort_values(by=['date_block_num'], ascending=True)
                       .groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(12))

#Delete 12 first month (nan)
indexname = (df2.loc[(df2['date_block_num'] == 0)].index)
indexname2 = (df2.loc[(df2['date_block_num'] == 1)].index)
indexname3 = (df2.loc[(df2['date_block_num'] == 2)].index)
indexname4 = (df2.loc[(df2['date_block_num'] == 3)].index)
indexname5 = (df2.loc[(df2['date_block_num'] == 4)].index)
indexname6 = (df2.loc[(df2['date_block_num'] == 5)].index)
indexname7 = (df2.loc[(df2['date_block_num'] == 6)].index)
indexname8 = (df2.loc[(df2['date_block_num'] == 7)].index)
indexname9 = (df2.loc[(df2['date_block_num'] == 8)].index)
indexname10 = (df2.loc[(df2['date_block_num'] == 9)].index)
indexname11 = (df2.loc[(df2['date_block_num'] == 10)].index)
indexname12 = (df2.loc[(df2['date_block_num'] == 11)].index)

df2.drop(indexname, inplace=True)
df2.drop(indexname2, inplace=True)
df2.drop(indexname3, inplace=True)
df2.drop(indexname4, inplace=True)
df2.drop(indexname5, inplace=True)
df2.drop(indexname6, inplace=True)
df2.drop(indexname7, inplace=True)
df2.drop(indexname8, inplace=True)
df2.drop(indexname9, inplace=True)
df2.drop(indexname10, inplace=True)
df2.drop(indexname11, inplace=True)
df2.drop(indexname12, inplace=True)

#Convert nan to 0
df2['items_cnt_lag1'].fillna(0,inplace=True)
df2['items_cnt_lag2'].fillna(0,inplace=True)
df2['items_cnt_lag3'].fillna(0,inplace=True)
df2['items_cnt_lag4'].fillna(0,inplace=True)
df2['items_cnt_lag5'].fillna(0,inplace=True)
df2['items_cnt_lag6'].fillna(0,inplace=True)
df2['items_cnt_lag7'].fillna(0,inplace=True)
df2['items_cnt_lag8'].fillna(0,inplace=True)
df2['items_cnt_lag9'].fillna(0,inplace=True)
df2['items_cnt_lag10'].fillna(0,inplace=True)
df2['items_cnt_lag11'].fillna(0,inplace=True)
df2['items_cnt_lag12'].fillna(0,inplace=True)


df2 = df2.set_index(['date_block_num', 'shop_id', 'item_id'])

#%% Divide into features and label
X = df2[['items_cnt_lag1', 'items_cnt_lag2', 'items_cnt_lag3','items_cnt_lag4','items_cnt_lag5',
         'items_cnt_lag6','items_cnt_lag7','items_cnt_lag8','items_cnt_lag9','items_cnt_lag10',
         'items_cnt_lag11','items_cnt_lag12']]

y = df2['item_cnt_month']

#Split into train og test
X = X.reset_index()
y = y.reset_index()

x_train = X[(X['date_block_num'] != 33)]
y_train = y[(y['date_block_num'] != 33)]
x_test = X[(X['date_block_num'] == 33)]
y_test = y[(y['date_block_num'] == 33)]

#New index
x_train = x_train.set_index(['date_block_num', 'shop_id', 'item_id'])
y_train = y_train.set_index(['date_block_num', 'shop_id', 'item_id'])
x_test = x_test.set_index(['date_block_num', 'shop_id', 'item_id'])
y_test = y_test.set_index(['date_block_num', 'shop_id', 'item_id'])

#%% Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_model = LinearRegression()
lr_model.fit(X=x_train, y=y_train)

y_train_pred = lr_model.predict(x_train)
y_test_pred = lr_model.predict(x_test)

#Testing train
print(mean_absolute_error(y_train,y_train_pred)) # 1.3000
print(np.sqrt(mean_squared_error(y_train,y_train_pred))) #5.9161
                           
#Testing test
print(mean_absolute_error(y_test,y_test_pred))  #1.3917
print(np.sqrt(mean_squared_error(y_test,y_test_pred))) # 12.4510 
                        
linearcsv = y_test.copy()
linearcsv['y_test_pred'] = y_test_pred
linearcsv.to_csv('linear_3outremoved.csv')



#%% XGBoost
import xgboost as xgb

xg_reg = xgb.XGBRegressor() 
xg_reg.fit(X=x_train, y=y_train)
xg_train_pred = xg_reg.predict(x_train)
xg_test_pred = xg_reg.predict(x_test)

#Train
print(mean_absolute_error(y_train,xg_train_pred)) #1,1786
print(np.sqrt(mean_squared_error(y_train,xg_train_pred)))   # 4,2713
                           
#Testing test
print(mean_absolute_error(y_test,xg_test_pred))  #1,3568
print(np.sqrt(mean_squared_error(y_test,xg_test_pred))) # 12,6243
                           
xgboostcsv = y_test.copy()
xgboostcsv['xg_test_pred'] = xg_test_pred
xgboostcsv.to_csv('xg_12layers.csv')


#%% VIDERE ARBEID
# items ute av sortiment/nye
# ta med pris- kanskje har prisforandring noe å si for antall items solgt
# sesongvariasjoner? Bruke tidserie på en anenn måte? 
#Tilbakemeldinger: 