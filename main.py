import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle
import os
sys.version_info
items = pd.read_csv('train_data/items.csv')
shops = pd.read_csv('train_data/shops.csv')
cats = pd.read_csv('train_data/item_categories.csv')
train = pd.read_csv('train_data/sales_train.csv')
# set index to ID to avoid droping it later
test  = pd.read_csv('train_data/test.csv').set_index('ID')
if not os.path.exists("data.pkl"):
    train = train[train.item_price<100000]
    train = train[train.item_cnt_day<1001]
    median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
    train.loc[train.item_price<0, 'item_price'] = median

    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57

    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58

    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id','city_code']]

    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
    # if subtype is nan then type
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
    cats = cats[['item_category_id','type_code', 'subtype_code']]

    items.drop(['item_name'], axis=1, inplace=True)
    len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test)
    ts = time.time()
    matrix = []
    cols = ['date_block_num','shop_id','item_id']
    for i in range(34):
        sales = train[train.date_block_num==i]
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
        
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols,inplace=True)
    time.time() - ts
    train['revenue'] = train['item_price'] *  train['item_cnt_day']
    ts = time.time()
    group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=cols, how='left')
    matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                    .fillna(0)
                                    .clip(0,20) # NB clip target here
                                    .astype(np.float16))
    time.time() - ts
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)
    ts = time.time()
    matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True) # 34 month
    time.time() - ts
    ts = time.time()
    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
    matrix['city_code'] = matrix['city_code'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['type_code'] = matrix['type_code'].astype(np.int8)
    matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
    time.time() - ts
    def lag_feature(df, lags, col):
        tmp = df[['date_block_num','shop_id','item_id',col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        return df
    ts = time.time()
    matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
    matrix.to_pickle('data.pkl')
    # leave test for submission
    gc.collect();
data = pd.read_pickle('data.pkl')
data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
]]
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
del data
gc.collect();
ts = time.time()

model = XGBRegressor(
    max_depth=9,
    n_estimators=800,
    min_child_weight=400, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.04,    
    seed=40)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)
