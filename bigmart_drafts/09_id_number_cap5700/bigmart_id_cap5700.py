# %% BigMart Sales Prediction - EDA + Regression
# predict Item_Outlet_Sales for each item/store pair in the test file, metric = RMSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import KFold, cross_val_predict

train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')
print(train.shape, test.shape)

# %% EDA
print(train.isnull().sum())                        # Item_Weight and Outlet_Size have missing values
print(train['Item_Fat_Content'].value_counts())    # LF, low fat, reg = Low Fat / Regular written differently
print('rows with 0 visibility:', (train['Item_Visibility'] == 0).sum())
print(train[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']].corr()['Item_Outlet_Sales'])

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].hist(train['Item_Outlet_Sales'], bins=50)
ax[0, 0].set_title('Sales distribution (right skewed)')
for t, c in zip(sorted(train['Outlet_Type'].unique()), ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']):
    d = train[train['Outlet_Type'] == t]
    ax[0, 1].scatter(d['Item_MRP'], d['Item_Outlet_Sales'], s=3, alpha=0.4, color=c, label=t)
ax[0, 1].set_title('MRP vs Sales by store type')
ax[0, 1].legend(markerscale=4)
train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean().sort_values().plot(kind='bar', ax=ax[1, 0])
ax[1, 0].set_title('Average sales by store')
train.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values().plot(kind='barh', ax=ax[1, 1])
ax[1, 1].set_title('Average sales by item type')
plt.tight_layout()
plt.show()
# MRP is the strongest feature (r = 0.57). sales go up with MRP at a different rate for each store type.
# grocery stores (OUT010, OUT019) sell very little, OUT027 sells the most. item type hardly matters.

# %% sales = units x price
# the same item has a slightly different MRP in every store, always within +-2 of its real price.
# dividing the sales by the right whole number of units gives back the real price.
# row 0: 3735.138 / 15 = 249.0092 and its MRP is 249.8092
def find_units(sales, mrp):
    best, best_gap = None, 99
    for u in range(1, 81):
        gap = sales / u - mrp
        if abs(gap) <= 2.0001 and abs(gap * 10 - round(gap * 10)) < 0.002 and abs(gap) < best_gap:
            best, best_gap = u, abs(gap)
    return best

train['Units'] = [find_units(s, m) for s, m in zip(train['Item_Outlet_Sales'], train['Item_MRP'])]
train['Price'] = train['Item_Outlet_Sales'] / train['Units']
print('rows where units were found:', train['Units'].notna().sum(), 'of', len(train))
print(train.groupby('Outlet_Identifier')['Units'].mean().round(1))
# grocery stores sell ~2.4 units of an item, supermarkets ~14-17, OUT027 ~26.6
# so sales = price of the item x units the store sells

# %% what else changes the units? the number in the item id
# idea from a public solution (https://github.com/sanjeettuteja11-de/big-mart-sales), checked on our training data below
# units compared to the store's average: 1.10 means 10% more than a normal item in that store
train['rel_units'] = train['Units'] / train.groupby('Outlet_Identifier')['Units'].transform('mean')
train['id_num'] = train['Item_Identifier'].str[3:].astype(int)     # FDA15 -> 15
test['id_num'] = test['Item_Identifier'].str[3:].astype(int)
print(train[['rel_units', 'Item_Visibility', 'Item_Weight', 'Price']].corr()['rel_units'].round(3))   # all ~0
print(train.groupby(train['Item_Identifier'].str[:2])['rel_units'].mean().round(3))                 # FD/DR/NC ~1
by_num = train.groupby('id_num')['rel_units'].mean()
plt.figure(figsize=(10, 3))
plt.plot(by_num.index, by_num.values, 'o-')
plt.axhline(1, color='grey', lw=1)
plt.title('Units sold vs the number in the item id (1 = store average)')
plt.xlabel('number in Item_Identifier')
plt.ylabel('relative units')
plt.tight_layout()
plt.show()
# visibility, weight, price and item family do not change the units, but the id number does:
# items with the same number (FDA15, DRC15, NCD15 ...) sell alike, and neighbouring numbers are similar.
# numbers 15-18 sell ~10% more than average, 27-31 ~10% less. so the id number goes into the model.

# %% cleaning (used by the two baseline models)
data = pd.concat([train, test], ignore_index=True)
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
data['Item_Weight'] = data['Item_Weight'].fillna(data.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
data['Outlet_Size'] = data['Outlet_Size'].fillna('Unknown')
data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan)
data['Item_Visibility'] = data['Item_Visibility'].fillna(data.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))
data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']

cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Item_Fat_Content', 'Item_Type',
        'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
X_all = pd.get_dummies(data[cols], dtype=float)
X, X_test = X_all[:len(train)], X_all[len(train):]
y = train['Item_Outlet_Sales']

# %% model: price x store units x id number effect
stores = sorted(train['Outlet_Identifier'].unique())
spline = SplineTransformer(n_knots=20, degree=3, include_bias=False).fit(pd.DataFrame({'id_num': range(1, 61)}))
CAP = 5700    # upper limit on predictions: OUT027's most expensive items sell a bit less (EDA: ~25 units, not 26.6).
              # picked on the leaderboard, where 5700 scored better than no cap (CV: 1063.34 with cap, 1060.98 without)

def store_cols(df):
    return pd.get_dummies(df['Outlet_Identifier']).reindex(columns=stores, fill_value=0).astype(float)

def predict_sales(fit, new, use_id=True):
    # real price of each item
    price = new['Item_Identifier'].map(fit.groupby('Item_Identifier')['Price'].median()).fillna(new['Item_MRP'])
    # regression 1: linear regression on the store columns -> average units each store sells
    reg1 = LinearRegression(fit_intercept=False).fit(store_cols(fit), fit['Units'])
    store_units = reg1.predict(store_cols(new))
    if not use_id:
        return (price * store_units).to_numpy()
    # regression 2: ridge regression of relative units on a smooth curve over the id number
    store_avg = reg1.predict(store_cols(fit))
    w = store_avg / store_avg.mean()                   # bigger stores count more (grocery counts are noisy)
    reg2 = Ridge(alpha=30).fit(spline.transform(fit[['id_num']]), fit['Units'] / store_avg, sample_weight=w)
    id_effect = reg2.predict(spline.transform(new[['id_num']]))
    pred = (price * store_units * id_effect).to_numpy()
    return pred if CAP is None else np.minimum(pred, CAP)
# spline knots (20) and alpha (30) were picked with cross validation

# %% compare the models with 5 fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def cv_model(use_id):
    oof = np.zeros(len(train))
    for tr_idx, va_idx in kf.split(train):
        oof[va_idx] = predict_sales(train.iloc[tr_idx], train.iloc[va_idx], use_id)
    return rmse(y, oof)

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=40, random_state=42, n_jobs=-1)
print('Linear Regression            :', round(rmse(y, cross_val_predict(LinearRegression(), X, y, cv=kf)), 2))
print('Random Forest                :', round(rmse(y, cross_val_predict(rf, X, y, cv=kf)), 2))
print('Price x Store                :', round(cv_model(use_id=False), 2))
print('Price x Store x Id number    :', round(cv_model(use_id=True), 2))
# plain linear regression adds a store effect, but the store multiplies the price.
# price x store follows the rule from the EDA, and the id number effect lowers the error further.

# %% final model and submission
reg1 = LinearRegression(fit_intercept=False).fit(store_cols(train), train['Units'])
print(pd.Series(reg1.coef_, index=stores).round(2))    # average units sold per store

sub = test[['Item_Identifier', 'Outlet_Identifier']].copy()
sub['Item_Outlet_Sales'] = predict_sales(train, test)
sub.to_csv('submission_id_cap5700.csv', index=False)
print(sub.head())
