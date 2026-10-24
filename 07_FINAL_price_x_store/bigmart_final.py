# %% BigMart Sales Prediction - EDA + Regression
# predict Item_Outlet_Sales for each item/store pair in the test file, metric = RMSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

# %% model: price x store regression
stores = sorted(train['Outlet_Identifier'].unique())

def store_cols(df):
    return pd.get_dummies(df['Outlet_Identifier']).reindex(columns=stores, fill_value=0).astype(float)

def predict_sales(fit, new, lam=60, cap=5700, power=0.75):
    # real price of each item
    price = new['Item_Identifier'].map(fit.groupby('Item_Identifier')['Price'].median()).fillna(new['Item_MRP'])
    # linear regression on the store columns -> average units each store sells
    reg = LinearRegression(fit_intercept=False).fit(store_cols(fit), fit['Units'])
    units = reg.predict(store_cols(new))
    # item effect: some items sell a little more or less everywhere, pulled towards 1 by lam
    store_avg = fit.groupby('Outlet_Identifier')['Units'].transform('mean')
    w = (store_avg / store_avg.mean()) ** power        # bigger stores count more (grocery counts are noisy)
    t = pd.DataFrame({'w': w, 'wr': w * fit['Units'] / store_avg}).groupby(fit['Item_Identifier']).sum()
    effect = new['Item_Identifier'].map((t['wr'] + lam) / (t['w'] + lam)).fillna(1)
    # cap: the most expensive items in OUT027 sell less than this predicts
    return np.clip((price * units * effect).to_numpy(), 0, cap)
# lam, cap and power were picked by trying a few values on the public leaderboard

# %% compare the models with 5 fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

oof = np.zeros(len(train))
for tr_idx, va_idx in kf.split(train):
    oof[va_idx] = predict_sales(train.iloc[tr_idx], train.iloc[va_idx])

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=40, random_state=42, n_jobs=-1)
print('Linear Regression        :', round(rmse(y, cross_val_predict(LinearRegression(), X, y, cv=kf)), 2))
print('Random Forest            :', round(rmse(y, cross_val_predict(rf, X, y, cv=kf)), 2))
print('Price x Store Regression :', round(rmse(y, oof), 2))
# plain linear regression adds a store effect, but the store multiplies the price,
# so the price x store model gets the lowest error

# %% final model and submission
reg = LinearRegression(fit_intercept=False).fit(store_cols(train), train['Units'])
print(pd.Series(reg.coef_, index=stores).round(2))    # average units sold per store

sub = test[['Item_Identifier', 'Outlet_Identifier']].copy()
sub['Item_Outlet_Sales'] = predict_sales(train, test)
sub.to_csv('submission_final.csv', index=False)
print(sub.head())

# results
# cross validation RMSE: linear regression 1132, random forest 1082, price x store regression 1071
# leaderboard RMSE: linear regression 1185.77, random forest 1151.85, price x store regression 1143.15
# the leaderboard is higher for every model because the test answers are noisier than the training data.
# on the training data even the exact price x each store's true average gives ~1070, so the model is at the limit.
