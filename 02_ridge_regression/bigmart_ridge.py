# %% BigMart Sales - draft: Ridge regression on units sold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')
print(train.shape, test.shape)

# %% EDA
print(train.isnull().sum())
print(train[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']].corr()['Item_Outlet_Sales'])
train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean().sort_values().plot(kind='bar', title='Average sales by store')
plt.show()
# MRP is the strongest feature and the store matters a lot

# %% sales = units x price
def find_units(sales, mrp):
    best, best_gap = None, 99
    for u in range(1, 81):
        gap = sales / u - mrp
        if abs(gap) <= 2.0001 and abs(gap * 10 - round(gap * 10)) < 0.002 and abs(gap) < best_gap:
            best, best_gap = u, abs(gap)
    return best

train['Units'] = [find_units(s, m) for s, m in zip(train['Item_Outlet_Sales'], train['Item_MRP'])]
train['Price'] = train['Item_Outlet_Sales'] / train['Units']
print(train.groupby('Outlet_Identifier')['Units'].mean().round(1))

# %% model: ridge regression of units on store + item columns, then x price
cols = ['Outlet_Identifier', 'Item_Identifier']

def predict_sales(fit, new):
    enc = OneHotEncoder(handle_unknown='ignore').fit(fit[cols])
    reg = Ridge(alpha=30).fit(enc.transform(fit[cols]), fit['Units'])     # alpha keeps item effects small
    units = reg.predict(enc.transform(new[cols]))
    price = new['Item_Identifier'].map(fit.groupby('Item_Identifier')['Price'].median()).fillna(new['Item_MRP'])
    return (price * units).to_numpy()

# %% 5 fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y = train['Item_Outlet_Sales'].to_numpy()
oof = np.zeros(len(train))
for tr_idx, va_idx in kf.split(train):
    oof[va_idx] = predict_sales(train.iloc[tr_idx], train.iloc[va_idx])
print('Ridge regression CV RMSE:', round(np.sqrt(np.mean((y - oof) ** 2)), 2))

# %% submission
sub = test[['Item_Identifier', 'Outlet_Identifier']].copy()
sub['Item_Outlet_Sales'] = predict_sales(train, test)
sub.to_csv('submission_ridge.csv', index=False)
print(sub.head())
