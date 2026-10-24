# %% BigMart Sales - draft: Random Forest regression on units sold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')
print(train.shape, test.shape)

# %% EDA
print(train.isnull().sum())
print(train['Item_Fat_Content'].value_counts())    # LF, low fat, reg are the same as Low Fat / Regular
for df in (train, test):
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
train.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values().plot(kind='barh', title='Average sales by item type')
plt.show()
# item type hardly changes sales, the store and MRP do

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

# %% model: random forest predicts units from store, price, item type, visibility, fat content
def features(df, price):
    x = pd.DataFrame({'price': price, 'visibility': df['Item_Visibility'].to_numpy()})
    for c in ['Outlet_Identifier', 'Item_Type', 'Item_Fat_Content']:
        x[c] = pd.Categorical(df[c], categories=sorted(train[c].unique())).codes
    return x

def predict_sales(fit, new):
    prices = fit.groupby('Item_Identifier')['Price'].median()
    p_fit = fit['Item_Identifier'].map(prices).to_numpy()
    p_new = new['Item_Identifier'].map(prices).fillna(new['Item_MRP']).to_numpy()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=100, random_state=42, n_jobs=-1)
    rf.fit(features(fit, p_fit), fit['Units'])
    return p_new * rf.predict(features(new, p_new))

# %% 5 fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y = train['Item_Outlet_Sales'].to_numpy()
oof = np.zeros(len(train))
for tr_idx, va_idx in kf.split(train):
    oof[va_idx] = predict_sales(train.iloc[tr_idx], train.iloc[va_idx])
print('Random Forest CV RMSE:', round(np.sqrt(np.mean((y - oof) ** 2)), 2))

# %% submission
sub = test[['Item_Identifier', 'Outlet_Identifier']].copy()
sub['Item_Outlet_Sales'] = predict_sales(train, test)
sub.to_csv('submission_random_forest.csv', index=False)
print(sub.head())
