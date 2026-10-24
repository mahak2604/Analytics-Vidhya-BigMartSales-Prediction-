# %% BigMart Sales - draft: Gradient Boosting regression on sales (cleaned features)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')
print(train.shape, test.shape)

# %% EDA
print(train.isnull().sum())                        # Item_Weight and Outlet_Size have missing values
print(train['Item_Fat_Content'].value_counts())    # LF, low fat, reg = Low Fat / Regular written differently
print('rows with 0 visibility:', (train['Item_Visibility'] == 0).sum())
print(train[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']].corr()['Item_Outlet_Sales'])
train['Item_Outlet_Sales'].plot(kind='hist', bins=50, title='Sales distribution (right skewed)')
plt.show()

# %% cleaning
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

# %% model + 5 fold cross validation
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, min_samples_leaf=50,
                               subsample=0.8, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(gb, X, y, cv=kf)
print('Gradient Boosting CV RMSE:', round(np.sqrt(np.mean((y - oof) ** 2)), 2))

# %% submission
gb.fit(X, y)
sub = test[['Item_Identifier', 'Outlet_Identifier']].copy()
sub['Item_Outlet_Sales'] = np.clip(gb.predict(X_test), 0, None)
sub.to_csv('submission_gradient_boosting.csv', index=False)
print(sub.head())
