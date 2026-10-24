# Big Mart Sales III - EDA + Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')

# ---------------- 1. EDA ----------------
print(train.shape, test.shape)
print(train.isnull().sum())                                         # gaps in Item_Weight, Outlet_Size
print(train.select_dtypes("number").corr()["Item_Outlet_Sales"])    # MRP is the strongest (0.57)
print(train.groupby("Outlet_Type")["Item_Outlet_Sales"].mean())     # store type matters a lot

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
for store_type, color in zip(sorted(train["Outlet_Type"].unique()), colors):
    d = train[train["Outlet_Type"] == store_type]
    ax[0].scatter(d["Item_MRP"], d["Item_Outlet_Sales"], s=6, alpha=0.4, color=color, label=store_type)
ax[0].set(xlabel="Item_MRP", ylabel="Item_Outlet_Sales", title="Sales rise with MRP, at a different rate per store type")
ax[0].legend(markerscale=3)
train.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean().sort_values().plot.barh(
    ax=ax[1], color="#2a78d6", xlabel="Average Item_Outlet_Sales", ylabel="", title="Average sales per outlet")
plt.tight_layout()
plt.show()

# ---------------- 2. Features ----------------
# Sales = MRP x (units sold), and units sold depends on the outlet.
# So each outlet gets its own column holding the item's MRP (0 for other outlets).
def make_features(df):
    return pd.get_dummies(df["Outlet_Identifier"]).mul(df["Item_MRP"], axis=0)

X, y, X_test = make_features(train), train["Item_Outlet_Sales"], make_features(test)

# ---------------- 3. Regression + validation ----------------
model = LinearRegression(fit_intercept=False)
folds = KFold(n_splits=5, shuffle=True, random_state=42)
rmse = -cross_val_score(model, X, y, cv=folds, scoring="neg_root_mean_squared_error").mean()
print("5-fold CV RMSE:", round(rmse, 1))

model.fit(X, y)
print(pd.Series(model.coef_, index=X.columns).round(2))   # units sold per item, per outlet

# ---------------- 4. Submission ----------------
test["Item_Outlet_Sales"] = model.predict(X_test)
test[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]].to_csv("submission_initial.csv", index=False)
