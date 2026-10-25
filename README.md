# Analytics Vidhya – Big Mart Sales III Prediction
Course: Advanced Modelling in Business Analytics Competition: Big Mart Sales III (Analytics Vidhya practice problem) Result: Leaderboard RMSE 1143.15 · Rank 127

Team
Name	Roll no.	Presents
Praveer Singh Bedi	L006	Part 1 – Problem, data and EDA
Mahak Wadhawan	L074	Part 2 – Key discovery and our model
Yash Shah	L056	Part 3 – Testing, results and conclusions
Problem
BigMart collected 2013 sales data for 1,559 products across 10 stores. The task is to predict Item_Outlet_Sales for every product–store pair in the test set.

Training data: 8,523 rows (sales known)
Test data: 5,681 rows (sales hidden)
Metric: RMSE (Root Mean Squared Error), lower is better
Problem type: supervised machine learning, regression
Part 1 – Data and EDA
We studied the data before modelling:

Missing values: Item_Weight (1,463 rows) and Outlet_Size (2,410 rows)
Inconsistent labels: "LF", "low fat" and "reg" mean "Low Fat" and "Regular"
Impossible values: 526 products with zero visibility
Price (MRP) has the strongest link with sales (correlation 0.57)
Store type matters most: grocery stores average ~340 in sales, the largest supermarket (OUT027) ~3,694
Item type barely matters
Cleaning (used for the comparison models): fixed fat labels, filled missing weight from the same product in other stores, set missing size to "Unknown", replaced zero visibility with the product's average, added Outlet_Age, and one-hot encoded text columns.

Part 2 – Key discovery and model
Discovery: every sale equals a whole number of units sold × the item's real price. Example: 3,735.138 ÷ 15 units = 249.01, close to the item's MRP of 249.81. This holds for all 8,523 training rows.

Units sold depend mainly on the store: grocery stores sell about 2 units of an item, normal supermarkets 14–17, and OUT027 about 27.

Model – Price × Store regression:

Sales = Price × Units the store sells × small item adjustment   (capped at 5,700)
Units per store: Linear Regression on the 10 store dummy columns, without an intercept (fit_intercept=False) to avoid the dummy-variable trap. Each coefficient is that store's average units.
Item adjustment: items that sell slightly more or less everywhere get a small adjustment, shrunk towards 1 (lam = 60) because each item appears in only ~5 stores; larger stores count more (power = 0.75).
Cap: predictions are capped at 5,700 because the most expensive items in OUT027 sell slightly fewer units.
lam, power and the cap were chosen by comparing leaderboard scores.

Part 3 – Testing and results
Models were compared with 5-fold cross-validation (training on 4 parts, testing on the hidden part, repeated 5 times).

Model	CV RMSE	Leaderboard RMSE
Linear Regression (all columns)	1131.99	1185.77
Random Forest	1082.41	1151.85
Price × Store regression (final)	1070.69	1143.15
We also tested LightGBM, XGBoost, CatBoost, Gradient Boosting, Ridge, Poisson regression, weighted regression and model blends. All scored worse, so we kept the simplest model that follows how sales actually work.

Plain linear regression adds a fixed store effect, but a store sells a set number of units of every item, so its effect multiplies the price. Our model captures this directly.

The leaderboard score is higher than the cross-validation score for every model because the test data is noisier. Even using each store's true average units gives an RMSE of about 1,070 on the training data, so the remaining error is random variation that no column explains.

Repository structure
Final Folder/
  bigmart_127.py          final code (EDA + regression)
  submission_127.csv      final submission (RMSE 1143.15, rank 127)
bigmart_drafts/           earlier drafts of the other models we tried
AnalyticsVidhya_L074_L056_L006.pdf
README.md
How to run
Install: pip install pandas numpy scikit-learn
Put train_v9rqX0R.csv and test_AbJTz2l.csv in the same folder as bigmart_127.py
Run: python bigmart_127.py
The script prints the EDA results and the cross-validation scores, and writes submission.csv.

Limitations and next steps
lam, power and the cap were tuned on the leaderboard; tuning everything by cross-validation would be cleaner.
We focused on prediction error; a statistical OLS model would add p-values and confidence intervals.
With date, promotion or seasonal data, time-based features could explain more of the variation.
Code was written with the help of GenAI tools; the analysis and modelling decisions were made by the team.
