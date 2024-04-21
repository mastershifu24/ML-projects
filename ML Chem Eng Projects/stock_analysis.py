import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from skopt import BayesSearchCV

# Load the preprocessed data
df = pd.read_csv('preprocessed_stock_data_with_features.csv', index_col=0)

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.describe())
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
sns.histplot(df['Stock 1 RSI'], kde=True)
plt.title('Distribution of Stock 1 RSI')
plt.show()
sns.scatterplot(data=df, x='Stock 1', y='Stock 2')
plt.title('Scatter Plot: Stock 1 vs Stock 2')
plt.show()

# Feature Selection
X = df.drop(columns=['Stock 1'])
y = df['Stock 1']
rf = RandomForestRegressor()
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
print(feature_importance)

# Model Building and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Explained Variance Score: {evs}")

# Hyperparameter Tuning using Bayesian Optimization
param_space = {
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5),
    'max_features': (0.1, 1.0)
}
opt = BayesSearchCV(
    RandomForestRegressor(),
    param_space,
    n_iter=50,  # Number of iterations
    cv=KFold(n_splits=5, shuffle=True, random_state=0),  # Cross-validation strategy
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
opt.fit(X, y)
best_params = opt.best_params_
print(f"Best Hyperparameters: {best_params}")
rf = RandomForestRegressor(**best_params)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
print(f"Cross-Validation MSE Scores: {mse_scores}")
print(f"Mean MSE: {np.mean(mse_scores)}")

# Final Model Building with Optimized Hyperparameters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf = RandomForestRegressor(**best_params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")