import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppress the glibc FutureWarning from xgboost
warnings.filterwarnings(
    "ignore",
    message=".*glibc.*",
    category=FutureWarning,
    module="xgboost.core",
)
from xgboost import XGBRegressor

# Load dataset
# Train.csv uses utf-8-sig encoding
data = pd.read_csv('Train.csv', encoding='utf-8-sig')

# Select features (drop target and non-numeric identifier)
X = data.drop(columns=['property', 'materials'])
y = data['property']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter grid for tuning
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}

# Perform grid search with 5-fold cross-validation
base_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predict and evaluate
r = best_model.predict(X_test)
# Older versions of scikit-learn may not support the ``squared``
# argument for ``mean_squared_error``. Compute RMSE manually for
# compatibility.
rmse = mean_squared_error(y_test, r) ** 0.5
r2 = r2_score(y_test, r)

print("Best parameters:", grid_search.best_params_)
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
