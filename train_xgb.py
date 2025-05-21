import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# Initialize and train model
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print(f'RMSE: {rmse:.4f}')
print(f'R^2: {r2:.4f}')
