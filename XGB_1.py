import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np # Numpy is often useful, though not strictly required for this basic script

# 1. 加载数据集
try:
    df = pd.read_csv("Train.csv")
except FileNotFoundError:
    print("错误：Train.csv 文件未找到。请确保文件路径正确。")
    df = None
except Exception as e:
    print(f"加载CSV时发生错误: {e}")
    df = None

if df is not None:
    print("数据集加载成功。")
    print(f"数据集形状: {df.shape}")
    print("\n" + "="*50 + "\n")

    # 2. 准备数据
    # 假设 'materials' 列是标识符或者非数值型特征，不直接用于模型训练
    # 如果 'materials' 需要被用作特征，它需要被编码 (例如 one-hot encoding)
    # 这里我们先将其移除
    if 'materials' in df.columns:
        X = df.drop(['property', 'materials'], axis=1)
        print("已移除 'materials' 列。")
    else:
        X = df.drop('property', axis=1)
    
    y = df['property']

    print("特征 (X) 的前5行:")
    print(X.head())
    print("\n目标变量 (y) 的前5行:")
    print(y.head())
    print("\n" + "="*50 + "\n")

    # 3. 分割数据为训练集和测试集
    # random_state 确保每次分割结果一致，便于复现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"测试集大小: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    print("\n" + "="*50 + "\n")

    # 4. 初始化并训练 XGBoost 回归模型
    # 您可以调整 XGBoost 的参数以获得更好的性能
    # 例如: n_estimators, max_depth, learning_rate 等
    print("开始训练 XGBoost 模型...")
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', # 回归任务的目标函数
                                     n_estimators=100,             # 树的数量
                                     learning_rate=0.1,            # 学习率
                                     max_depth=3,                  # 每棵树的最大深度
                                     random_state=42)              # 随机种子

    xgb_regressor.fit(X_train, y_train)
    print("模型训练完成。")
    print("\n" + "="*50 + "\n")

    # 5. 在测试集上进行预测
    print("在测试集上进行预测...")
    y_pred = xgb_regressor.predict(X_test)
    print("预测完成。")
    print("\n部分预测值 vs 真实值:")
    predictions_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(predictions_comparison.head())
    print("\n" + "="*50 + "\n")

    # 6. 评估模型
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # 或者 mse**(0.5)
    r2 = r2_score(y_test, y_pred)

    print("模型性能评估:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"R 平方值 (R²): {r2:.4f}")
    print("\n" + "="*50 + "\n")

    # (可选) 显示特征重要性
    print("特征重要性:")
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': xgb_regressor.feature_importances_})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    print(feature_importances)

else:
    print("由于数据加载失败，无法进行模型训练。")

