import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import optuna # 引入 Optuna
import matplotlib.pyplot as plt # 引入 matplotlib 用于绘图

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

    # 将 Pandas DataFrame/Series 转换为 Numpy array
    X_np = X.values
    y_np = y.values

    # 3. 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    print(f"训练集大小: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"测试集大小: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    print("\n" + "="*50 + "\n")

    # 4. 定义 Optuna 的目标函数
    n_cv_splits = min(5, len(X_train)) 
    if n_cv_splits < 2:
        print(f"警告：训练样本过少 ({len(X_train)})，无法进行有效的交叉验证 (n_splits={n_cv_splits})。优化时将不使用CV，这可能导致过拟合。")
        kfold = None
    else:
        kfold = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
        model = xgb.XGBRegressor(**params)
        if kfold:
            scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
            return -np.mean(scores)
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            return mean_squared_error(y_train, y_pred_train)

    print("开始 Optuna 超参数优化...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    n_optuna_trials = 30
    if len(X_train) < 10 and kfold:
        print(f"训练集非常小 ({len(X_train)}), 减少优化试验次数。")
        n_optuna_trials = max(10, len(X_train) * 2)
    study.optimize(objective, n_trials=n_optuna_trials, n_jobs=-1)

    print("Optuna 优化完成。")
    print(f"最佳试验的均方误差 (交叉验证): {study.best_value:.4f}")
    print("找到的最佳超参数:")
    best_params = study.best_params
    print(best_params)
    print("\n" + "="*50 + "\n")

    # 6. 使用找到的最佳超参数训练最终模型
    print("使用最佳超参数训练最终模型...")
    final_xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    final_xgb_regressor.fit(X_train, y_train)
    print("最终模型训练完成。")
    print("\n" + "="*50 + "\n")

    # 7. 在测试集上进行预测
    print("在测试集上进行预测...")
    y_pred_test = final_xgb_regressor.predict(X_test)
    print("预测完成.")
    print("\n部分预测值 vs 真实值 (测试集):")
    predictions_comparison_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    print(predictions_comparison_test.head())
    print("\n" + "="*50 + "\n")

    # 8. 评估最终模型
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    print("最终模型在测试集上的性能评估:")
    print(f"均方误差 (MSE): {mse_test:.4f}")
    print(f"均方根误差 (RMSE): {rmse_test:.4f}")
    print(f"R 平方值 (R²): {r2_test:.4f}")
    print("\n" + "="*50 + "\n")

    # 9. 特征重要性分析与可视化
    print("最终模型的特征重要性:")
    X_train_df = pd.DataFrame(X_train, columns=X.columns) # 使用原始特征名
    
    feature_importances_df = pd.DataFrame({
        'feature': X_train_df.columns,
        'importance': final_xgb_regressor.feature_importances_
    })
    feature_importances_df = feature_importances_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\n所有特征的重要性 (降序):")
    print(feature_importances_df)

    # 打印前十个最重要的特征
    top_n = 10
    print(f"\n前 {top_n} 个最重要的特征:")
    print(feature_importances_df.head(top_n))
    print("\n" + "="*50 + "\n")

    # 可视化特征重要性，并高亮前十个
    plt.figure(figsize=(12, 8))
    # 为所有条形创建颜色列表，默认为蓝色
    colors = ['skyblue'] * len(feature_importances_df)
    # 将前 top_n 个特征的颜色改为红色
    for i in range(min(top_n, len(colors))):
        colors[i] = 'salmon'
    
    # 创建条形图
    # 由于特征重要性已降序排列，直接绘制即可
    bars = plt.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
    plt.xlabel('特征重要性 (Importance Score)')
    plt.ylabel('特征 (Feature)')
    plt.title('XGBoost 模型特征重要性 (前10高亮)')
    plt.gca().invert_yaxis() # 将最重要的特征显示在顶部
    
    # 为条形图添加数值标签 (可选, 如果条形太多可能会显得拥挤)
    # for bar in bars:
    #     plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
    #              f'{bar.get_width():.3f}',
    #              va='center', ha='left')
             
    plt.tight_layout() # 调整布局以防止标签重叠
    plt.show() # 显示图表
    print("特征重要性图表已生成。")
    print("\n" + "="*50 + "\n")

    # 10. Optuna 可视化 (如果安装了 matplotlib)
    print("尝试生成 Optuna 可视化图表...")
    try:
        # 如果在非交互式环境（如脚本直接运行结束）中，.show() 可能不会持久显示窗口
        # 在Jupyter Notebook或IPython中通常表现更好
        fig_opt_history = optuna.visualization.plot_optimization_history(study)
        fig_opt_history.show()
        print("优化历史图已生成。")

        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.show()
        print("参数切片图已生成。")

        fig_param_importances = optuna.visualization.plot_param_importances(study)
        fig_param_importances.show()
        print("Optuna参数重要性图已生成。")
        
    except ImportError:
        print("\n请安装 matplotlib 以查看 Optuna 可视化图表: pip install matplotlib plotly")
        print("注意: Optuna 的某些可视化可能更推荐使用 plotly。")
    except Exception as e_vis:
        print(f"\n生成 Optuna 可视化时出错: {e_vis}")
        print("如果图表未显示，请确保您的环境支持GUI窗口弹出，或者在Jupyter Notebook等交互式环境运行。")

else:
    print("由于数据加载失败，无法进行模型训练。")

