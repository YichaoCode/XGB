import pandas as pd

# Load the dataset
# 假设 CSV 文件名为 "Train.csv" 并且与脚本在同一目录下
# 如果文件在其他位置，请提供完整路径
try:
    df = pd.read_csv("Train.csv")
except FileNotFoundError:
    print("错误：Train.csv 文件未找到。请确保文件路径正确。")
    df = None
except Exception as e:
    print(f"加载CSV时发生错误: {e}")
    df = None

if df is not None:
    # 显示数据集的前5行
    print("数据集前5行:")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # 显示数据集的基本信息
    print("数据集信息:")
    df.info()
    print("\n" + "="*50 + "\n")

    # 生成数值型列的描述性统计数据
    print("数值型列的描述性统计:")
    print(df.describe(include='number'))
    print("\n" + "="*50 + "\n")

    # 生成对象/类别型列的描述性统计数据
    # include='object' 用于分析字符串类型的列
    # include='all' 可以同时显示数值型和对象型列的统计（但某些统计指标可能只对特定类型有意义）
    print("对象/类别型列的描述性统计:")
    print(df.describe(include='object'))
    print("\n" + "="*50 + "\n")

    # 检查每列的缺失值数量
    print("每列的缺失值数量:")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")

    # 如果您想查看特定列的唯一值及其计数（对于类别型数据特别有用）
    # 例如，分析 'materials' 列
    if 'materials' in df.columns:
        print("'materials' 列的唯一值计数:")
        print(df['materials'].value_counts())
        print("\n" + "="*50 + "\n")
else:
    print("由于数据加载失败，无法进行分析。")
