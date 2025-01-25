from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
from math import sqrt
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 调整路径为您的文件实际路径
filepath = r'C:\python\2024\3\3屠宰\4.3\眼肌pH\pH无固定效应.csv'
# 尝试使用utf-8编码读取

try:
    df = pd.read_csv(filepath, encoding='utf-8')
except UnicodeDecodeError:
    # 如果utf-8失败，尝试使用gbk编码读取
    df = pd.read_csv(filepath, encoding='gbk')


df_cleaned = df.dropna(axis=0)
#剩余288条

# 确定哪些列是字符串类型的性状
#categorical_columns = ['公司', '采样时间','猪种']  # 用实际的列名替换'column1', 'column2', ...

# 将字符串性状转换为独热编码
#df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_columns)


X = df_cleaned.drop(columns=['ApH48h','BpH48h'])
y = df_cleaned['ApH48h']
#y = df_cleaned['BpH48h']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

# 构建多种模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
    'SVR': SVR(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'Bayesian Ridge': BayesianRidge(),
    'Decision Tree': DecisionTreeRegressor(),
    'AdaBoost': AdaBoostRegressor(n_estimators=100),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100),
    #'Bagging': BaggingRegressor(n_estimators=100),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'MLP': MLPRegressor(),
    'Linear SVR': LinearSVR(),
    'NuSVR': NuSVR(),
    'Gaussian Process': GaussianProcessRegressor(),
    'LightGBM': LGBMRegressor(n_estimators=100),
    'XGBoost': XGBRegressor(n_estimators=100),
    'CatBoost': CatBoostRegressor(iterations=100)
}

# 创建一个空的数据框
results_df = pd.DataFrame(columns=['Model', 'R^2','RMSE', 'MAE', 'MAPE'])

# 假设 results 是一个包含了模型名称和模型对象的字典
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # 计算平均绝对百分比误差（MAPE）
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # 将结果添加到数据框中
    results_df = pd.concat(
        [results_df, pd.DataFrame([[name, r2,rmse, mae,  mape]], columns=['Model', 'R^2','RMSE', 'MAE', 'MAPE'])],
        ignore_index=True)
# 保存结果为 CSV 文件
#results_df.to_csv(r'C:\python\2024\3\3屠宰\4.3\眼肌pH\眼肌pHA组.csv', index=False)



import matplotlib.pyplot as plt
import numpy as np



# Loop through each model, fit, predict, and plot
for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Reference line
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'True Value vs Predicted Value: {name} A')

    # Save the figure
    file_name = f'{name.replace(" ", "_")}A.png'
    save_path = f'C:\\python\\2024\\3\\3屠宰\\4.3\\眼肌pH\\A组\\{file_name}'

    # Save the figure
    plt.savefig(save_path)
    plt.close()