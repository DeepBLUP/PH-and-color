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


# 初始化基本回归模型
base_estimators = [
    #('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100)),
    #('SVR', SVR()),
    #('Ridge', Ridge()),
    #('Lasso', Lasso()),
    #('ElasticNet', ElasticNet()),
    #('Bayesian Ridge', BayesianRidge()),
    #('Decision Tree', DecisionTreeRegressor()),
    ('AdaBoost', AdaBoostRegressor(n_estimators=100)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators=100)),
    #('K-Nearest Neighbors', KNeighborsRegressor()),
    #('MLP', MLPRegressor()),
    #('Linear SVR', LinearSVR()),
    #('NuSVR', NuSVR()),
    #('Gaussian Process', GaussianProcessRegressor()),
    ('LightGBM', LGBMRegressor(n_estimators=100)),
    ('XGBoost', XGBRegressor(n_estimators=100)),
    ('CatBoost', CatBoostRegressor(iterations=100))
]



# 选择一个基本回归模型作为基本估计器
base_regressor = RandomForestRegressor(n_estimators=100)  # 这里选择随机森林模型作为示例

# 初始化BaggingRegressor
bagging_regressor = BaggingRegressor(base_estimator=base_regressor, n_estimators=10, random_state=42)

# 训练模型
bagging_regressor.fit(X_train, y_train)

# 预测
y_pred = bagging_regressor.predict(X_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # 绘制参考线
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('True Value vs Predicted Value: BaggingRegressor')

plt.savefig(r'C:\python\2024\3\3屠宰\4.3\眼肌pH\BaggingRegressor.png')
