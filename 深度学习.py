

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd


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


from sklearn.ensemble import VotingRegressor

# Create a list of tuples containing the model name and the model itself
estimators = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100)),
    ('SVR', SVR()),
    ('Ridge', Ridge()),
    ('Lasso', Lasso()),
    ('ElasticNet', ElasticNet()),
    ('Bayesian Ridge', BayesianRidge()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('AdaBoost', AdaBoostRegressor(n_estimators=100)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators=100)),
    ('K-Nearest Neighbors', KNeighborsRegressor()),
    ('MLP', MLPRegressor()),
    ('Linear SVR', LinearSVR()),
    ('NuSVR', NuSVR()),
    ('Gaussian Process', GaussianProcessRegressor()),
    ('LightGBM', LGBMRegressor(n_estimators=100)),
    ('XGBoost', XGBRegressor(n_estimators=100)),
    ('CatBoost', CatBoostRegressor(iterations=100))
]

# Initialize the VotingRegressor with the list of models
voting_regressor = VotingRegressor(estimators)

# Fit the VotingRegressor on the training data
voting_regressor.fit(X_train, y_train)

# Evaluate the ensemble model
score = voting_regressor.score(X_test, y_test)