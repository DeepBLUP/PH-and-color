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



# 初始化 BaggingRegressor
bagging_regressor = BaggingRegressor(base_estimator=None, n_estimators=10, random_state=42)

# 训练模型
for name, model in base_estimators:
    bagging_regressor.base_estimator = model
    bagging_regressor.fit(X_train, y_train)

# 预测
y_pred = bagging_regressor.predict(X_test)
# 评估性能
results_df = pd.DataFrame(columns=[ 'R^2','RMSE', 'MAE', 'MAPE'])
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
    # 计算平均绝对百分比误差（MAPE）
mape = mean_absolute_percentage_error(y_test, y_pred)
    # 将结果添加到数据框中
results_df = pd.concat(
    [results_df, pd.DataFrame([[r2,rmse, mae,  mape]], columns=['R^2','RMSE', 'MAE', 'MAPE'])],
    ignore_index=True)
# 保存结果为 CSV 文件
results_df.to_csv(r'C:\python\2024\3\3屠宰\4.3\眼肌pH\眼肌pHA组Bagging1.csv', index=False)


