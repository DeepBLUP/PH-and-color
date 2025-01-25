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
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import shap


# 调整路径为您的文件实际路径
#filepath = r'C:\python\2024\3\3屠宰\4.3\眼肌pH\pH无固定效应.csv'
filepath = r'C:\python\2024\3\3屠宰\4.3\预处理后的数据.csv'
# 尝试使用utf - 8编码读取
try:
    df = pd.read_csv(filepath, encoding='utf - 8')
except UnicodeDecodeError:
    # 如果utf - 8失败，尝试使用gbk编码读取
    df = pd.read_csv(filepath, encoding='gbk')

df_cleaned = df.dropna(axis = 0)
# 确定哪些列是字符串类型的性状
# categorical_columns = ['公司', '采样时间','猪种']  # 用实际的列名替换'column1', 'column2',...

# 将字符串性状转换为独热编码
# df_cleaned = pd.get_dummies(df_cleaned, columns = categorical_columns)

X = df_cleaned.drop(columns=['pH of eye muscle at 48h', 'pH of psoas at 48h', 'eye muscle color L value at 48h', 'eye muscle color a value at 48h', 'eye muscle color b value at 48h'
                             , 'psoas color L value at 48h', 'psoas color a value at 48h', 'psoas color b value at 48h'])

y = df_cleaned['pH of eye muscle at 48h']
# y = df_cleaned['BpH48h']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 32)

# 初始化基本回归模型
base_estimators = [
    #('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators = 100)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators = 100)),
    #('SVR', SVR()),
    #('Ridge', Ridge()),
    #('Lasso', Lasso()),
    #('ElasticNet', ElasticNet()),
    #('Bayesian Ridge', BayesianRidge()),
    #('Decision Tree', DecisionTreeRegressor()),
    ('AdaBoost', AdaBoostRegressor(n_estimators = 100)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators = 100)),
    #('K - Nearest Neighbors', KNeighborsRegressor()),
    #('MLP', MLPRegressor()),
    #('Linear SVR', LinearSVR()),
    #('NuSVR', NuSVR()),
    #('Gaussian Process', GaussianProcessRegressor()),
    ('LightGBM', LGBMRegressor(n_estimators = 100)),
    ('XGBoost', XGBRegressor(n_estimators = 100)),
    ('CatBoost', CatBoostRegressor(iterations = 100))
]

# 初始化BaggingRegressor
bagging_regressor = BaggingRegressor(base_estimator = None, n_estimators = 10, random_state = 42)

# 训练模型
for name, model in base_estimators:
    bagging_regressor.base_estimator = model
    bagging_regressor.fit(X_train, y_train)

# 预测
y_pred = bagging_regressor.predict(X_test)

# 评估性能
results_df = pd.DataFrame(columns=['R^2', 'RMSE', 'MAE', 'MAPE'])
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 计算平均绝对百分比误差（MAPE）
mape = mean_absolute_percentage_error(y_test, y_pred)
# 将结果添加到数据框中
results_df = pd.concat(
    [results_df, pd.DataFrame([[r2, rmse, mae, mape]], columns=['R^2', 'RMSE', 'MAE', 'MAPE'])],
    ignore_index = True)


import shap
import matplotlib.pyplot as plt

# 创建一个新的 BaggingRegressor 实例并使用最后一个训练过的基模型进行训练
final_bagging_regressor = BaggingRegressor(base_estimator=bagging_regressor.base_estimator, n_estimators=10, random_state=42)
final_bagging_regressor.fit(X_train, y_train)

# 确保 base_estimator 是 CatBoostRegressor 并已拟合
base_estimator = final_bagging_regressor.base_estimator
if isinstance(base_estimator, CatBoostRegressor):
    if not base_estimator.is_fitted():
        base_estimator.fit(X_train, y_train)

# 使用 TreeExplainer 来解释模型，因为 base_estimator 是 CatBoostRegressor
explainer = shap.TreeExplainer(base_estimator)
shap_values = explainer.shap_values(X_test)


shap.summary_plot(shap_values, X_test)

# 尝试使用 shap 自带的保存功能
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(r'C:\python\2024\3\3屠宰\4.3\眼肌pH\shap_summary_plot.png')