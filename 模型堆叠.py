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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


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


def get_stacking_base_datasets(models, X_train_n, y_train_n, X_test_n, n_folds):
    # 生成K-fold交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # 为最终模型准备的训练数据集和测试数据集
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(models.__class__.__name__, '模型开始')

    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        # 切分出训练数据和验证数据
        print(f'\t折数 {folder_counter + 1} 开始')
        X_tr = X_train_n.iloc[train_index]
        y_tr = y_train_n.iloc[train_index]
        X_te = X_train_n.iloc[valid_index]

        # 训练基模型
        models.fit(X_tr, y_tr)

        # 使用基模型对验证集进行预测，并保存结果
        train_fold_pred[valid_index, :] = models.predict(X_te).reshape(-1, 1)

        # 使用基模型对测试集进行预测，并保存结果
        test_pred[:, folder_counter] = models.predict(X_test_n)

    # 对测试集预测结果取平均值
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

    return train_fold_pred, test_pred_mean


# 使用基模型生成训练和测试数据
model_list = [LinearRegression(), RandomForestRegressor(n_estimators=100), LGBMRegressor(n_estimators=100)]

X_train_stack = np.zeros((X_train.shape[0], len(model_list)))
X_test_stack = np.zeros((X_test.shape[0], len(model_list)))

for i, model in enumerate(model_list):
    X_train_pred, X_test_pred = get_stacking_base_datasets(model, X_train, y_train, X_test, 10)
    X_train_stack[:, i] = X_train_pred.ravel()  # 将预测结果转换为一维数组
    X_test_stack[:, i] = X_test_pred.ravel()

# 训练元模型
meta_model_lasso = Lasso()
meta_model_lasso.fit(X_train_stack, y_train)

# 元模型预测
final = meta_model_lasso.predict(X_test_stack)

# 评估性能

results_df = pd.DataFrame(columns=['R^2','RMSE', 'MAE', 'MAPE'])

rmse = sqrt(mean_squared_error(y_test, final))
mae = mean_absolute_error(y_test, final)
r2 = r2_score(y_test, final)
    # 计算平均绝对百分比误差（MAPE）
mape = mean_absolute_percentage_error(y_test, final)
    # 将结果添加到数据框中
results_df = pd.concat(
    [results_df, pd.DataFrame([[r2, rmse, mae,  mape]], columns=['R^2','RMSE', 'MAE', 'MAPE'])],
    ignore_index=True)
# 保存结果为 CSV 文件
results_df.to_csv(r'C:\python\2024\3\3屠宰\4.3\眼肌pH\眼肌pHA组模型堆叠.csv', index=False)
