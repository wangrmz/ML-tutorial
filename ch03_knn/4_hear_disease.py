import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1.加载数据集
heart_disease_data = pd.read_csv('../data/heart_disease.csv')

# 处理缺失值
heart_disease_data.dropna(inplace=True)

print(heart_disease_data.head())

# 2、数据集的划分
# 划分特征和标签
X = heart_disease_data.drop(labels='是否患有心脏病', axis=1)
y = heart_disease_data['是否患有心脏病']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 适用场景：归一化，标准化
# 数值特征标准化，类别特征

# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]

# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]
# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        # 对数值型特征进行标准化
        ("num", StandardScaler(), numerical_features),
        # 对类别型特征进行独热编码，使用drop="first"避免多重共线性
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        # 二元特征不进行处理
        ("binary", "passthrough", binary_features),
    ]
)
# 执行特征转换
x_train = preprocessor.fit_transform(X_train) # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(X_test)  # 使用训练集计算的信息对测试集进行转换

# 4.定义模型
knn = KNeighborsClassifier(n_neighbors=3)

# 5.模型训练
knn.fit(x_train, y_train)

# 6.模型评估,计算准确率
score = knn.score(x_test, y_test)
print(score) # 0.9253246753246753

# 7.保存模型
# 可以使用Python的joblib库保存训练好的模型：
# 保存在当前路径
# joblib.dump(knn, "knn_heart_disease")
#
# # 加载先前保存的模型：
# # 加载模型
# knn_loaded = joblib.load("knn_heart_disease")
# # 预测
# y_pred = knn_loaded.predict(x_test[10:11])
# # 打印真实值与预测值
# print(y_test.iloc[10], y_pred)

# 创建KNN分类器
knn = KNeighborsClassifier()
# 网格搜索参数，K值设置为1到10
# param_grid = {"n_neighbors": list(range(1, 10))}
param_grid = {"n_neighbors": list(range(1, 10)),"weights": ["uniform", "distance"]}
# GridSearchCV(estimator=模型, param_grid=网格搜索参数, cv=k折交叉验证)
grid_search_cv = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10)

# 模型训练
grid_search_cv.fit(x_train, y_train)
results = pd.DataFrame(grid_search_cv.cv_results_).to_string()
print(results) # 所有交叉验证结果
# 直接获取最佳模型和最佳得分
print(grid_search_cv.best_estimator_) # 最佳模型
print(grid_search_cv.best_score_) # 最佳得分

# 使用最佳模型进行测试评估
knn = grid_search_cv.best_estimator_
# print(knn.score(x_test, y_test)) # 0.9805194805194806
print(knn.score(x_test, y_test)) # 加入权重





