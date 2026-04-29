from sklearn.datasets import make_classification # 自动生成分类数据集
from sklearn.model_selection import train_test_split  # 训练测试
from sklearn.linear_model import LogisticRegression # 逻辑回归分析模型
from sklearn.metrics import classification_report  # 生成分类评估报告

# 1.生成一个二分类数据集（1000 *20 的矩阵）分类数为2，二分类
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# print(X.shape)
# print(y.shape)

# 2.划分训练集和测试集
# 测试比例0.3
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

#训练一个逻辑回归模型
# 3.定义个逻辑分类模型（逻辑回归）
model = LogisticRegression()
# 4.模型训练
model.fit(x_train, y_train)

# 5.预测(测试)
y_pred = model.predict(x_test)

# 6.生成分类报告
report = classification_report(y_test, y_pred)
print(report)
'''
                   precision    recall  f1-score   support

           0       0.83      0.88      0.85       151
           1       0.87      0.81      0.84       149

    accuracy                           0.85       300
   macro avg       0.85      0.85      0.85       300
weighted avg       0.85      0.85      0.85       300
'''
# 获取预测正类的概率值(0,1)
y_pred_proba= model.predict_proba(x_test)[:,1]
from sklearn.metrics import roc_auc_score
# 计算 AUC的值
roc_aoc= roc_auc_score(y_test, y_pred_proba)
# 越接近1越准确
print(roc_aoc) # 0.8968398595493133