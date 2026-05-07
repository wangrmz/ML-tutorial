import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 设置中文字体：避免绘图中中文乱码，Heiti TC是macOS下的黑体；关闭unicode减号显示问题。
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 1.生成数据
# 生成数据：产生300个样本，2个簇中心，标准差2。X是特征数组（300×2），y是每个样本的标签（0或1）。
X,y =make_blobs(n_samples=300, centers=2, cluster_std=2)

# 画出原始数据的散点图
# 创建子图：生成2行1列（因为只传了2，默认是nrows=2, ncols=1）的子图网格。
# 返回值解包
# 这行代码同时返回两个对象：
# fig (Figure对象)
# 代表整个图形窗口/画布
# 是最外层的容器，包含所有的子图
# 可以设置整个图形的大小、标题、背景色等
# ax (Axes对象或数组)
# 代表具体的子图（坐标系）
# 用于绘制实际的数据（折线图、散点图等）
# 这里因为有2个子图，所以 ax 是一个包含2个Axes对象的数组
fig,ax = plt.subplots(2, figsize=(8, 8))
# 在第一个子图上画散点图：X轴是第0列，Y轴是第1列，颜色灰色，点大小50，标签"原始数据"。
ax[0].scatter(X[:,0],X[:,1],c="gray",s=50,label="原始数据")
# 设置标题和图例。
ax[0].set_title('原始数据')
ax[0].legend()

# 定义模型并聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 3.获取聚类结果
centers = kmeans.cluster_centers_

# 4.预测，得到所有样本点的标签
y_pred = kmeans.predict(X)

# 5.画出聚类结果
ax[1].scatter(X[:,0],X[:,1],c=y_pred,s=50)
ax[1].scatter(centers[:,0],centers[:,1],c="red",s=200,label="聚类中心")
ax[1].set_title('聚类结果')
ax[1].legend()
plt.show()