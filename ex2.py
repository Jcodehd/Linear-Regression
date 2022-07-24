from numpy import *;
from matplotlib.pyplot import *;
from pandas import *;
from GradientDescent import gradientDescent;
from computeCost import computeCost;
from featureNormalize import featureNormalize;
from NormalEquations import normalEquations;

# 读取数据 第一列房子大小、第二列卧室数量、第三列房子价格
data = mat(read_table('Machine Learning\Linear Regression\ex1data2.txt',sep = ',', header = None));

# 取出X、Y
X = data[:,:2];
Y = data[:,2:];

# 取出样本数
row = len(X);

# 将X中的样本特征归一化

X, aver_, std_= featureNormalize(X);

# 添加一列X0
X = hstack((ones((row,1), dtype=int), X));

# 设置迭代数、 学习率、 theta;
iters = 8500; # 迭代次数
alpha = 0.01; # 学习率
theta = zeros((X.shape[1], 1));

# 梯度下降
theta, J = gradientDescent(X, Y, theta, iters, alpha);

# 绘制损失函数迭代图
plot(range(iters), J);
show();

# 预测面积为1650,房间数为3的房间价格
t = mat([1650,3]);
t = (t - aver_)/std_;
t = c_[ones((1,1)), t];
predict = dot(t, theta);
print("当房子面积为1650，房间数为3时，房子价格为(梯度下降)", predict[0,0]);

# 采用正规方程预测
theta_ = normalEquations(hstack((ones((row,1), dtype=int), data[:,:2])), data[:,2:]);
predict_ = dot(mat([1,1650,3]), theta_);

print("当房子面积为1650，房间数为3时，房子价格为(正规方程)", predict_[0,0]);





