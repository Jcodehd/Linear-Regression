from matplotlib.pyplot import plot, show
from numpy import *;
import pandas as pd;
from computeCost import computeCost;
from matplotlib import *;
from GradientDescent import gradientDescent;

# 读取数据 第一列城市人口、第二列城市利润
data = array(pd.read_table('Machine Learning\Linear Regression\ex1data1.txt',sep = ',', header = None));

# 取出X、Y
X = data[:,0:1];
Y = data[:,1:];

# 获取数据的样本数
row = len(X); # 样本数

# 添加x0
X = hstack((ones((row, 1), dtype=int), X));


# 设置学习率、迭代次数、theta初始值全为0;
iters = 1500; # 迭代次数
alpha = 0.01; # 学习率
theta = zeros((X.shape[1], 1));

# 梯度下降
theta, J= gradientDescent(X, Y, theta, iters, alpha);
print("theta", theta[0,0], theta[1,0]);

# 绘制损失函数迭代图
plot(range(iters), J);
show();

# 分别预测35000、70000人口时的利润
predict1 = dot(mat([1,3.5]), theta)*10000;
predict2 = dot(mat([1,7]), theta)*10000;
print("当城市人口为35000时，利润为", predict1[0,0]);
print("当城市人口为70000时，利润为", predict2[0,0]);