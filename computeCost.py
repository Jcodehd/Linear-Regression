# 计算损失函数
import numpy as np;
def computeCost(X, Y, theta):
    m = len(X); #计算样本数
    J =  sum(np.array((np.dot(X, theta) - Y))**2)/(2*m);
    return J;

    