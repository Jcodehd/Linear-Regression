from numpy import *;
from computeCost import computeCost;

def gradientDescent(X, Y, theta, iters, alpha):
    m = len(X);
    J = zeros(iters); #损失函数值
    for i in range(iters):
        theta = theta - (alpha/m)*dot((X.T),(dot(X, theta)-Y));
        J[i] = computeCost(X, Y, theta);

    return theta, J;





