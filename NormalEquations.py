from numpy import *;
from numpy.linalg import *;
# 正规方程

def normalEquations(data, Y):
    theta = zeros((data.shape[1],1));

    theta = dot(dot(inv(dot(data.T, data)), data.T), Y);

    return theta;
