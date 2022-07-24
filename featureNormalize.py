from numpy import *;

# 归一化
def featureNormalize(data):
    m = len(data);
    aver_ = mean(data, axis=0); # 平均值
    std_  = std(data, axis=0); # 标准差
    


    #归一化
    data = (data - repeat(aver_, m, axis=0))/repeat(std_, m, axis=0);
    # repeat 复制行列 axis为0时复制行 为1时复制列

    return data, aver_, std_;