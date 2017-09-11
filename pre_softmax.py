import numpy as np
import pandas as pd
import tensorflow as tf

#input
row_data = np.loadtxt("7.csv", delimiter=",", skiprows=1)

#y分列 softmax
def sp_softmax(input,n):
    result=[]
    for i in range(len(input)):
        temp=np.zeros(n)
        temp[int(input[i])]=1
        result.append(temp)
    return result

#7分类问题
row_data = np.array(row_data)
row_y = row_data[:,-1]
row_y = sp_softmax(row_y, 7)
print(row_y)

row_data = np.column_stack([row_data[:,:-1], row_y])

#标准化
row_data=pd.DataFrame(row_data)
standed_data=np.nan_to_num(row_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))))

#随机分 7成train 3成test
def sprite(standed_data, n=0.7):
    index = np.random.choice(len(standed_data), (int(len(standed_data) * n)), replace=False)
    print(len(index),index)
    train_data = [standed_data[i] for i in index]
    test_data_list = np.array(list(set(range(len(standed_data))) - set(index)))
    print(len(test_data_list),test_data_list)
    test_data = [standed_data[i] for i in test_data_list]
    return [train_data, test_data]

[train_data, test_data] = sprite(standed_data)
train_data = np.array(train_data)
test_data = np.array(test_data)

#分x，y
train_x = train_data[:,0:-7]
train_y = train_data[:,-7:]
test_x = test_data[:,0:-7]
test_y = test_data[:,-7:]
print(train_y,test_y)

#输出
np.savetxt('new_train_x.csv', train_x, delimiter = ',', fmt="%.3f")
np.savetxt('new_train_y.csv', train_y, delimiter = ',', fmt="%.3f")
np.savetxt('new_test_x.csv', test_x, delimiter = ',', fmt="%.3f")
np.savetxt('new_test_y.csv', test_y, delimiter = ',', fmt="%.3f")
