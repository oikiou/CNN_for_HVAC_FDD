import numpy as np
import pandas as pd

'''
a = np.reshape(range(20),(-1,4))
print(a)

a=pd.DataFrame(a)
print(a)

b=a.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(b)
'''

row_data = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
row_data=pd.DataFrame(row_data)
standed_data=np.nan_to_num(row_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))))
np.savetxt('train_x.csv', standed_data, delimiter = ',', fmt="%.3f")