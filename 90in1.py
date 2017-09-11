import numpy as np

row_data = np.loadtxt(open("test_x_bems.csv","rb"),delimiter=",",skiprows=1)

output = np.reshape(row_data,(-1,144*108))

np.savetxt('cnn_test_x_bems.csv', output, delimiter = ',', fmt="%.4f")