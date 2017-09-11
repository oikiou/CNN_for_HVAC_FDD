import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array((5, 6, 7, 8,0,0,9,9))
c = np.array([[[[[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10],],],],])
(row,col,*ff)=c.shape
print(b,c,c.shape,row,col,*ff)

b.shape=2,-1
print(b)

d=b.reshape((-1,2,2))
print(d)

print(np.arange(10), np.linspace(0, 1, 12),np.logspace(0, 2, 20))

def func2(i, j):
    return (i+1)*(j+1)
print(np.fromfunction(func2,(9,9)))

