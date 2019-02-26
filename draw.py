import numpy as np
import matplotlib.pyplot as plt

A=np.loadtxt("AgMOPSO_CEC2009_UF3_2_T1")
x = []
y = []
z = []
for i in range(len(A)):
    x.append(A[i][0])
    y.append(A[i][1])
zdt2 = np.loadtxt('UF3.pf')
plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
plt.scatter(x, y, marker='o', color='red', s=40)
plt.xlabel('f1')
plt.ylabel('f2')
plt.show()
plt.close('all')