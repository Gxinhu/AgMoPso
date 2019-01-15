import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg
ObjV =np.array([[9,1],[7,2],[5,4],[4,5],[3,6],[2,7],[1,9],[10,3],[8,5],
          [7,6],[5,7],[4,8],[3,9],[10,5],[9,6],[8,7],[7,9],[10,6],[9,7],[8,9]])

zdt1 = np.loadtxt('ZDT1.txt')
plt.scatter(ObjV[:, 0], ObjV[:, 1], marker='o', color='green', s=40)
CDA=pg.crowding_distance(ObjV)
print(CDA)
CDA=sorted(CDA)
print(CDA)
plt.show()