import numpy
import copy
import pygmo as pg
import numpy as np
import math
def immune_search(A,NC,N):
    E=CloneOperator(A,NC,N)
    print(E)
    for i in range(len(E)):
        ###到这里了简单遗传变异
        pass


def CloneOperator(A,NC,N):
    PC=copy.copy(A)
    if(len(PC)>NC):
        CDA=pg.crowding_distance(PC)
        index=np.argsort(CDA)
        new_PC=np.zeros((NC,A[0].shape[0]))
        for idx,val in enumerate(index[-NC:]):
            new_PC[idx]=PC[val]
    E=[]
    q=np.zeros((NC))
    for i in range(NC):
        q[i]=math.ceil(N*CDA[-i]/sum(CDA[-NC:]))
        E.append(cloneoprter(q[i],new_PC[i]))
    return E
class cloneoprter():
    def __init__(self,q,f):
        self.q=q
        self.f=f




