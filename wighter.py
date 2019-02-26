import numpy as np

class Mean_vector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=13, m=3):
        self.H = H
        self.m = m
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
    #生成权均匀向量
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws

    def save_mv_to_file(self, mv, name='out.csv'):
    #保存为csv
        f = np.array(mv, dtype=np.float64)
        np.savetxt(fname=name, X=f)

    def test(self):
    #测试
        return np.array(self.get_mean_vectors())
        # self.save_mv_to_file(m_v, 'test.csv')
# mv = Mean_vector(13, 3)
# mv.test()
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# data = np.loadtxt('test.csv')
# print(data.shape[0])
#
# fig = plt.figure()
# ax = Axes3D(fig)
#
# x, y, z = data[:, 0], data[:, 1], data[:, 2]
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# ax.scatter(x, y, z, marker='.', s=50, label='',color='r')
#
# VecStart_x = np.zeros(data.shape[0])
# VecStart_y = np.zeros(data.shape[0])
# VecStart_z = np.zeros(data.shape[0])
# VecEnd_x = data[:, 0]
# VecEnd_y = data[:, 1]
# VecEnd_z = data[:, 2]
#
# for i in range(VecStart_x.shape[0]):
#     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]], zs=[VecStart_z[i], VecEnd_z[i]])
#
# plt.show()

