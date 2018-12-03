"""
Traffic.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from os import listdir
from re import split

from OU import OU
from helper import softmax


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in split(r'(\d+)', string_)]


#



class Traffic():

    def __init__(self, nodes_num, type, capacity):
        self.nodes_num = nodes_num
        self.prev_traffic = None
        self.type = type
        self.capacity = capacity * nodes_num / (nodes_num - 1)
        self.dictionary = {}
        self.dictionary['NORM'] = self.normal_traffic
        self.dictionary['UNI'] = self.uniform_traffic
        self.dictionary['CONTROLLED'] = self.controlled_uniform_traffic
        self.dictionary['EXP'] = self.exp_traffic
        self.dictionary['OU'] = self.ou_traffic
        self.dictionary['STAT'] = self.stat_traffic
        self.dictionary['STATEQ'] = self.stat_eq_traffic
        self.dictionary['FILE'] = self.file_traffic
        self.dictionary['DIR'] = self.dir_traffic
        if self.type.startswith('DIR:'):
            self.dir = sorted(listdir(self.type.split('DIR:')[-1]), key=lambda x: natural_key((x)))
        self.static = None
        self.total_ou = OU(1, self.capacity/2, 0.1, self.capacity/2)
        self.nodes_ou = OU(self.nodes_num**2, 1, 0.1, 1)

    def normal_traffic(self):
        t = np.random.normal(capacity/2, capacity/2)
        return np.asarray(t * softmax(np.random.randn(self.nodes_num, self.nodes_num))).clip(min=0.001)

    def uniform_traffic(self):
        t = np.random.uniform(0, self.capacity*1.25)
        return np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.nodes_num]*2))).clip(min=0.001)

    def controlled_uniform_traffic(self):
        t = np.random.uniform(0, self.capacity*1.25)
        if self.prev_traffic is None:
            self.prev_traffic = np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.nodes_num]*2))).clip(min=0.001)
        dist = [1]
        dist += [0]*(self.nodes_num**2 - 1)
        ch = np.random.choice(dist, [self.nodes_num]*2)

        tt = np.multiply(self.prev_traffic, 1 - ch)

        nt = np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.nodes_num]*2))).clip(min=0.001)
        nt = np.multiply(nt, ch)

        self.prev_traffic = tt + nt

        return self.prev_traffic

    # xxxxxxx
    # xxx
    # 指数分布
    def exp_traffic(self):
        a = np.random.exponential(size=self.nodes_num)
        b = np.random.exponential(size=self.nodes_num)

        # https://blog.csdn.net/u011599639/article/details/77926402
        # 计算向量 a,b 外积，[a,b]
        T = np.outer(a, b)
        # 对角线 填 -1
        np.fill_diagonal(T, -1)

        T[T!=-1] = np.asarray(np.random.exponential()*T[T!=-1]/np.average(T[T!=-1])).clip(min=0.001)

        return T

    def stat_traffic(self):
        if self.static is None:
            string = self.type.split('STAT:')[-1]
            v = np.asarray(tuple(float(x) for x in string.split(',')[:self.nodes_num**2]))
            M = np.split(v, self.nodes_num)
            self.static = np.vstack(M)
        return self.static

    def stat_eq_traffic(self):
        if self.static is None:
            value = float(self.type.split('STATEQ:')[-1])
            self.static = np.full([self.nodes_num]*2, value, dtype=float)
        return self.static

    def ou_traffic(self):
        t = self.total_ou.evolve()[0]
        nt = t * softmax(self.nodes_ou.evolve())
        i = np.split(nt, self.nodes_num)
        return np.vstack(i).clip(min=0.001)

    def file_traffic(self):
        if self.static is None:
            fname = 'traffic/' + self.type.split('FILE:')[-1]
            v = np.loadtxt(fname, delimiter=',')
            self.static = np.split(v, self.nodes_num)
        return self.static

    def dir_traffic(self):
        while len(self.dir) > 0:
            tm = self.dir.pop(0)
            if not tm.endswith('.txt'):
                continue
            fname = self.type.split('DIR:')[-1] + '/' + tm
            v = np.loadtxt(fname, delimiter=',')
            return np.split(v, self.nodes_num)
        return False


    def generate(self):
        return self.dictionary[self.type.split(":")[0]]()
    # 这样 dictionary [14,14] 代表什么意思呢？？？
    #[[-1.00000000e+00   4.82597027e-01   1.64885219e-01   2.39937374e-01
    #  4.24195039e-01   3.90513477e-01   1.73313504e-01   2.39531467e-01
    #  8.81383591e-01   2.35750495e-01   9.86736084e-01   1.10174305e+00
    #  2.91715890e-02   1.24369249e-01]
    # [1.24568554e-01 - 1.00000000e+00   4.58794226e-02   6.67627350e-02
    #  1.18032554e-01   1.08660636e-01   4.82245987e-02   6.66497910e-02
    #  2.45245574e-01   6.55977330e-02   2.74559976e-01   3.06560741e-01
    #  8.11701418e-03   3.46058268e-02]
    # [4.41894837e-01   4.76355699e-01 - 1.00000000e+00   2.36834313e-01
    #  4.18709012e-01   3.85463046e-01   1.71072076e-01   2.36433655e-01
    #  8.69984838e-01   2.32701582e-01   9.73974829e-01   1.08749443e+00
    #  2.87943189e-02   1.22760807e-01]
    # [3.99306289e-01   4.30445913e-01   1.47067148e-01 - 1.00000000e+00
    #  3.78355046e-01   3.48313231e-01   1.54584644e-01   2.13646863e-01
    #  7.86138212e-01   2.10274476e-01   8.80105948e-01   9.82684859e-01
    #  2.60192056e-02   1.10929475e-01]
    # [8.86430104e-01   9.55557740e-01   3.26478072e-01   4.75083769e-01
    #  - 1.00000000e+00   7.73229329e-01   3.43166350e-01   4.74280060e-01
    #  1.74516805e+00   4.66793613e-01   1.95376940e+00   2.18148691e+00
    #  5.77606908e-02   2.46255140e-01]
    # [6.17537874e-01   6.65696136e-01   2.27443285e-01   3.30970507e-01
    #  5.85136215e-01 - 1.00000000e+00   2.39069293e-01   3.30410597e-01
    #  1.21578381e+00   3.25195110e-01   1.36110743e+00   1.51974846e+00
    #  4.02393985e-02   1.71555405e-01]
    # [2.31567998e-01   2.49626667e-01   8.52880256e-02   1.24109275e-01
    #  2.19417832e-01   2.01995810e-01 - 1.00000000e+00   1.23899316e-01
    #  4.55901789e-01   1.21943582e-01   5.10396098e-01   5.69884250e-01
    #  1.50892072e-02   6.43308585e-02]
    # [1.68174721e+00   1.81289710e+00   6.19398623e-01   9.01335366e-01
    #  1.59350744e+00   1.46698116e+00   6.51059851e-01 - 1.00000000e+00
    #  3.31095647e+00   8.85607170e-01   3.70671778e+00   4.13874653e+00
    #  1.09584366e-01   4.67198589e-01]
    # [1.81548791e+00   1.95706747e+00   6.68656207e-01   9.73013928e-01
    #  1.72023088e+00   1.58364262e+00   7.02835289e-01   9.71367859e-01
    #  - 1.00000000e+00   9.56034948e-01   4.00149396e+00   4.46787974e+00
    #  1.18299046e-01   5.04352487e-01]
    # [3.39180759e+00   3.65631535e+00   1.24922518e+00   1.81784523e+00
    #  3.21384249e+00   2.95865979e+00   1.31308067e+00   1.81476994e+00
    #  6.67765480e+00 - 1.00000000e+00   7.47584027e+00   8.34717123e+00
    #  2.21013647e-01   9.42262733e-01]
    # [3.80108803e+00   4.09751324e+00   1.39996587e+00   2.03719980e+00
    #  3.60164835e+00   3.31567343e+00   1.47152663e+00   2.03375342e+00
    #  7.48342971e+00   2.00165090e+00 - 1.00000000e+00   9.35440226e+00
    #  2.47682778e-01   1.05596308e+00]
    # [1.92350407e-01   2.07350720e-01   7.08439274e-02   1.03090538e-01
    #  1.82257953e-01   1.67786468e-01   7.44651911e-02   1.02916137e-01
    #  3.78691769e-01   1.01291620e-01   4.23957102e-01 - 1.00000000e+00
    #  1.25337489e-02   5.34359969e-02]
    # [3.30800641e-01   3.56597899e-01   1.21836065e-01   1.77293184e-01
    #  3.13443828e-01   2.88556037e-01   1.28063846e-01   1.76993253e-01
    #  6.51267039e-01   1.74199439e-01   7.29113514e-01   8.14093818e-01
    #  - 1.00000000e+00   9.18982306e-02]
    # [6.67661381e-01   7.19728489e-01   2.45904104e-01   3.57834288e-01
    #  6.32629787e-01   5.82398272e-01   2.58473757e-01   3.57228932e-01
    #  1.31446496e+00   3.51590122e-01   1.47158401e+00   1.64310142e+00
    #  4.35054974e-02 - 1.00000000e+00]]

