"""
Environment.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from scipy import stats
import subprocess
import networkx as nx

from helper import pretty, softmax
from Traffic import Traffic


OMTRAFFIC = 'Traffic.txt'
OMBALANCING = 'Balancing.txt'
OMROUTING = 'Routing.txt'
OMDELAY = 'Delay.txt'

TRAFFICLOG = 'TrafficLog.csv'
BALANCINGLOG = 'BalancingLog.csv'
REWARDLOG = 'rewardLog.csv'
WHOLELOG = 'Log.csv'
OMLOG = 'omnetLog.csv'


# FROM MATRIX
def matrix_to_rl(matrix):
    return matrix[(matrix!=-1)]

matrix_to_log_v = matrix_to_rl

def matrix_to_omnet_v(matrix):
    return matrix.flatten()

def vector_to_file(vector, file_name, action):
    string = ','.join(pretty(_) for _ in vector)
    with open(file_name, action) as file:
        return file.write(string + '\n')


# FROM FILE
def file_to_csv(file_name):
    # reads file, outputs csv
    with open(file_name, 'r') as file:
        return file.readline().strip().strip(',')

def csv_to_matrix(string, nodes_num):
    # reads text, outputs matrix
    v = np.asarray(tuple(float(x) for x in string.split(',')[:nodes_num**2]))
    M = np.split(v, nodes_num)
    return np.vstack(M)

def csv_to_lost(string):
    return float(string.split(',')[-1])


# FROM RL
def rl_to_matrix(vector, nodes_num):
    M = np.split(vector, nodes_num)
    for _ in range(nodes_num):
        M[_] = np.insert(M[_], _, -1)
    return np.vstack(M)


# TO RL
# STATUM = 'T'  :  每个节点之间的 traffic
# STATUM = 'RT' :  每个节点之间的 balancing， 每个节点之间的 traffic
# 返回state，这里有两种方式

def rl_state(env):
    if env.STATUM == 'RT':
        return np.concatenate((matrix_to_rl(env.env_B), matrix_to_rl(env.env_T)))
    elif env.STATUM == 'T':
        return matrix_to_rl(env.env_T)

# 计算reward，主要是通过delay
def rl_reward(env):

    delay = np.asarray(env.env_D)
    # np.inf 无穷大
    # 这里是做一个mask，将 delay 里 值为np.inf 的位置 置为 1 ，其余为 0
    mask = delay == np.inf
    # ~ 是取反操作， len(delay)应该是n^2
    # np.max(delay[~mask]) 取出所有的真正delay值，然后取最大的（短板效应，最慢的到了，才完全到）
    delay[mask] = len(delay)*np.max(delay[~mask])

    # PRAEMIUM = AVG
    if env.PRAEMIUM == 'AVG':
        reward = -np.mean(matrix_to_rl(delay))
    elif env.PRAEMIUM == 'MAX':
        reward = -np.max(matrix_to_rl(delay))
    elif env.PRAEMIUM == 'AXM':
        reward = -(np.mean(matrix_to_rl(delay)) + np.max(matrix_to_rl(delay)))/2
    elif env.PRAEMIUM == 'GEO':
        reward = -stats.gmean(matrix_to_rl(delay))
    elif env.PRAEMIUM == 'LOST':
        reward = -env.env_L
    return reward


# WRAPPER ITSELF
def omnet_wrapper(env):
    if env.ENV == 'label':
        sim = 'router'
    elif env.ENV == 'balancing':
        sim = 'balancer'

    prefix = ''
    if env.CLUSTER == 'arvei':
        prefix = '/scratch/nas/1/giorgio/rlnet/'

    simexe = prefix + 'omnet/' + sim + '/networkRL'
    simfolder = prefix + 'omnet/' + sim + '/'
    simini = prefix + 'omnet/' + sim + '/' + 'omnetpp.ini'

    try:
        omnet_output = subprocess.check_output([simexe, '-n', simfolder, simini, env.folder + 'folder.ini']).decode()
    except Exception as e:
        omnet_output = e.stdout.decode()

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [_.strip() for _ in omnet_output.split('\n') if _ is not '']
        omnet_output = ','.join(o_u_l[4:])
    else:
        omnet_output = 'ok'

    vector_to_file([omnet_output], env.folder + OMLOG, 'a')


def ned_to_capacity(env):
    if env.ENV == 'label':
        sim = 'router'
    elif env.ENV == 'balancing':
        sim = 'balancer'
    NED = 'omnet/' + sim + '/NetworkAll.ned'

    capacity = 0

    with open(NED) as nedfile:
        for line in nedfile:
            if "SlowChannel" in line and "<-->" in line:
                capacity += 3
            elif "MediumChannel" in line and "<-->" in line:
                capacity += 5
            elif "FastChannel" in line and "<-->" in line:
                capacity += 10
            elif "Channel" in line and "<-->" in line:
                capacity += 10

    return capacity or None


# balancing environment
class OmnetBalancerEnv():

    def __init__(self, DDPG_config, folder):
        self.ENV = 'balancing'
        self.ROUTING = 'Balancer'

        self.folder = folder

        self.ACTIVE_NODES = DDPG_config['ACTIVE_NODES']

        self.ACTUM = DDPG_config['ACTUM']
        self.a_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES     # routing table minus diagonal

        self.s_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES    # traffic minus diagonal

        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        if 'MAX_DELTA' in DDPG_config.keys():
            self.MAX_DELTA = DDPG_config['MAX_DELTA']

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        self.TRAFFIC = DDPG_config['TRAFFIC']
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity)

        self.CLUSTER = DDPG_config['CLUSTER'] if 'CLUSTER' in DDPG_config.keys() else False

        self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # traffic
        self.env_B = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # balancing
        self.env_D = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # delay
        self.env_L = -1.0  # lost packets

        self.counter = 0


    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        np.fill_diagonal(self.env_T, -1)

    def upd_env_B(self, matrix):
        self.env_B = np.asarray(matrix)
        np.fill_diagonal(self.env_B, -1)

    def upd_env_D(self, matrix):
        self.env_D = np.asarray(matrix)
        np.fill_diagonal(self.env_D, -1)

    def upd_env_L(self, number):
        self.env_L = number


    def logheader(self):
        nice_matrix = np.chararray([self.ACTIVE_NODES]*2, itemsize=20)
        for i in range(self.ACTIVE_NODES):
            for j in range(self.ACTIVE_NODES):
                nice_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix!=b'_')])
        th = ['t' + _.decode('ascii') for _ in nice_list]
        rh = ['r' + _.decode('ascii') for _ in nice_list]
        dh = ['d' + _.decode('ascii') for _ in nice_list]
        if self.STATUM == 'T':
            sh = ['s' + _.decode('ascii') for _ in nice_list]
        elif self.STATUM == 'RT':
            sh = ['sr' + _.decode('ascii') for _ in nice_list] + ['st' + _.decode('ascii') for _ in nice_list]
        ah = ['a' + _.decode('ascii') for _ in nice_list]
        header = ['counter'] + th + rh + dh + ['lost'] + sh + ah + ['reward']
        vector_to_file(header, self.folder + WHOLELOG, 'w')


    def render(self):
        return True


    def reset(self):
        if self.counter != 0:
            return None

        self.logheader()

        # balancing
        self.upd_env_B(np.full([self.ACTIVE_NODES]*2, 0.50, dtype=float))
        if self.ACTUM == 'DELTA':
            vector_to_file(matrix_to_omnet_v(self.env_B), self.folder + OMBALANCING, 'w')

        # traffic
        self.upd_env_T(self.tgen.generate())

        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        return rl_state(self)


    def step(self, action):
        self.counter += 1

        # define action: NEW or DELTA
        if self.ACTUM == 'NEW':
            # bound the action
            self.upd_env_B(rl_to_matrix(np.clip(action, 0, 1), self.ACTIVE_NODES))
        if self.ACTUM == 'DELTA':
            # bound the action
            self.upd_env_B(rl_to_matrix(np.clip(action * self.MAX_DELTA + matrix_to_rl(self.env_B), 0, 1), self.ACTIVE_NODES))

        # write to file input for Omnet: Balancing
        vector_to_file(matrix_to_omnet_v(self.env_B), self.folder + OMBALANCING, 'w')

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        om_output = file_to_csv(self.folder + OMDELAY)
        self.upd_env_D(csv_to_matrix(om_output, self.ACTIVE_NODES))
        self.upd_env_L(csv_to_lost(om_output))

        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)
        log = np.concatenate(([self.counter], matrix_to_log_v(self.env_T), matrix_to_log_v(self.env_B), matrix_to_log_v(self.env_D), [self.env_L], cur_state, action, [-reward]))
        vector_to_file(log, self.folder + WHOLELOG, 'a')

        # generate traffic for next iteration
        self.upd_env_T(self.tgen.generate())
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        return new_state, reward, 0


    def end(self):
        return


# label environment
class OmnetLinkweightEnv():

    def __init__(self, DDPG_config, folder):
        self.ENV = 'label'
        self.ROUTING = 'Linkweight'

        self.folder = folder

        # nodes = 14
        self.ACTIVE_NODES = DDPG_config['ACTIVE_NODES']

        self.ACTUM = DDPG_config['ACTUM']

        # 利用 networkX 创建 网络拓扑图 graph
        topology = 'omnet/router/NetworkAll.matrix'
        self.graph = nx.Graph(np.loadtxt(topology, dtype=int))

        # 这里可以画出来 graph 的拓扑
        import matplotlib.pyplot as plt
        nx.draw(self.graph)
        plt.show()


        if self.ACTIVE_NODES != self.graph.number_of_nodes():
            return False
        ports = 'omnet/router/NetworkAll.ports'

        # self.ports
        # [[-1  0  1  2 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1]
        # [0 - 1  1 - 1 - 1 - 1 - 1  2 - 1 - 1 - 1 - 1 - 1 - 1]
        # [0  1 - 1 - 1 - 1  2 - 1 - 1 - 1 - 1 - 1 - 1 - 1 - 1]
        # [0 - 1 - 1 - 1  1 - 1 - 1 - 1  2 - 1 - 1 - 1 - 1 - 1]
        # [-1 - 1 - 1  0 - 1  1  2 - 1 - 1 - 1 - 1 - 1 - 1 - 1]
        # [-1 - 1  0 - 1  1 - 1 - 1 - 1 - 1 - 1  2 - 1  3 - 1]
        # [-1 - 1 - 1 - 1  0 - 1 - 1  1 - 1 - 1 - 1 - 1 - 1 - 1]
        # [-1  0 - 1 - 1 - 1 - 1  1 - 1 - 1  2 - 1 - 1 - 1 - 1]
        # [-1 - 1 - 1  0 - 1 - 1 - 1 - 1 - 1 - 1 - 1  1 - 1  2]
        # [-1 - 1 - 1 - 1 - 1 - 1 - 1  0 - 1 - 1  1  2 - 1  3]
        # [-1 - 1 - 1 - 1 - 1  0 - 1 - 1 - 1  1 - 1 - 1 - 1 - 1]
        # [-1 - 1 - 1 - 1 - 1 - 1 - 1 - 1  0  1 - 1 - 1  2 - 1]
        # [-1 - 1 - 1 - 1 - 1  0 - 1 - 1 - 1 - 1 - 1  1 - 1  2]
        # [-1 - 1 - 1 - 1 - 1 - 1 - 1 - 1  0  1 - 1 - 1  2 - 1]]
        self.ports = np.loadtxt(ports, dtype=int)

        # actions 是所有的 edges
        self.a_dim = self.graph.number_of_edges()
        # state 维度 n*(n-1) n是节点个数，很显然，任意两点之前组队，一共有n*(n-1) = n^2-n
        self.s_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES    # traffic minus diagonal

        # ？？？
        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        # ？？？ traffic
        self.TRAFFIC = DDPG_config['TRAFFIC']
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity)

        self.CLUSTER = DDPG_config['CLUSTER'] if 'CLUSTER' in DDPG_config.keys() else False

        # 填充 np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)
        # shape = [ACTIVE_NODES, ACTIVE_NODES] !!! 因为numpy中矩阵乘是这样
        self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # traffic
        self.env_W = np.full([self.a_dim], -1.0, dtype=float)           # weights
        self.env_R = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)    # routing
        self.env_Rn = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)   # routing (nodes)
        self.env_D = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # delay
        self.env_L = -1.0  # lost packets

        self.counter = 0

    #将对角线上的填成-1，节点自身到自身
    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        np.fill_diagonal(self.env_T, -1)

    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    # 根据权重，重新计算route
    def upd_env_R(self):
        weights = {}

        for e, w in zip(self.graph.edges(), self.env_W):
            weights[e] = w

        nx.set_edge_attributes(self.graph, 'weight', weights)

        routing_nodes = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)
        routing_ports = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)


        # 计算所有点之间的最短路径（带权），all_shortest：
        # {0: {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 7: [0, 1, 7], 5: [0, 2, 5], 4: [0, 3, 4], 8: [0, 3, 8],
        #      6: [0, 1, 7, 6], 9: [0, 1, 7, 9], 10: [0, 2, 5, 10], 12: [0, 2, 5, 12], 11: [0, 3, 8, 11],
        #      13: [0, 3, 8, 13]},
        #  1: {1: [1], 0: [1, 0], 2: [1, 2], 7: [1, 7], 3: [1, 0, 3], 5: [1, 2, 5], 6: [1, 7, 6], 9: [1, 7, 9],
        #      4: [1, 0, 3, 4], 8: [1, 0, 3, 8], 10: [1, 2, 5, 10], 12: [1, 2, 5, 12], 11: [1, 7, 9, 11],
        #      13: [1, 7, 9, 13]},
        #  2: {2: [2], 0: [2, 0], 1: [2, 1], 5: [2, 5], 3: [2, 0, 3], 7: [2, 1, 7], 4: [2, 5, 4], 10: [2, 5, 10],
        #      12: [2, 5, 12], 8: [2, 0, 3, 8], 6: [2, 1, 7, 6], 9: [2, 1, 7, 9], 11: [2, 5, 12, 11], 13: [2, 5, 12, 13]},
        #  3: {3: [3], 0: [3, 0], 4: [3, 4], 8: [3, 8], 1: [3, 0, 1], 2: [3, 0, 2], 5: [3, 4, 5], 6: [3, 4, 6],
        #      11: [3, 8, 11], 13: [3, 8, 13], 7: [3, 0, 1, 7], 10: [3, 4, 5, 10], 12: [3, 4, 5, 12], 9: [3, 8, 11, 9]},
        #  4: {4: [4], 3: [4, 3], 5: [4, 5], 6: [4, 6], 0: [4, 3, 0], 8: [4, 3, 8], 2: [4, 5, 2], 10: [4, 5, 10],
        #      12: [4, 5, 12], 7: [4, 6, 7], 1: [4, 3, 0, 1], 11: [4, 3, 8, 11], 13: [4, 3, 8, 13], 9: [4, 5, 10, 9]},
        #  5: {5: [5], 2: [5, 2], 4: [5, 4], 10: [5, 10], 12: [5, 12], 0: [5, 2, 0], 1: [5, 2, 1], 3: [5, 4, 3],
        #      6: [5, 4, 6], 9: [5, 10, 9], 11: [5, 12, 11], 13: [5, 12, 13], 7: [5, 2, 1, 7], 8: [5, 4, 3, 8]},
        #  6: {6: [6], 4: [6, 4], 7: [6, 7], 3: [6, 4, 3], 5: [6, 4, 5], 1: [6, 7, 1], 9: [6, 7, 9], 0: [6, 4, 3, 0],
        #      8: [6, 4, 3, 8], 2: [6, 4, 5, 2], 10: [6, 4, 5, 10], 12: [6, 4, 5, 12], 11: [6, 7, 9, 11],
        #      13: [6, 7, 9, 13]},
        #  7: {7: [7], 1: [7, 1], 6: [7, 6], 9: [7, 9], 0: [7, 1, 0], 2: [7, 1, 2], 4: [7, 6, 4], 10: [7, 9, 10],
        #      11: [7, 9, 11], 13: [7, 9, 13], 3: [7, 1, 0, 3], 5: [7, 1, 2, 5], 8: [7, 9, 11, 8], 12: [7, 9, 11, 12]},
        #  8: {8: [8], 3: [8, 3], 11: [8, 11], 13: [8, 13], 0: [8, 3, 0], 4: [8, 3, 4], 9: [8, 11, 9], 12: [8, 11, 12],
        #      1: [8, 3, 0, 1], 2: [8, 3, 0, 2], 5: [8, 3, 4, 5], 6: [8, 3, 4, 6], 7: [8, 11, 9, 7], 10: [8, 11, 9, 10]},
        #  9: {9: [9], 7: [9, 7], 10: [9, 10], 11: [9, 11], 13: [9, 13], 1: [9, 7, 1], 6: [9, 7, 6], 5: [9, 10, 5],
        #      8: [9, 11, 8], 12: [9, 11, 12], 0: [9, 7, 1, 0], 2: [9, 7, 1, 2], 4: [9, 7, 6, 4], 3: [9, 11, 8, 3]},
        #  10: {10: [10], 5: [10, 5], 9: [10, 9], 2: [10, 5, 2], 4: [10, 5, 4], 12: [10, 5, 12], 7: [10, 9, 7],
        #       11: [10, 9, 11], 13: [10, 9, 13], 0: [10, 5, 2, 0], 1: [10, 5, 2, 1], 3: [10, 5, 4, 3], 6: [10, 5, 4, 6],
        #       8: [10, 9, 11, 8]},
        #  11: {11: [11], 8: [11, 8], 9: [11, 9], 12: [11, 12], 3: [11, 8, 3], 13: [11, 8, 13], 7: [11, 9, 7],
        #       10: [11, 9, 10], 5: [11, 12, 5], 0: [11, 8, 3, 0], 4: [11, 8, 3, 4], 1: [11, 9, 7, 1], 6: [11, 9, 7, 6],
        #       2: [11, 12, 5, 2]},
        #  12: {12: [12], 5: [12, 5], 11: [12, 11], 13: [12, 13], 2: [12, 5, 2], 4: [12, 5, 4], 10: [12, 5, 10],
        #       8: [12, 11, 8], 9: [12, 11, 9], 0: [12, 5, 2, 0], 1: [12, 5, 2, 1], 3: [12, 5, 4, 3], 6: [12, 5, 4, 6],
        #       7: [12, 11, 9, 7]},
        #  13: {13: [13], 8: [13, 8], 9: [13, 9], 12: [13, 12], 3: [13, 8, 3], 11: [13, 8, 11], 7: [13, 9, 7],
        #       10: [13, 9, 10], 5: [13, 12, 5], 0: [13, 8, 3, 0], 4: [13, 8, 3, 4], 1: [13, 9, 7, 1], 6: [13, 9, 7, 6],
        #       2: [13, 12, 5, 2]}}

        all_shortest = nx.all_pairs_dijkstra_path(self.graph)

        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s != d:
                    # 根据最短路径，取出下一跳
                    next = all_shortest[s][d][1]
                    port = self.ports[s][next]
                    routing_nodes[s][d] = next
                    routing_ports[s][d] = port
                else:
                    routing_nodes[s][d] = -1
                    routing_ports[s][d] = -1

        self.env_R = np.asarray(routing_ports)
        self.env_Rn = np.asarray(routing_nodes)

    def upd_env_R_from_R(self, routing):
        routing_nodes = np.fromstring(routing, sep=',', dtype=int)
        M = np.split(np.asarray(routing_nodes), self.ACTIVE_NODES)
        routing_nodes = np.vstack(M)

        routing_ports = np.zeros([self.ACTIVE_NODES]*2, dtype=int)

        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s != d:
                    next = routing_nodes[s][d]
                    port = self.ports[s][next]
                    routing_ports[s][d] = port
                else:
                    routing_ports[s][d] = -1

        # 下一跳的端口和节点 port and node
        self.env_R = np.asarray(routing_ports)
        self.env_Rn = np.asarray(routing_nodes)

    def upd_env_D(self, matrix):
        self.env_D = np.asarray(matrix)
        np.fill_diagonal(self.env_D, -1)

    def upd_env_L(self, number):
        self.env_L = number


    def logheader(self, easy=False):
        nice_matrix = np.chararray([self.ACTIVE_NODES]*2, itemsize=20)
        for i in range(self.ACTIVE_NODES):
            for j in range(self.ACTIVE_NODES):
                nice_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix!=b'_')])
        th = ['t' + _.decode('ascii') for _ in nice_list]
        rh = ['r' + _.decode('ascii') for _ in nice_list]
        dh = ['d' + _.decode('ascii') for _ in nice_list]
        ah = ['a' + str(_[0]) + '-' + str(_[1]) for _ in self.graph.edges()]
        header = ['counter'] + th + rh + dh + ['lost'] + ah + ['reward']
        if easy:
            header = ['counter', 'lost', 'AVG', 'MAX', 'AXM', 'GEO']
        vector_to_file(header, self.folder + WHOLELOG, 'w')


    def render(self):
        return


    def reset(self, easy=False):
        if self.counter != 0:
            return None

        self.logheader(easy)

        # routing
        # 初始化每条 link 的权重 为 0.5
        self.upd_env_W(np.full([self.a_dim], 0.50, dtype=float))

        # 初始化route的信息，包括下一跳 端口 和 node
        self.upd_env_R()
        if self.ACTUM == 'DELTA':
            # 把 port 信息写到文件里了
            vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
            # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # traffic
        # 生成流量 ？？？
        self.upd_env_T(self.tgen.generate())

        # 把 env_T 的内容写到 OMROUTING = 'Routing.txt'
        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        # 返回一个 state （初始state）
        return rl_state(self)


    # step 和 easy_step 的区别???
    def step(self, action):
        # 每 step 一步，就 +1， 方便后面写log。
        self.counter += 1

        # 采取action，更新网络权重，更新路由路径
        self.upd_env_W(action)
        self.upd_env_R()

        # write to file input for Omnet: Routing
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        # omnet 能够模拟出 delay 和 lost
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        # 将delay结果([14, 14]的matrix) 写进 csv ，然后又从 csv 里读出来，去更新env_D
        om_output = file_to_csv(self.folder + OMDELAY)
        # print('1================')
        # print(csv_to_matrix(om_output, self.ACTIVE_NODES).shape)
        # print(csv_to_matrix(om_output, self.ACTIVE_NODES))
        # print('2================')
        # print(csv_to_lost(om_output).shape)
        # print(csv_to_lost(om_output))

        # 1 == == == == == == == == delay
        # (14, 14)
        # [-1.         3.51201    3.66289    3.69545    7.50641    7.49637
        #  8.27289    7.07785    4.0411    14.993     11.2011     4.14121
        #  8.01044   11.564]
        # [1.14229 - 1.         3.47341    4.82957    8.76208    7.21359
        #  6.94054    3.57243    5.23124    6.91001   11.123      7.0854     7.69547
        #  11.6891]
        # [0.270918   1.18452 - 1.         4.04027    7.51116    3.86305
        #  8.29159    4.75594    8.35347   11.2755     7.57422   11.7512     4.12305
        #  7.8555]
        # [3.62741    7.1529     7.35245 - 1.         3.85188    4.16954
        #  4.59376    4.75284    0.343392  11.665      7.98605    0.503863
        #  4.56736    3.99884]
        # [3.86831    7.24493    3.65727    0.12021 - 1.         0.349723
        #  0.761109   0.965939   0.476662   7.96204    4.15928    7.8609
        #  0.713785   4.47978]
        # [3.52532    4.45058    3.3105     3.83133    3.68642 - 1.         4.45619
        #  9.95862    4.35592    7.45982    3.79037    7.57671    0.336358
        #  3.97568]
        # [4.038      0.349844   3.8094     0.245764   0.15913    0.504872 - 1.
        #  0.12457    3.73599    3.38039    4.28101    3.58252    0.912259
        #  3.69271]
        # [1.39263    0.224802   3.67793    3.50983    3.4978    11.0095     3.32827
        #  - 1.         3.85321    3.36299    7.10829    3.47422    7.10515    3.4046]
        # [7.17265   10.7004    14.2255     3.51309    7.28065   11.0063    13.2084
        #  9.71356 - 1.         7.60786   11.3967     0.145051   7.32933
        #  3.62536]
        # [11.2949     2.50971   10.6622    11.875     11.0362     7.46214
        #  5.67483    2.25802    0.579494 - 1.         3.77957    0.144717
        #  3.85759    0.204499]
        # [7.21107    8.11905    7.0584     7.60491    7.40109    3.69669
        #  8.20338    5.88641    4.13291    3.60885 - 1.         3.76658
        #  4.04847    3.8151]
        # [7.58004    6.19725   14.3862     3.93585   15.0035    11.2643     9.27191
        #  5.88987    0.295507   3.65358    7.42911 - 1.         0.128408
        #  3.88493]
        # [7.14599    8.05927    7.06878    7.37756    7.37176    3.69752
        #  8.23408    9.74943    3.95804    7.46998    7.46441    0.160165 - 1.
        #  3.63136]
        # [10.8382    11.7285    10.7063     3.79832   11.0339     7.25231
        #  9.52673    6.09302    0.310538   3.86918    7.56789    3.92079
        #  3.56174 - 1.]
        #
        # 2 == == == == == == == == lost
        # 326.0

        self.upd_env_D(csv_to_matrix(om_output, self.ACTIVE_NODES))

        self.upd_env_L(csv_to_lost(om_output))

        # 计算reward
        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)

        # 看写入了哪些东西
        # counter + traffic + route_node + delay + lost + weight + -reward
        log = np.concatenate(([self.counter], matrix_to_log_v(self.env_T), matrix_to_log_v(self.env_Rn), matrix_to_log_v(self.env_D), [self.env_L], matrix_to_log_v(self.env_W), [-reward]))
        vector_to_file(log, self.folder + WHOLELOG, 'a')

        # generate traffic for next iteration
        self.upd_env_T(self.tgen.generate())
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        # 返回一个新 state 和 reward
        # ？？？ 这个新state为什么是直接生产，而不是step出来的
        return new_state, reward, 0


    def easystep(self, action):
        self.counter += 1

        self.upd_env_R_from_R(action)

        # write to file input for Omnet: Routing
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        om_output = file_to_csv(self.folder + OMDELAY)
        self.upd_env_D(csv_to_matrix(om_output, self.ACTIVE_NODES))
        self.upd_env_L(csv_to_lost(om_output))

        # 计算reward
        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)
        log = np.concatenate(([self.counter], [self.env_L], [np.mean(matrix_to_rl(self.env_D))], [np.max(matrix_to_rl(self.env_D))], [(np.mean(matrix_to_rl(self.env_D)) + np.max(matrix_to_rl(self.env_D)))/2], [stats.gmean(matrix_to_rl(self.env_D))]))
        vector_to_file(log, self.folder + WHOLELOG, 'a')

        # generate traffic for next iteration
        self.upd_env_T(self.tgen.generate())
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE', 'DIR'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        return new_state, reward, 0


    def end(self):
        return
