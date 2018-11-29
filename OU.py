"""
OU.py
"""
__author__ = "giorgio@ac.upc.edu"
__credits__ = ["https://github.com/yanpanlau", "https://gist.github.com/jimfleming/9a62b2f7ed047ff78e95b5398e955b9e"]

import numpy as np
from scipy.stats import norm


# https://blog.csdn.net/u013745804/article/details/78461253
# 上周在实现DDPG的过程中，发现其中用到了一个没见过的随机过程，叫做Ornstein-Uhlenbeck过程
#
# Ornstein-Uhlenbeck process 是一个随机过程，它常用来描述短期利率围绕着长期均衡利率旋转的随时间随机变动的规律，常用一个参数k描述短期利率回归长期均衡利率的速度，k假定为一个大于零的正数，k值越大，说明短期利率逼近长期均衡利率的速度越快。
#
# 而线性回归描述的是两个变量的线性相关关系，如果横轴是时间的话，那么如果x和y正相关，则y变量的值随时间的增大而逐渐增大。
#
# 请问这两者是一回事吗？
#
# 如果有人把两者说成是一回事的话，则只能说明她太无知了！


# Ornstein-Uhlenbeck Process
class OU(object):

    def __init__(self, processes, mu=0, theta=0.15, sigma=0.3):
        self.dt = 0.1
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.processes = processes
        self.state = np.ones(self.processes) * self.mu

    def reset(self):
        self.state = np.ones(self.processes) * self.mu

    def evolve(self):
        X = self.state
        dw = norm.rvs(scale=self.dt, size=self.processes)
        dx = self.theta * (self.mu - X) * self.dt + self.sigma * dw
        self.state = X + dx
        return self.state
