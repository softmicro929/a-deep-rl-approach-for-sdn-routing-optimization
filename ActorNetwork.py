"""
ActorNetwork.py
"""
__author__ = "giorgio@ac.upc.edu"
__credits__ = "https://github.com/yanpanlau"

from keras.initializations import normal, glorot_normal
from keras.activations import relu
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

from helper import selu


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, DDPG_config):
        self.HIDDEN1_UNITS = DDPG_config['HIDDEN1_UNITS']
        self.HIDDEN2_UNITS = DDPG_config['HIDDEN2_UNITS']

        self.sess = sess
        self.BATCH_SIZE = DDPG_config['BATCH_SIZE']
        self.TAU = DDPG_config['TAU']
        self.LEARNING_RATE = DDPG_config['LRA']
        self.ACTUM = DDPG_config['ACTUM']

        if self.ACTUM == 'NEW':
            self.acti = 'sigmoid'
        elif self.ACTUM == 'DELTA':
            self.acti = 'tanh'

        self.h_acti = relu
        if DDPG_config['HACTI'] == 'selu':
            self.h_acti = selu

        K.set_session(sess)

        # Now create the model: eval and target
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)

        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])

        # 最大化 取-self.action_gradient 最小化
        # 先 compute grad： d_output/d_weights * -self.action_gradient(权重)
        # 再 apply_gradients
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        # (gradient, variable) pairs
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    # 这里actor的输入有来自 critic 的 梯度 d_p/d_w, 后面用的时候还要取-
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    # 这里就是更新 target 网络， 就是用eval的权重去更新target的权重
    # 具体以 TAU 的方式，一种 soft 更新的方式， "TAU": 0.001 ===> TAU * actor_eval.weight + (1-TAU) * actor_traget.weight
    # 其实还有另外一种方式，每隔 N 个 epoch， 去复制一遍 eval 的权重给 target
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    # 创建网络，输入是state_size n*(n-1)？？？ 难道还转成了one-hot表示？, 输出是 V： action_dim 是所有的 edges
    def create_actor_network(self, state_size, action_dim):
        S = Input(shape=[state_size], name='a_S')
        h0 = Dense(self.HIDDEN1_UNITS, activation=self.h_acti, init=glorot_normal, name='a_h0')(S)
        h1 = Dense(self.HIDDEN2_UNITS, activation=self.h_acti, init=glorot_normal, name='a_h1')(h0)
        # https://github.com/fchollet/keras/issues/374
        V = Dense(action_dim, activation=self.acti, init=glorot_normal, name='a_V')(h1)
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S
