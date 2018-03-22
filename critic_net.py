import numpy as np
import tensorflow as tf

# paramters
N_HIDDEN_1 = 200
N_HIDDEN_2 = 100
MINI_BATCH_SIZE = 100
LEARNING_RATE = 0.001
TAU = 0.01
EPSILON = 1e-6

class CriticNet:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.num_states = num_states

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.Session()

            # Learned network
            self.state_W1, self.action_W1, self.scale1, self.beta1, self.W2, self.scale2, self.beta2, self.W3,\
                self.B3, self.critic_model, self.state_input, self.action_input = self.__create_graph()

            # Target network
            self.t_state_W1, self.t_action_W1, self.t_scale1, self.t_beta1, self.t_W2, self.t_scale2, self.t_beta2, self.t_W3,\
                self.t_B3, self.t_critic_model, self.t_state_input, self.t_action_input = self.__create_graph()

            # Supervised leaaning
            self.q_teacher = tf.placeholder(tf.float32, [None, 1])
            l2_regularizer = tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3)
            loss = tf.pow(self.critic_model - self.q_teacher, 2) / MINI_BATCH_SIZE + l2_regularizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

            # Q gradient computation
            self.dQ_da = tf.gradients(self.critic_model, self.action_input)

            # Update target network making closer to learned network
            self.target_net_update_ops = [
                self.t_state_W1.assign(TAU * self.state_W1 + (1-TAU) * self.t_state_W1),
                self.t_action_W1.assign(TAU * self.action_W1 + (1-TAU) * self.t_action_W1),
                self.t_W2.assign(TAU * self.W2 + (1-TAU) * self.t_W2),
                self.t_W3.assign(TAU * self.W3 + (1-TAU) * self.t_W3),
                self.t_B3.assign(TAU * self.B3 + (1-TAU) * self.t_B3),
                self.t_scale1.assign(TAU * self.scale1 * (1-TAU) * self.t_scale1),
                self.t_scale2.assign(TAU * self.scale2 * (1-TAU) * self.t_scale2),
                self.t_beta1.assign(TAU * self.beta1 * (1-TAU) * self.t_beta1),
                self.t_beta2.assign(TAU * self.beta2 * (1-TAU) * self.t_beta2),
            ]

            # Global variables have to be initialized
            self.sess.run(tf.global_variables_initializer())

            # Confirm learned/target net have the same values
            self.sess.run([
                self.t_state_W1.assign(self.state_W1),
                self.t_action_W1.assign(self.action_W1),
                self.t_W2.assign(self.W2),
                self.t_W3.assign(self.W3),
                self.t_B3.assign(self.B3),
                self.t_scale1.assign(self.scale1),
                self.t_scale2.assign(self.scale2),
                self.t_beta1.assign(self.beta1),
                self.t_beta2.assign(self.beta2),
            ])

    def __create_graph(self):
        state_input = tf.placeholder(tf.float32, {None, self.num_states})
        action_input = tf.placeholder(tf.float32, [None, self.num_actions])

        # He initialization
        state_W1 = tf.Variable(tf.random_uniform([self.num_states, N_HIDDEN_1],\
            -np.sqrt(2/(self.num_states + self.num_actions)), np.sqrt(2/(self.num_states + self.num_actions))))
        action_W1 = tf.Variable(tf.random_uniform([self.num_actions, N_HIDDEN_1],\
            -np.sqrt(2/(self.num_states + self.num_actions)), np.sqrt(2/(self.num_states + self.num_actions))))
        W2 = tf.Variable(tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -np.sqrt(2/N_HIDDEN_1), np.sqrt(2/N_HIDDEN_1)))
        W3 = tf.Variable(tf.random_uniform([N_HIDDEN_2, 1], -np.sqrt(2/N_HIDDEN_2), np.sqrt(2/N_HIDDEN_2)))
        B3 = tf.Variable(tf.zeros([1]))

        # Batch normalization
        u1 = tf.matmul(state_input, state_W1) + tf.matmul(action_input, action_W1)
        batch_mean1, batch_var1 = tf.nn.moments(u1, [0])
        scale1 = tf.Variable(tf.ones([N_HIDDEN_1]))
        beta1 = tf.Variable(tf.zeros([N_HIDDEN_1]))
        bn1 = tf.nn.batch_normalization(u1, batch_mean1, batch_var1, beta1, scale1, EPSILON)
        z1 = tf.nn.relu(bn1)

        u2 = tf.matmul(z1, W2)
        batch_mean2, batch_var2 = tf.nn.moments(u2, [0])
        scale2 = tf.Variable(tf.ones([N_HIDDEN_2]))
        beta2 = tf.Variable(tf.zeros([N_HIDDEN_2]))
        bn2 = tf.nn.batch_normalization(u2, batch_mean2, batch_var2, beta2, scale2, EPSILON)
        z2 = tf.nn.relu(bn2)

        critic_model = tf.matmul(z2, W3) + B3

        return state_W1, action_W1, scale1, beta1, W2, scale2, beta2, W3, B3, critic_model, state_input, action_input

    def forward_target_net(self, state_batch, action_batch):
        return self.sess.run(self.t_critic_model,\
                feed_dict={self.t_state_input: state_batch, self.t_action_input: action_batch})

    def train(self, state_batch, action_batch, target_q_batch):
        self.sess.run(self.optimizer, feed_dict={self.state_input: state_batch, self.action_input: action_batch,\
                self.q_teacher: target_q_batch})

    def compute_dQ_da(self, state_batch, action_batch):
        return self.sess.run(self.dQ_da, feed_dict={self.state_input: state_batch, self.action_input: action_batch})[0]

    def update_target_net(self):
        self.sess.run(self.target_net_update_ops)

