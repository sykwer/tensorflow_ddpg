import numpy as np
import tensorflow as tf

# paramters
N_HIDDEN_1 = 200
N_HIDDEN_2 = 100
MINI_BATCH_SIZE = 100
LEARNING_RATE = 0.001
TAU = 0.01
EPSILON = 1e-6

class ActorNet:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.Session()

            # Learned network
            self.W1, self.scale1, self.beta1, self.W2, self.scale2, self.beta2, self.W3, self.B3,\
                self.actor_model, self.state_input = self.__create_graph()

            # Target network
            self.t_W1, self.t_scale1, self.t_beta1, self.t_W2, self.t_scale2, self.t_beta2, self.t_W3, self.t_B3,\
                self.t_actor_model, self.t_state_input = self.__create_graph()

            # Update accroding to DDPG theorem
            self.dQ_da_input = tf.placeholder(tf.float32, [None, self.num_actions])
            params = [self.W1, self.scale1, self.beta1, self.W2, self.scale2, self.beta2, self.W3, self.B3]
            param_grads_sum_over_batch = tf.gradients(self.actor_model, params, -self.dQ_da_input)
            param_grads = [tf.divide(param_grad, MINI_BATCH_SIZE) for param_grad in param_grads_sum_over_batch]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).apply_gradients(zip(param_grads, params))

            # Update target network making closer to learned network
            self.target_net_update_ops = [
                self.t_W1.assign(TAU * self.W1 + (1-TAU) * self.t_W1),
                self.t_W2.assign(TAU * self.W2 + (1-TAU) * self.t_W2),
                self.t_W3.assign(TAU * self.W3 + (1-TAU) * self.t_W3),
                self.t_B3.assign(TAU * self.B3 + (1-TAU) * self.t_B3),
                self.t_scale1.assign(TAU * self.scale1 + (1-TAU) * self.t_scale1),
                self.t_scale2.assign(TAU * self.scale2 + (1-TAU) * self.t_scale2),
                self.t_beta1.assign(TAU * self.beta1 + (1-TAU) * self.t_beta1),
                self.t_beta2.assign(TAU * self.beta2 + (1-TAU) * self.t_beta2),
            ]

            # Global variables have to be initialized
            self.sess.run(tf.global_variables_initializer())

            # Confirm learned/target net have the same values
            self.sess.run([
                self.t_W1.assign(self.W1),
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

        # He initialization
        W1 = tf.Variable(tf.random_uniform([self.num_states, N_HIDDEN_1], -np.sqrt(2/self.num_states), np.sqrt(2/self.num_states)))
        W2 = tf.Variable(tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -np.sqrt(2/N_HIDDEN_1), np.sqrt(2/N_HIDDEN_1)))
        W3 = tf.Variable(tf.random_uniform([N_HIDDEN_2, self.num_actions], -np.sqrt(2/N_HIDDEN_2), np.sqrt(2/N_HIDDEN_2)))
        B3 = tf.Variable(tf.zeros([self.num_actions]))

        # Batch normalization
        u1 = tf.matmul(state_input, W1)
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

        actor_model = tf.matmul(z2, W3) + B3

        return W1, scale1, beta1, W2, scale2, beta2, W3, B3, actor_model, state_input

    def forward_target_net(self, state_batch):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_state_input: state_batch})

    def forward_learned_net(self, state_batch):
        return self.sess.run(self.actor_model, feed_dict={self.state_input: state_batch})

    def train(self, state_batch, dQ_da_batch):
        self.sess.run(self.optimizer, feed_dict={self.state_input: state_batch, self.dQ_da_input: dQ_da_batch})

    def update_target_net(self):
        self.sess.run(self.target_net_update_ops)

