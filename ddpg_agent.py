import numpy as np
from collections import deque

# parameters
REPLAY_BUFFER_SIZE = 10000

class DDPGAgent:
    def __init__(self, env):
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.observation_space.shape[0]

        self.replay_buffer = deque()

    def feed_forward_actor(self, state):
        # FIXME: pass
        return np.array([[1., 1., 1., 1., 1., 1., 1., 1.]])

    def add_experience(self, s_t, a_t, s_t_1, reward, done):
        self.replay_buffer.append((s_t, a_t, s_t_1, reward, done))

        if (len(self.replay_buffer) > REPLAY_BUFFER_SIZE):
            self.replay_buffer.popleft()

    def train(self):
        pass
