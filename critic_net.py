import numpy as np

class CriticNet:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.num_states = num_states

    def forward_target_net(self, state_batch, action_batch):
        # FIXME: pass
        batch_size = state_batch.shape[0]
        return np.random.random([batch_size, 1])

    def train(self, state_batch, action_batch, target_q_batch):
        pass # FIXME:

    def compute_dQ_da(self, state_batch, action_batch):
        # FIXME] pass
        batch_size = state_batch.shape[0]
        return np.random.random([batch_size, action_batch[0].shape[0]])

    def update_target_net(self):
        # FIXME: pass
        pass

