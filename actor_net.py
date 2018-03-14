import numpy as np

class ActorNet:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    def forward_target_net(self, state_batch):
        # FIXME: pass
        batch_size = state_batch.shape[0]
        return np.random.random([batch_size, self.num_actions])

    def forward_learned_net(self, state_batch):
        # FIXME: pass
        batch_size = state_batch.shape[0]
        return np.random.random([batch_size, self.num_actions])

    def train(self, state_batch, dQ_da_batch):
        # FIXME: pass
        pass

    def update_target_net(self):
        # FIXME: pass
        pass

