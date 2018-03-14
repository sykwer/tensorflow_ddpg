import numpy as np

class OUNoise:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def generate(self):
        # FIXME: pass
        return np.random.random(self.num_actions)

    def reset(self):
        # FIXME: pass
        pass

