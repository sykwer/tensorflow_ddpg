import numpy as np

class OUNoise:
    def __init__(self, num_actions, mu=0, theta=0.15, sigma=0.3):
        self.num_actions = num_actions
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def generate(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.ones(self.num_actions) * self.mu

if __name__ == "__main__":
    noise = OUNoise(3)
    outputs = []
    for _ in range(1000):
        outputs.append(noise.generate())
    import matplotlib.pyplot as plt
    plt.plot(outputs)
    plt.show()

