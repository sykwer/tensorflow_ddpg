import numpy as np
import gym

from ddpg_agent import DDPGAgent, MINI_BATCH_SIZE
from ou_noise import OUNoise

# Select aigym envrironment name.
# You have to select env whose state and action spaces are continuous.
ENV_NAME = "Ant-v2"

# parameters
episodes_num = 20000
is_movie_on = True

def main():
    # Instanciate specified environment.
    env = gym.make(ENV_NAME)

    # Confirm that state and action spaces are continuous
    assert isinstance(env.observation_space, gym.spaces.Box), "State space must be continuous!"
    assert isinstance(env.action_space, gym.spaces.Box), "Action space must be continuous!"

    # Get environment specs
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    steps_limit = env.spec.timestep_limit

    # Print specs
    print("-----------Env specs (%s)------------" % ENV_NAME)
    print("Number of states: %d" % num_states)
    print("Number of actions: %d" % num_actions)
    print("Limit of steps per episode: %d" % steps_limit)
    print("-----------------------------------------")

    # Instanciate reinforcement learning agent which contains Actor/Critic DNN.
    agent = DDPGAgent(env)
    # Exploration noise generator which uses Ornstein-Uhlenbeck process.
    noise = OUNoise(num_actions)

    for i in range(episodes_num):
        print("Start episodes no: %d" % i)
        reward_per_episode = 0
        observation = env.reset()

        for j in range(steps_limit):
            if is_movie_on: env.render()

            # Select action off-policy
            state = observation
            action = agent.feed_forward_actor(np.reshape(state, [1, num_states]))
            action = action[0] + noise.generate()

            # Throw action to environment
            observation, reward, done, info = env.step(action)

            # For replay buffer. (s_t, a_t, s_t+1, r)
            agent.add_experience(state, action, observation, reward, done)

            # Train actor/critic network
            if j > MINI_BATCH_SIZE: agent.train()

            reward_per_episode += reward

            if (done or j == steps_limit -1):
                print("--------Episode %d--------" % i)
                print("Steps count: %d" % j)
                print("Total reward: %d" % reward_per_episode)

                noise.reset()

                with open("reward_log.csv", "a") as f:
                    f.write("%d,%f\n" % (i, reward_per_episode))

                break

if __name__ == "__main__":
    main()
