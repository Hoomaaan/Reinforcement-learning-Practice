import numpy as np
import gym

env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
	
discrete_state = get_discrete_state(env.reset())

print(discrete_state)

# q_table[discrete_state]

print(np.argmax(q_table[discrete_state]))
done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()
    # print(reward)

env.close()