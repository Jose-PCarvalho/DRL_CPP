from DQNAgent import DQNAgent
import gym
import numpy as np
import matplotlib.pyplot as plt

# parameters
num_frames = 20000
memory_size = 10000
batch_size = 128
target_update = 100

# environment
env_id = "CartPole-v1"
env = gym.make(env_id)

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, memory_size, batch_size, target_update)

n_steps = 0
scores, eps_history, steps_array = [], [], []
n_episodes = 400
best_score = -np.inf
i = 0
while i < num_frames:
    done = False
    observation, info = env.reset()
    score = 0
    j = 0
    while True:
        i += 1
        j += 1
        action = agent.select_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        score += reward
        agent.store_experience(observation, action, reward, observation_, done or truncated)
        agent.learn()
        agent.update_beta(i, num_frames)
        observation = observation_
        n_steps += 1
        if done or truncated:
            scores.append(score)
            if np.mean(scores[-5:]) > best_score and j > 5:
                print(np.mean(scores[-5:]), best_score)
                agent.save_models()
                best_score = np.mean(scores[-5:])
            score = 0
            break
    steps_array.append(n_steps)
    avg_score = np.mean(scores[-n_episodes:])
    print(i)

print(scores)
agent.load_models()
env = gym.make("CartPole-v1", render_mode="human")
done = False
truncated = False
observation, info = env.reset()

score = 0
while not (done or truncated):
    action = agent.exploit_action(observation)
    observation_, reward, done, truncated, info = env.step(action)
    env.render()
    score += reward
    observation = observation_
print(score)

plt.figure(figsize=(20, 5))
plt.title('score: %s' % (np.mean(scores[-10:])))
plt.plot(scores)
plt.show()
# agent.train(num_frames)
# agent.env=gym.make(env_id,render_mode="human")
# video_folder = "videos/dqn"
# agent.test(video_folder=video_folder)
